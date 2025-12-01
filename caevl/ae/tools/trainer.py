import torch
import torch.nn.functional as F

from tqdm import tqdm
import os
import numpy as np

from caevl.ae.tools.early_stopping import EarlyStopping


def make_dir(path):
    """Create a directory (and all its parents) at the given path if not already existing.

    Parameters
    ----------
    path : str
        Path to create the directory to.
    """
    
    if not os.path.isdir(path):
        os.makedirs(path)


class AutoEncoder_Trainer:
    """
    Trainer class for an autoencoder model.

    Parameters
    ----------
    model : torch.nn.Module
        The autoencoder model to train.
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    val_loader : torch.utils.data.DataLoader
        DataLoader for the validation dataset.
    early_stopping : EarlyStopping, optional
        EarlyStopping object to stop training early. Default is None.
    """
    
    def __init__(self, model, train_loader, val_loader, early_stopping=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.early_stopping = early_stopping
        
    
    def validate(self, loader=None):
        """
        Validate the model on the given dataset.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader, optional
            DataLoader for the validation dataset. Default is None, which will use val_loader.

        Returns
        -------
        float
            Average validation loss.
        """

        self.model.eval()
        loss = 0.

        if loader is None:
            loader = self.val_loader
        
        nb_samples = 0
        for inputs, coordinates, _ in loader:
            batch_size = inputs.shape[0]
            inputs = inputs.to(self.model.device)
            
            outputs, embeddings = self.model(inputs)
            loss_batch = batch_size * self.model.loss_function(inputs, outputs, embeddings, coordinates)
            loss += loss_batch.item()
            nb_samples += batch_size
            
        loss /= nb_samples
        return loss

    def train_epoch(self, train_loader, epoch):
        """
        Train the model for one epoch.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            DataLoader for the training dataset.
        epoch : int
            Current epoch number.

        Returns
        -------
        float
            Average training loss for the epoch.
        """

        self.model.train()
        train_loss = 0.
        nb_samples = 0
        
        with tqdm(range(len(train_loader)), unit='batch') as bar:
            for inputs, coordinates, _ in train_loader:
                bar.set_description(f'Epoch {epoch}')
                
                batch_size = inputs.shape[0]
                inputs = inputs.to(self.model.device)

                # Forward pass
                outputs, embeddings = self.model(inputs)
                loss = self.model.loss_function(inputs, outputs, embeddings, coordinates)
                
                # Backward pass and optimization
                self.model.optimizer.zero_grad()
                loss.backward()
                self.model.optimizer.step()
                
                train_loss += batch_size * loss.item()
                nb_samples += batch_size

                bar.set_postfix(train_loss=train_loss/nb_samples)
                bar.update(1)
            
            train_loss /= nb_samples
        return train_loss
    

    def save_checkpoint(self, save_dir, epoch, save_optimizer=False):
        """
        Save the model and optimizer state dictionaries.

        Parameters
        ----------
        save_dir : str
            Directory to save the checkpoint.
        epoch : int
            Current epoch number.
        save_optimizer : bool, optional
            Flag to indicate if the optimizer should be saved. Default is False.
        """
        
        make_dir(os.path.join(save_dir, f'epoch{epoch}'))
        torch.save(self.model.state_dict(), os.path.join(save_dir, f'epoch{epoch}', f'model_epoch{epoch}.pth'))
        if save_optimizer:
            torch.save(self.model.optimizer.state_dict(), os.path.join(save_dir, f'epoch{epoch}', f'optimizer_epoch{epoch}.pth'))
        if self.model.scheduler is not None:
            torch.save(self.model.scheduler.state_dict(), os.path.join(save_dir, f'epoch{epoch}', f'scheduler_epoch{epoch}.pth'))


    def train(self, num_epochs, model_save_path=None,
              dir_save_losses=None, force_save=False, save_checkpoint=0, save_optimizer=False,
              resume_training=False):
        """
        Train the model for the specified number of epochs.

        Parameters
        ----------
        num_epochs : int
            Number of epochs to train the model.
        model_save_path : str, optional
            Path to save the model. If None, the weights are not saved. Default is None.
        dir_save_losses : str, optional
            Directory to save the training and validation losses. Default is None.
        force_save : bool, optional
            Flag to force saving the model at every epoch, even if val_loss is going up. Default is False.
        save_checkpoint : int, optional
            Interval to save checkpoints. Default is 0.
        save_optimizer : bool, optional
            Flag to indicate if the optimizer should be saved. Default is False.
        resume_training : bool, optional
            Flag to indicate if training should be resumed from a previous training. Default is False.

        Returns
        -------
        train_losses : list
            List of training losses.
        val_losses : list
            List of validation losses.
        """
        
        def load_list(path, resume_training):
            try:
                assert resume_training
                return list(np.load(path))
            except:
                return []
            
        train_losses = load_list(os.path.join(dir_save_losses, 'train_losses.npy'), resume_training)
        val_losses = load_list(os.path.join(dir_save_losses, 'val_losses.npy'), resume_training)

        if self.early_stopping is None:
            self.early_stopping = EarlyStopping(patience=num_epochs, min_delta=0)
            
        if model_save_path is not None:
            model_save_dir, _ = os.path.split(model_save_path)
            make_dir(model_save_dir)
            
        if dir_save_losses is not None:
            make_dir(dir_save_losses)
            

        for epoch in range(len(train_losses), num_epochs):    
            train_loss = self.train_epoch(self.train_loader, epoch)
            train_losses.append(train_loss)

            if len(self.val_loader.dataset) > 0:
                val_loss = self.validate(self.val_loader)
                val_losses.append(val_loss)
            else:
                val_loss = -1

            current_learning_rate = self.model.optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch}, cur_lr: {current_learning_rate}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, '
                    f'patience: {self.early_stopping.counter}/{self.early_stopping.patience}')
                                
            if self.model.scheduler is not None and val_loss >= 0:
                self.model.scheduler.step(val_loss)
            
            if dir_save_losses is not None:
                np.save(os.path.join(dir_save_losses, 'train_losses.npy'), train_losses)
                np.save(os.path.join(dir_save_losses, 'val_losses.npy'), val_losses)
                    
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print("### Early stopping ###")
                break
            
            if save_checkpoint > 0 and epoch % save_checkpoint == 0:
                save_dir, _ = os.path.split(model_save_path)
                self.save_checkpoint(save_dir, epoch, save_optimizer)
            
            elif model_save_path is not None and (self.early_stopping.counter == 0 or force_save):
                torch.save(self.model.state_dict(), model_save_path)
                torch.save(self.model.optimizer.state_dict(), os.path.join(model_save_dir, 'optimizer.pth'))
                torch.save(self.model.scheduler.state_dict(), os.path.join(model_save_dir, 'scheduler.pth'))
        
        return train_losses, val_losses
