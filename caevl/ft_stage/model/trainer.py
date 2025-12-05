import torch
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


class FtStageTrainer:
    """
    A trainer class for the finetuning stage of the model, handling training, validation, and localization.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    val_loader : torch.utils.data.DataLoader
        DataLoader for the validation dataset.
    optimizer : torch.optim.Optimizer
        Optimizer for training the model.
    scheduler : torch.optim.lr_scheduler, optional
        Learning rate scheduler. The default is None.
    early_stopping : EarlyStopping, optional
        Early stopping object. The default is None.
    """

    def __init__(self, model, train_loader, val_loader,
                 optimizer, scheduler, early_stopping=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping

    def validate(self, loader=None):
        """
        Validate the model on the given loader.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader, optional
            DataLoader to use for validation. The default is None.
            If None, the val_loader of teh Trainer class is used.

        Returns
        -------
        float
            The average validation loss.
        """

        self.model.eval_mode()
        loss = 0.

        if loader is None:
            loader = self.val_loader

        nb_samples = 0
        for elements in iter(loader):
            inputs = elements[0]
            transformed_inputs = elements[1]

            batch_size = inputs.shape[0]
            inputs = inputs.to(self.model.device)
            transformed_inputs = transformed_inputs.to(self.model.device)

            loss_batch = batch_size * self.model(inputs, transformed_inputs)
            loss += loss_batch.mean().item()
            nb_samples += batch_size

        if nb_samples > 0:
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
            The average training loss for the epoch.
        """

        self.model.train_mode()
        self.model.train()
        train_loss = 0.
        nb_samples = 0
        with torch.enable_grad(), tqdm(range(len(train_loader)), unit='batch') as bar:
            for elements in iter(train_loader):
                inputs = elements[0]
                transformed_inputs = elements[1]
                locations = None if len(elements) == 4 else elements[4]

                bar.set_description(f'Epoch {epoch}')
                batch_size = inputs.shape[0]

                inputs = inputs.to(self.model.device)
                transformed_inputs = transformed_inputs.to(self.model.device)
                if locations is not None: locations = locations.to(self.model.device)

                losses = self.model(inputs, transformed_inputs, locations=locations)
                loss = losses.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += batch_size * loss.item()
                nb_samples += batch_size

                bar.set_postfix(train_loss=train_loss / nb_samples)
                bar.update(1)

            if nb_samples > 0:
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
        torch.save(self.model.state_dict(), os.path.join(save_dir,
                                                         f'epoch{epoch}', f'model_epoch{epoch}.pth'))
        if save_optimizer:
            torch.save(self.optimizer.state_dict(), os.path.join(save_dir,
                                                                 f'epoch{epoch}', f'optimizer_epoch{epoch}.pth'))
        if self.scheduler is not None:
            torch.save(self.scheduler.state_dict(), os.path.join(save_dir,
                                                                 f'epoch{epoch}', f'scheduler_epoch{epoch}.pth'))

    def train_(self, num_epochs, model_save_path=None,
               dir_save_losses=None, force_save=False, save_checkpoint=0, save_optimizer=False):
        """
        Train the model for a specified number of epochs.

        Parameters
        ----------
        num_epochs : int
            Number of epochs to train the model.
        model_save_path : str, optional
            Path to save the model. If None, the weights are not saved. Default is None.
        dir_save_losses : str, optional
            Directory to save the training and validation losses. The default is None.
        force_save : bool, optional
            Whether to force save the model. The default is False.
        save_checkpoint : int, optional
            Frequency of saving checkpoints. The default is 0.
        save_optimizer : bool, optional
            Whether to save the optimizer state. The default is False.

        Returns
        -------
        train_losses : list of float
            List of training losses for each epoch.
        val_losses : list of float
            List of validation losses for each epoch.
        """

        train_losses, val_losses = [], []

        if self.early_stopping is None:
            self.early_stopping = EarlyStopping(patience=num_epochs, min_delta=0)

        if model_save_path is not None:
            model_save_dir, _ = os.path.split(model_save_path)
            make_dir(model_save_dir)

        if dir_save_losses is not None:
            make_dir(dir_save_losses)

        for epoch in range(num_epochs):

            train_loss = self.train_epoch(self.train_loader, epoch)
            train_losses.append(train_loss)

            if len(self.val_loader.dataset) > 0:
                val_loss = self.validate(self.val_loader)
            else:
                val_loss = -1
            val_losses.append(val_loss)

            current_learning_rate = self.optimizer.param_groups[0]['lr']

            print(f'Epoch {epoch}, cur_lr: {current_learning_rate}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, '
                  f'patience: {self.early_stopping.counter}/{self.early_stopping.patience}')

            if dir_save_losses is not None:
                np.save(os.path.join(dir_save_losses, 'train_losses.npy'), train_losses)
                np.save(os.path.join(dir_save_losses, 'val_losses.npy'), val_losses)

            validation_val = val_loss if val_loss >= 0 else None

            if self.scheduler is not None and validation_val is not None:
                self.scheduler.step(validation_val)

            self.early_stopping(validation_val)
            if self.early_stopping.early_stop:
                print("### Early stopping ###")
                break

            if save_checkpoint > 0 and epoch % save_checkpoint == 0:
                save_dir, _ = os.path.split(model_save_path)
                self.save_checkpoint(save_dir, epoch, save_optimizer)

            if model_save_path is not None and (self.early_stopping.counter == 0 or force_save):
                save_dir, _ = os.path.split(model_save_path)
                torch.save(self.model.state_dict(), model_save_path)
                torch.save(self.optimizer.state_dict(), os.path.join(save_dir, 'optimizer.pth'))
                if self.scheduler is not None:
                    torch.save(self.scheduler.state_dict(), os.path.join(save_dir, 'scheduler.pth'))

        return train_losses, val_losses
