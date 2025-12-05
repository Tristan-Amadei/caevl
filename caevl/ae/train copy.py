import argparse
import yaml
from glob import glob
import os
import pickle
import time
import numpy as np

from pathlib import Path
import sys
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

import torch
from torch.utils.data import DataLoader

from caevl.data_generation.dataset.custom_dataset import CustomDataset
from caevl.ae.autoencoder.ae_model import AutoEncoder
from caevl.ae.tools.trainer import AutoEncoder_Trainer
from caevl.ae.tools.early_stopping import EarlyStopping

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class BatchShuffleSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __len__(self):
        return int(len(self.dataset)/self.batch_size)+1

    def __iter__(self):
        n = len(self.dataset)
        indices = np.array(range(n))
        if self.shuffle:
            np.random.shuffle(indices)
        for i in range(0, n, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            if len(batch_indices) <= 1:
                batch_indices = np.insert(batch_indices, 0, indices[0])
            yield batch_indices
            
            
def count_parameters(model, nb_input_channels, input_size, batch_size, device, initialize):
    if initialize:
        dummy_batch = torch.randn((batch_size, nb_input_channels, *input_size)).to(device)
        dummy_output = model(dummy_batch) # initializes the model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    
    # Format the time components into a readable string using f-strings
    time_parts = [f"{days}d" if days > 0 else "",
                  f"{hours}h" if hours > 0 else "",
                  f"{minutes}m" if minutes > 0 else "",
                  f"{seconds:.02f}s" if seconds > 0 else ""]
    
    # Join non-empty components with a space
    return ' '.join(part for part in time_parts if part)


def make_dir(path):
    """Create a directory (and all its parents) at the given path if not already existing.

    Parameters
    ----------
    path : str
        Path to create the directory to.
    """
    
    if not os.path.exists(path) or not os.path.isdir(path):
        os.makedirs(path)

def list_images(dir_):
    images =  glob(os.path.join(dir_, '*tif*'))
    images += glob(os.path.join(dir_, '*jpg'))
    images += glob(os.path.join(dir_, '*JPG'))
    images += glob(os.path.join(dir_, '*png'))
    images += glob(os.path.join(dir_, '*PNG'))
    return images


if __name__ == '__main__':
    start = time.time()
    
    parser = argparse.ArgumentParser(description='Instantiate and train an AutoEncoder')
    parser.add_argument('-c', '--config',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file')
    parser.add_argument('--load-weights',
                        dest="load_weights",
                        action=argparse.BooleanOptionalAction,
                        help = 'Whether to load weights of this model already previously trained, for instance if a training session was stopped too early.',
                        default=False)
    args = parser.parse_args()
    
    path = args.filename
    if not path.endswith('.yml'):
        path += '.yml'
        
    with open(path, 'r') as config_file:
        config = yaml.safe_load(config_file)
        
    ### SET SEED
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    
    image_size = tuple(config['image_size'])
    
    ### AUTOENCODER ###
    
    nb_input_channels = config['AutoEncoder']['nb_input_channels']
    nb_channels_fst_feature_map = config['AutoEncoder']['nb_channels_fst_feature_map']
    latent_dim = config['AutoEncoder']['latent_dim']
    learning_rate = float(config['AutoEncoder']['learning_rate'])
    scheduler = config['AutoEncoder']['scheduler']
    loss_name = config['AutoEncoder']['loss_name']
    device = config['AutoEncoder']['device']
    alpha = config['AutoEncoder']['alpha']
    flag_perceptual_loss = config['AutoEncoder']['flag_perceptual_loss']
    canny_edges = config['AutoEncoder'].get('canny_edges', False)
    scalar_divide_channels_fst_feature_map_for_decoder = config['AutoEncoder'].get('scalar_divide_channels_fst_feature_map_for_decoder', 1)
    encoder_only = config['AutoEncoder'].get('encoder_only', False)
    softmax = config['AutoEncoder'].get('softmax', False)
    sigmoid = config['AutoEncoder'].get('sigmoid', False)
    use_cbam = config['AutoEncoder'].get('use_cbam', False)
    
    print(f'Using loss: {loss_name}. Using perceptual loss: {flag_perceptual_loss}')
    print(f'Using Canny edges: {canny_edges}.')
    
    autoencoder = AutoEncoder(input_dimensions=image_size,
                              nb_input_channels=nb_input_channels,
                              nb_channels_fst_feature_map=nb_channels_fst_feature_map,
                              latent_dim=latent_dim,
                              learning_rate=learning_rate,
                              scheduler=scheduler,
                              loss_name=loss_name,
                              device=device,
                              alpha=alpha,
                              flag_perceptual_loss=flag_perceptual_loss,
                              canny_edges=canny_edges,
                              scalar_divide_channels_fst_feature_map_for_decoder=scalar_divide_channels_fst_feature_map_for_decoder,
                              encoder_only=encoder_only,
                              softmax=softmax,
                              sigmoid=sigmoid,
                              use_cbam=use_cbam)
    autoencoder = autoencoder.to(autoencoder.device)
    
    batch_size = int(config['trainer']['batch_size'])
    nb_params = count_parameters(autoencoder, nb_input_channels, image_size, batch_size, autoencoder.device, initialize=True)
    nb_params_encoder = count_parameters(autoencoder.encoder, nb_input_channels, image_size, batch_size, autoencoder.device, initialize=False)
    nb_params_decoder = count_parameters(autoencoder.decoder, nb_input_channels, image_size, batch_size, autoencoder.device, initialize=False)
    print(f'Autoencoder has {nb_params:,} parameters to train, with encoder having {nb_params_encoder:,} parameters and decoder having {nb_params_decoder:,} parameters.')
    if flag_perceptual_loss and use_cbam:
        nb_params_perceptual = sum(p.numel() for p in autoencoder.perceptual_loss.cbam_bottleneck.parameters())
        print(f'Perceptual loss has {nb_params_perceptual:,} parameters to train.')
    
    print(f'Training on {autoencoder.device}.')
    model_save_path = config['trainer']['model_save_path']
    save_dir = os.path.split(model_save_path)[0]
    make_dir(save_dir)
    
    yaml_save_name = os.path.join(save_dir, 'train_config.yml')
    with open(yaml_save_name, 'w') as yml_file:
        yaml.dump(config, yml_file)
        
    try:
        assert args.load_weights
        autoencoder.load_state_dict(torch.load(model_save_path))
        path_optimizer = os.path.join(save_dir, 'optimizer.pth')
        if os.path.exists(path_optimizer): autoencoder.optimizer.load_state_dict(torch.load(path_optimizer))
        path_scheduler = os.path.join(save_dir, 'scheduler.pth')
        if os.path.exists(path_scheduler): autoencoder.scheduler.load_state_dict(torch.load(path_scheduler))
        print('Resuming training.')
    except:
        args.load_weights = False # either it is already False or there was a problem loading the model
    
    #### TRAINING ####
    ### DATASETS ###
    images_directory = config['Dataset']['images_directory']
    val_images_directory = config['Dataset']['val_images_directory']
    use_PIL = config['Dataset']['use_PIL']
    apply_transform = config['Dataset']['apply_transform']
    apply_normalization = config['Dataset']['apply_normalization']
    max_workers = config['Dataset']['max_workers']
    mean = config['Dataset'].get('mean')
    std = config['Dataset'].get('std')
    grayscale = config['Dataset'].get('grayscale', True)
    
    path_dict_coordinates = config['Dataset']['path_dict_coordinates']
    
    train_image_paths = list_images(images_directory)       
    train_image_paths.sort()
    
    with open(path_dict_coordinates, 'rb') as f:
        dict_coordinates = pickle.load(f)

    val_image_paths = list_images(val_images_directory)
    val_image_paths.sort()
    
    train_dataset = CustomDataset(image_paths=train_image_paths[:100],
                               dict_coordinates=dict_coordinates,
                               use_PIL=use_PIL,
                               apply_transform=apply_transform,
                               apply_normalization=apply_normalization,
                               output_size=image_size,
                               mean=mean,
                               std=std,
                               canny_edges=canny_edges,
                               return_index=True,
                               grayscale=grayscale)
    
    val_dataset = CustomDataset(image_paths=val_image_paths,
                             dict_coordinates=dict_coordinates,
                             use_PIL=use_PIL,
                             apply_transform=False,
                             apply_normalization=apply_normalization,
                             output_size=image_size,
                             mean=mean,
                             std=std,
                             canny_edges=canny_edges,
                             return_index=True,
                             grayscale=grayscale)
    
    if config['Dataset'].get('transform') is not None:
        rot_degrees = config['Dataset']['transform'].get('rot_degrees', 30)
        translate = config['Dataset']['transform'].get('translate', (20, 20))
        blur_kernel_size = config['Dataset']['transform'].get('blur_kernel_size', 5)
        brightness_factor = config['Dataset']['transform'].get('brightness_factor', 1.)
        contrast_factor = config['Dataset']['transform'].get('contrast_factor', 1.)
        noise_mean = config['Dataset']['transform'].get('noise_mean', 0)
        noise_std = config['Dataset']['transform'].get('noise_std', 0.1)
        zoom_in_prob = config['Dataset']['transform'].get('zoom_in_prob', 0.8)
        blur_prob = config['Dataset']['transform'].get('blur_prob', 0.5)
        brightness_prob = config['Dataset']['transform'].get('brightness_prob', 0.5)
        contrast_prob = config['Dataset']['transform'].get('contrast_prob', 0.5)
        noise_prob = config['Dataset']['transform'].get('noise_prob', 0.25)
        vignetting_prob = config['Dataset']['transform'].get('vignetting_prob', 0.9)
        masking_prob = config['Dataset']['transform'].get('masking_prob', 1)
        vignetting_val = config['Dataset']['transform'].get('vignetting_val', 70)
        
        train_dataset.set_transform(rot_degrees=rot_degrees, translate=translate,
                                    blur_kernel_size=blur_kernel_size, brightness_factor=brightness_factor,
                                    contrast_factor=contrast_factor, noise_mean=noise_mean,
                                    noise_std=noise_std, zoom_in_prob=zoom_in_prob,
                                    blur_prob=blur_prob, brightness_prob=brightness_prob,
                                    contrast_prob=contrast_prob, noise_prob=noise_prob,
                                    vignetting_prob=vignetting_prob, masking_prob=masking_prob, vignetting_val=vignetting_val)
        
        print(train_dataset.transform)
                               
    ### TRAINER ###
    batch_size = int(config['trainer']['batch_size'])
    num_workers_loader = config['trainer']['num_workers_loader']
    print(f'Batch size: {batch_size}, num_workers loader: {num_workers_loader}')
    
    train_loader = DataLoader(train_dataset, 
                              batch_sampler=BatchShuffleSampler(train_dataset, batch_size, True), 
                              num_workers=num_workers_loader, persistent_workers=False)
    val_loader = DataLoader(val_dataset, 
                            batch_sampler=BatchShuffleSampler(val_dataset, batch_size, False), 
                            num_workers=num_workers_loader, persistent_workers=False)
    
    
    num_epochs = config['trainer']['num_epochs']
    patience = config['trainer']['patience']
    losses_save_dir_path = config['trainer']['losses_save_dir_path']
    force_save = config['trainer']['force_save']
    make_dir(losses_save_dir_path)
    
    ### TRAIN WHOLE MODEL ###
    
    early_stopping = None if patience is None else EarlyStopping(patience=patience)
    save_checkpoint = config['trainer'].get('save_checkpoint', 0)
    save_optimizer = config['trainer'].get('save_optimizer', False)
    
    trainer = AutoEncoder_Trainer(model=autoencoder,
                                  train_loader=train_loader,
                                  val_loader=val_loader,
                                  early_stopping=early_stopping)
    
    print(f'Starting training with {num_epochs} epochs, with a patience of {patience}.')

    train_losses, val_losses = trainer.train(num_epochs, 
                  model_save_path=model_save_path, 
                  dir_save_losses=losses_save_dir_path,
                  force_save=force_save,
                  save_checkpoint=save_checkpoint,
                  save_optimizer=save_optimizer,
                  resume_training=args.load_weights)
    
    
    np.save(os.path.join(losses_save_dir_path, 'train_losses.npy'), train_losses)
    np.save(os.path.join(losses_save_dir_path, 'val_losses.npy'), val_losses)
        
                  
    stop = time.time()
    time_spent = format_time(stop - start)
    print(f'It took {time_spent} to run.')