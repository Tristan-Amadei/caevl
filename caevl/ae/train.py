import argparse
import yaml
import os
import time
import numpy as np

from pathlib import Path
import sys
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

import torch

from caevl.data_generation.dataset.load_dataset import load_dataset
from caevl.ae.tools.load_ae import load_ae
from caevl.ae.tools.trainer import AutoEncoder_Trainer
from caevl.ae.tools.early_stopping import EarlyStopping

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def count_parameters(model, nb_input_channels, input_size, batch_size, device, initialize):
    if initialize:
        dummy_batch = torch.randn((batch_size, nb_input_channels, *input_size)).to(device)
        _ = model(dummy_batch)  # initializes the model
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


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser(description='Instantiate and train an AutoEncoder')
    parser.add_argument('-c', '--config',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file')
    parser.add_argument('--load-weights',
                        dest="load_weights",
                        action=argparse.BooleanOptionalAction,
                        help='Whether to load weights of this model already previously trained, '
                             'for instance if a training session was stopped too early.',
                        default=False)
    args = parser.parse_args()

    path = args.filename
    if not path.endswith('.yml'):
        path += '.yml'

    with open(path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    ### SET SEED
    if config.get('seed') is not None:
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed(config['seed'])

    image_size = tuple(config['image_size'])

    ### AUTOENCODER ###

    autoencoder = load_ae(config)
    autoencoder = autoencoder.to(autoencoder.device)

    nb_input_channels = config['AutoEncoder']['nb_input_channels']
    flag_perceptual_loss = config['AutoEncoder']['flag_perceptual_loss']
    use_cbam = config['AutoEncoder'].get('use_cbam', False)

    batch_size = int(config['trainer']['batch_size'])
    nb_params = count_parameters(autoencoder, nb_input_channels, image_size, batch_size, autoencoder.device, initialize=True)
    nb_params_encoder = count_parameters(autoencoder.encoder, nb_input_channels, image_size, batch_size,
                                         autoencoder.device, initialize=False)
    nb_params_decoder = count_parameters(autoencoder.decoder, nb_input_channels, image_size, batch_size,
                                         autoencoder.device, initialize=False)
    print(f'Autoencoder has {nb_params:,} parameters to train, with encoder having {nb_params_encoder:,} '
          f'parameters and decoder having {nb_params_decoder:,} parameters.')
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
        args.load_weights = False  # either it is already False or there was a problem loading the model

    #### TRAINING ####
    ### DATASETS ###

    canny_edges = config['AutoEncoder'].get('canny_edges', False)
    train_loader, val_loader = load_dataset(config, canny_edges)

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
