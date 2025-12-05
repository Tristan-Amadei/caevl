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
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from caevl.data_generation.dataset.load_dataset import load_dataset
from caevl.ae.tools.early_stopping import EarlyStopping
from caevl.ft_stage.model.encoder import CaevlFT
from caevl.ft_stage.model.projector import Projector
from caevl.ft_stage.model.trainer import FtStageTrainer
from caevl.ft_stage.utils import load_ae_backbone, get_local_projector

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')


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
    if not os.path.exists(path) or not os.path.isdir(path):
        os.makedirs(path)


if __name__ == '__main__':

    mp.set_start_method('spawn', force=True)

    start = time.time()

    parser = argparse.ArgumentParser(description='Finetune the encoder')
    parser.add_argument('-c', '--config',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file')
    parser.add_argument('--load-weights',
                        dest="load_weights",
                        action=argparse.BooleanOptionalAction,
                        help='Whether to load weights of this model already previously trained, for instance '
                             'if a training session was stopped too early.',
                        default=False)
    args = parser.parse_args()

    path = args.filename
    if not path.endswith('.yml'):
        path += '.yml'

    with open(path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    image_size = tuple(config['image_size'])

    model_save_path = config['trainer']['model_save_path']
    losses_save_dir_path = config['trainer']['losses_save_dir_path']
    make_dir(losses_save_dir_path)

    save_dir = os.path.split(model_save_path)[0]
    yaml_save_name = os.path.join(save_dir, 'train_config.yml')
    with open(yaml_save_name, 'w') as yml_file:
        yaml.dump(config, yml_file)

    ### SET SEED
    if config.get('seed') is not None:
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed(config['seed'])

    #### TRAINING ####
    ### DATASETS ###

    canny_edges = config['Dataset'].get('canny_edges', False)
    train_loader, val_loader = load_dataset(config, canny_edges, return_both_images=True)

    ### TRAINER ###
    num_epochs = config['trainer']['num_epochs']
    patience = config['trainer']['patience']
    early_stopping = None if patience is None else EarlyStopping(patience=patience)

    device = torch.device(config['architecture'].get('device', 'cpu'))
    load_weights_backbone = config['architecture'].get('load_weights_backbone', True)

    dir_model = config['architecture'].get('dir_model', 'AutoEncoder')
    inverse_pyramid = False
    norm_layer = 'layer_norm'
    backbone = config['architecture']['backbone']

    path_to_model_config = os.path.join(f'caevl/models/trained/{dir_model}/{backbone}', 'train_config.yml')

    # save backbone config file
    with open(path_to_model_config, 'r') as backbone_cfg:
        backbone_config = yaml.safe_load(backbone_cfg)
    name_backbone_config = os.path.join(save_dir, 'backbone_config.yml')
    with open(name_backbone_config, 'w') as yml_file:
        yaml.dump(backbone_config, yml_file)

    if dir_model == 'AutoEncoder':
        autoencoder, norm_layer = load_ae_backbone(path_to_model_config, device=device, load_weights=load_weights_backbone)
        backbone = autoencoder.encoder

    alpha = config['architecture'].get('alpha', 0.75)
    invariance_coeff = config['architecture'].get('invariance_coeff', 25)
    std_coeff = config['architecture'].get('std_coeff', 25)
    cov_coeff = config['architecture'].get('cov_coeff', 1)

    projector = Projector(features=3 * [4 * backbone.latent_dim],
                          norm_layer=norm_layer)
    local_projector = get_local_projector(train_loader,
                                          device,
                                          backbone,
                                          image_size,
                                          norm_layer,
                                          nb_neurons=512,
                                          inverse_pyramid=inverse_pyramid)

    model = CaevlFT(backbone=backbone,
                    projector=projector,
                    local_projector=local_projector,
                    device=device,
                    invariance_coeff=invariance_coeff,
                    std_coeff=std_coeff,
                    cov_coeff=cov_coeff,
                    alpha=alpha,
                    input_dimensions=image_size)

    model = model.to(model.device)
    model.train_mode()

    dummy_batch = next(iter(train_loader))[0].to(model.device)
    _ = model(dummy_batch, dummy_batch)
    nb_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Nb parameters to train: {nb_parameters:,}')
    nb_parameters_backbone = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    print(f'Nb parameters to train in the backbone: {nb_parameters_backbone:,}')
    if alpha > 0:
        nb_parameters_projector = sum(p.numel() for p in model.projector.parameters() if p.requires_grad)
        print(f'Nb parameters to train in the global projector: {nb_parameters_projector:,}')
    if alpha < 1:
        nb_parameters_local_projector = sum(p.numel() for p in model.local_projector.parameters() if p.requires_grad)
        print(f'Nb parameters to train in the local projector: {nb_parameters_local_projector:,}')

    learning_rate = float(config['trainer']['learning_rate'])
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    use_scheduler = config['trainer'].get('use_scheduler', True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=3) if use_scheduler else None

    try:
        assert args.load_weights
        model.load_state_dict(torch.load(model_save_path))
        print('\n\n\nLOADING WEIGHTS\n\n\n')
        path_optimizer = os.path.join(save_dir, 'optimizer.pth')
        if os.path.exists(path_optimizer):
            print('\n\nloading optimizer')
            optimizer.load_state_dict(torch.load(path_optimizer))
        path_scheduler = os.path.join(save_dir, 'scheduler.pth')
        if os.path.exists(path_scheduler):
            print("\n\nloading scheduler")
            scheduler.load_state_dict(torch.load(path_scheduler))
        print('Resuming training.')
    except Exception as e:
        if args.load_weights:
            print('\n\n\nCOULDN\'T LOAD WEIGHTS BECAUSE ->', e)
        args.load_weights = False  # either it is already False or there was a problem loading the model

    trainer = FtStageTrainer(model, train_loader, val_loader,
                             optimizer, scheduler, early_stopping)

    save_checkpoint = config['trainer'].get('save_checkpoint', 0)
    save_optimizer = config['trainer'].get('save_optimizer', False)

    train_losses, val_losses = trainer.train_(num_epochs,
                                              model_save_path=model_save_path,
                                              dir_save_losses=losses_save_dir_path,
                                              force_save=False,
                                              save_checkpoint=save_checkpoint,
                                              save_optimizer=save_optimizer)

    np.save(os.path.join(losses_save_dir_path, 'train_losses.npy'), train_losses)
    np.save(os.path.join(losses_save_dir_path, 'val_losses.npy'), val_losses)

    stop = time.time()
    time_spent = format_time(stop - start)
    print(f'It took {time_spent} to run.')
