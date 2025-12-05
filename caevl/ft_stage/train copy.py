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
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from caevl.data_generation.dataset.custom_dataset import CustomDataset
from caevl.ae.tools.early_stopping import EarlyStopping
from caevl.models.pretrained.dinov2.dinoV2 import DinoV2
from caevl.ft_stage.model.encoder import CaevlFT
from caevl.ft_stage.model.projector import Projector
from caevl.ft_stage.model.trainer import FtStageTrainer

# from utils import *

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
            
            
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
    if not os.path.exists(path) or not os.path.isdir(path):
        os.makedirs(path)


if __name__ == '__main__':

    mp.set_start_method('spawn', force=True)

    start = time.time()
    
    parser = argparse.ArgumentParser(description='Instantiate and train an AutoEncoder')
    parser.add_argument('-c', '--config',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file within ./configs')
    parser.add_argument('--load-weights',
                        dest="load_weights",
                        action=argparse.BooleanOptionalAction,
                        help = 'Whether to load weights of this model already previously trained, for instance if a training session was stopped too early.',
                        default=False)
    parser.add_argument('--freeze',
                        dest="freeze",
                        action=argparse.BooleanOptionalAction,
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
    
    #### TRAINING ####
    ### DATASETS ###
    images_directory = config['Dataset']['images_directory']
    use_PIL = config['Dataset'].get('use_PIL', True)
    apply_transform = config['Dataset']['apply_transform']
    apply_normalization = config['Dataset']['apply_normalization']
    max_workers = config['Dataset'].get('max_workers', 1)
    mean = config['Dataset'].get('mean')
    std = config['Dataset'].get('std')
    canny_edges = config['Dataset'].get('canny_edges', False)
    lsd_edges = config['Dataset'].get('lsd_edges', False)
    real_width = config['Dataset'].get('real_width', 1000)
    real_height = config['Dataset'].get('real_height', 1000)
    add_seg_mask = config['Dataset'].get('add_seg_mask', False)
    grayscale = config['Dataset'].get('grayscale', True)
    argmax_channels = config['Dataset'].get('argmax_channels', False)
    return_locations = config['Dataset'].get('return_locations', False)
    edge_detector = config['Dataset'].get('edge_detector', 'canny')
    if return_locations:
        print('Using locations in training.')
    
    print(f'Using data augmentations: {apply_transform}.')
        
    if apply_normalization:
        print(f'Apply normalization: {apply_normalization} with mean = {mean}, std = {std}.')
    
    path_dict_coordinates = config['Dataset']['path_dict_coordinates']
    
    ### TRAIN & VALIDATION DATASET ###
    years_for_train_dataset = config['Dataset']['train_dataset']['years']
    zones_for_train_dataset = config['Dataset']['train_dataset']['zones']
    train_image_paths = []
    for year in years_for_train_dataset:
        for zone in zones_for_train_dataset:
            train_image_paths += glob(os.path.join(images_directory, f'*{zone}*{year}*'))
    
    print(f'Using {len(train_image_paths)} images for training, from {images_directory}.')    
    train_image_paths.sort()
    
    ### SET SEED
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    
    np.random.shuffle(train_image_paths)
    
    val_split = config['Dataset']['val_dataset']['split']
    train_split = 1 - val_split
    nb_train_images = int(len(train_image_paths) * train_split)
    
    train_image_paths, val_image_paths = train_image_paths[:nb_train_images], train_image_paths[nb_train_images:]
    train_image_paths.sort(); val_image_paths.sort()
    np.save(os.path.join(save_dir, 'validation_images.npy'), np.array(val_image_paths))
    
    with open(path_dict_coordinates, 'rb') as f:
        dict_coordinates = pickle.load(f)
    
    train_dataset = IGNDataset(image_paths=train_image_paths,
                               dict_coordinates=dict_coordinates,
                               use_PIL=use_PIL,
                               apply_transform=apply_transform,
                               apply_normalization=apply_normalization,
                               return_both_images=True,
                               max_workers=max_workers,
                               output_size=image_size,
                               mean=mean,
                               std=std,
                               return_index=True,
                               canny_edges=canny_edges,
                               lsd_edges=lsd_edges,
                               real_width=real_width,
                               real_height=real_height,
                               return_locations=return_locations,
                               grayscale=grayscale,
                               argmax_channels=argmax_channels,
                               add_seg_mask=add_seg_mask,
                               edge_detector=edge_detector)
    
    val_dataset = IGNDataset(image_paths=val_image_paths,
                             dict_coordinates=dict_coordinates,
                             use_PIL=use_PIL,
                             apply_transform=apply_transform,
                             apply_normalization=apply_normalization,
                             return_both_images=True,
                             max_workers=max_workers,
                             output_size=image_size,
                             mean=mean,
                             std=std,
                             return_index=True,
                             canny_edges=canny_edges,
                             lsd_edges=lsd_edges,
                             real_width=real_width,
                             real_height=real_height,
                             return_locations=return_locations,
                             grayscale=grayscale,
                             argmax_channels=argmax_channels,
                             add_seg_mask=add_seg_mask,
                             edge_detector=edge_detector)   
    
    if config['Dataset'].get('transform') is not None:
        rot_degrees = config['Dataset']['transform'].get('rot_degrees', 30)
        translate = config['Dataset']['transform'].get('translate', (100, 100))
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
        apply_softmax = config['Dataset']['transform'].get('apply_softmax', False)
        
        train_dataset.set_transform(rot_degrees=rot_degrees, translate=translate,
                                    blur_kernel_size=blur_kernel_size, brightness_factor=brightness_factor,
                                    contrast_factor=contrast_factor, noise_mean=noise_mean, apply_softmax=apply_softmax,
                                    noise_std=noise_std, zoom_in_prob=zoom_in_prob,
                                    blur_prob=blur_prob, brightness_prob=brightness_prob,
                                    contrast_prob=contrast_prob, noise_prob=noise_prob,
                                    vignetting_prob=vignetting_prob, masking_prob=masking_prob)
                               
    ### TEST DATASET ###
    years_for_test_dataset = config['Dataset']['test_dataset']['years']
    zones_for_test_dataset = config['Dataset']['test_dataset']['zones']
    test_image_paths = []
    for year in years_for_test_dataset:
        for zone in zones_for_test_dataset:
            # test_image_paths += glob(os.path.join(images_directory, f'*_{zone}_{year}*'))
            test_image_paths += glob(os.path.join(images_directory, f'*{zone}*{year}*'))
    
    test_image_paths.sort()
    test_dataset = IGNDataset(image_paths=test_image_paths,
                               path_dict_coordinates=path_dict_coordinates,
                               use_PIL=use_PIL,
                               apply_transform=apply_transform,
                               apply_normalization=apply_normalization,
                               return_both_images=True,
                               max_workers=max_workers,
                               output_size=image_size,
                               mean=mean,
                               std=std,
                               return_index=True,
                               canny_edges=canny_edges,
                               lsd_edges=lsd_edges,
                               real_width=real_width,
                               real_height=real_height,
                               grayscale=grayscale,
                               argmax_channels=argmax_channels,
                               add_seg_mask=add_seg_mask,
                               edge_detector=edge_detector) 
    
    ### IGN-VAL DATASET ###
    path_ign_val_images = config['Dataset']['val_images_directory']
    ign_val_images = glob(os.path.join(path_ign_val_images, '*.jpg'))
    ign_val_images.sort()
    
    path_dict_coordinates_val = config['Dataset']['path_dict_coordinates_val']
    
    ign_val_dataset = IGNDataset(image_paths=ign_val_images,
                                 path_dict_coordinates=path_dict_coordinates_val,
                                 use_PIL=use_PIL,
                                 apply_transform=apply_transform,
                                 apply_normalization=apply_normalization,
                                 return_both_images=True,
                                 output_size=image_size,
                                 mean=mean,
                                 std=std,
                                 return_index=True,
                                 canny_edges=canny_edges,
                                 lsd_edges=lsd_edges,
                                 real_width=real_width,
                                 real_height=real_height,
                                 grayscale=grayscale,
                                 argmax_channels=argmax_channels,
                                 add_seg_mask=add_seg_mask,
                                 edge_detector=edge_detector)
    
    ### TRUE DATASET ###
    flight_number = config['trainer'].get('flight_number', 10)
    if config['Dataset'].get('path_dict_coordinates_drone', None) is not None:
        path_dict_coordinates_drone = config['Dataset'].get('path_dict_coordinates_drone', None)
    else:
        path_dict_coordinates_drone = f'/data_ssd/GNSS_DENIED/STAGE_2023/coordinates/vol{flight_number}_undistorted.pkl'
        
    if config['Dataset'].get('drone_images_directory', None) is not None:
        path_true_images = config['Dataset'].get('drone_images_directory', None)
    else:
        path_true_images = f'/data_ssd/GNSS_DENIED/STAGE_2023/vol{flight_number}_images_undistorted'        
    
    true_images = glob(os.path.join(path_true_images, '*.tif*'))
    true_images += glob(os.path.join(path_true_images, '*.jpg*'))
    true_images += glob(os.path.join(path_true_images, '*.png'))
    true_images.sort()
    
    print(f'{len(true_images)} drone images, from {path_true_images}.')
    
    true_dataset = IGNDataset(image_paths=true_images,
                              path_dict_coordinates=path_dict_coordinates_drone,
                              use_PIL=True,
                              apply_transform=apply_transform,
                              apply_normalization=apply_normalization,
                              return_both_images=True,
                              output_size=image_size,
                              mean=mean,
                              std=std,
                              return_index=True,
                              canny_edges=canny_edges,
                              lsd_edges=lsd_edges,
                              real_width=real_width,
                              real_height=real_height,
                              grayscale=grayscale,
                              argmax_channels=argmax_channels,
                              add_seg_mask=add_seg_mask,
                              edge_detector=edge_detector)
    
    min_x_coordinate = config['Dataset'].get('min_x_coordinate', None)
    max_x_coordinate = config['Dataset'].get('max_x_coordinate', None)
    min_y_coordinate = config['Dataset'].get('min_y_coordinate', None)
    max_y_coordinate = config['Dataset'].get('max_y_coordinate', None)
    if min_x_coordinate is not None or max_x_coordinate is not None or min_y_coordinate is not None or max_y_coordinate is not None:
        train_dataset.restrict_coordinates(min_x_coordinate, max_x_coordinate, min_y_coordinate, max_y_coordinate)
        val_dataset.restrict_coordinates(min_x_coordinate, max_x_coordinate, min_y_coordinate, max_y_coordinate)
        test_dataset.restrict_coordinates(min_x_coordinate, max_x_coordinate, min_y_coordinate, max_y_coordinate)
        ign_val_dataset.restrict_coordinates(min_x_coordinate, max_x_coordinate, min_y_coordinate, max_y_coordinate)
        true_dataset.restrict_coordinates(min_x_coordinate, max_x_coordinate, min_y_coordinate, max_y_coordinate)
        
    new_tiling_path = config['Dataset'].get('new_tiling_path', None)
    if new_tiling_path is not None:
        new_tiling = np.load(new_tiling_path)
        train_dataset.change_tiling(new_tiling)
        val_dataset.change_tiling(new_tiling)
        test_dataset.change_tiling(new_tiling)
        
                               
    ### TRAINER ###
    batch_size = int(config['trainer']['batch_size'])
    num_workers_loader = config['trainer']['num_workers_loader']
    print(f'Batch size: {batch_size}, num_workers loader: {num_workers_loader}')
    print(f'Using {len(train_dataset)} images for training, {len(val_dataset)} images for validation and {len(test_dataset)} images for testing.')
    print(f'Using {len(true_dataset)} true images.')
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, shuffle=True, drop_last=False,
                              num_workers=num_workers_loader, persistent_workers=True)
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size, shuffle=False, drop_last=False,
                            num_workers=num_workers_loader, persistent_workers=True)
    test_loader = DataLoader(test_dataset, 
                             batch_size=batch_size, shuffle=False, drop_last=False,
                             num_workers=num_workers_loader, persistent_workers=True)
    ign_val_loader = DataLoader(ign_val_dataset, 
                                batch_size=batch_size, shuffle=False, drop_last=False,
                                num_workers=num_workers_loader, persistent_workers=True)
    true_loader = DataLoader(true_dataset, 
                                batch_size=batch_size, shuffle=False, drop_last=False,
                                num_workers=num_workers_loader, persistent_workers=True)
    
    print('Using transform:', train_dataset.transform)
    
    num_epochs = config['trainer']['num_epochs']
    patience = config['trainer']['patience']
    early_stopping = None if patience is None else EarlyStopping(patience=patience)
    
    device = torch.device(config['architecture'].get('device', 'cpu'))
    load_weights_backbone = config['architecture'].get('load_weights_backbone', True)
    
    dir_model = config['architecture'].get('dir_model', 'AutoEncoder')
    inverse_pyramid = False
    norm_layer = 'layer_norm'
    backbone = config['architecture']['backbone']
    if backbone.lower() == 'convnext':
        backbone = ConvNeXt(device)
    elif backbone.lower() in ['dino', 'dinov2']:
        backbone = DinoV2(device)
    elif backbone.lower() == 'mae':
        mask_ratio = config['architecture'].get('mask_ratio', 0)
        architecture_mae = config['architecture'].get('architecture_mae', 'base')
        backbone = load_mae(device, mask_ratio, architecture_mae)
        inverse_pyramid = False
    else:
        backbone = config['architecture']['backbone']
        path_to_model_config = os.path.join(f'../models/trained/{dir_model}/{backbone}', 'train_config.yml')

        # save backbone config file
        with open(path_to_model_config, 'r') as backbone_cfg:
            backbone_config = yaml.safe_load(backbone_cfg) 
        name_backbone_config = os.path.join(save_dir, 'backbone_config.yml')
        with open(name_backbone_config, 'w') as yml_file:
            yaml.dump(backbone_config, yml_file)

        if dir_model == 'AutoEncoder':
            autoencoder, norm_layer = load_ae_backbone(path_to_model_config, image_size, device=device, load_weights=load_weights_backbone)
            backbone = autoencoder.encoder
        else:
            backbone = load_tuned_mae(path_to_model_config)
        
        
    alpha = config['architecture'].get('alpha', 0.75)
    invariance_coeff = config['architecture'].get('invariance_coeff', 25)
    std_coeff = config['architecture'].get('std_coeff', 25)
    cov_coeff = config['architecture'].get('cov_coeff', 1)
    
    projector = Projector(features=3*[4*backbone.latent_dim], norm_layer=norm_layer)
    local_projector = get_local_projector(train_loader, device, backbone, image_size, norm_layer, nb_neurons=512, inverse_pyramid=inverse_pyramid)
    
    vicreg = VICRegL(backbone=backbone,
                     projector=projector,
                     local_projector=local_projector,
                     device=device,
                     invariance_coeff=invariance_coeff,
                     std_coeff=std_coeff,
                     cov_coeff=cov_coeff,
                     alpha=alpha,
                     input_dimensions=image_size)
    
    vicreg = vicreg.to(vicreg.device)
    vicreg.train_mode()
    
    print(vicreg)
    
    dummy_batch = next(iter(train_loader))[0].to(vicreg.device)
    _ = vicreg(dummy_batch, dummy_batch)
    nb_parameters = sum(p.numel() for p in vicreg.parameters() if p.requires_grad)
    print(f'Nb parameters to train: {nb_parameters:,}')
    nb_parameters_backbone = sum(p.numel() for p in vicreg.backbone.parameters() if p.requires_grad)
    print(f'Nb parameters to train in the backbone: {nb_parameters_backbone:,}')
    if alpha > 0:
        nb_parameters_projector = sum(p.numel() for p in vicreg.projector.parameters() if p.requires_grad)
        print(f'Nb parameters to train in the global projector: {nb_parameters_projector:,}')
    if alpha < 1:
        nb_parameters_local_projector = sum(p.numel() for p in vicreg.local_projector.parameters() if p.requires_grad)
        print(f'Nb parameters to train in the local projector: {nb_parameters_local_projector:,}')
    
    learning_rate = float(config['trainer']['learning_rate'])
    optimizer = AdamW(vicreg.parameters(), lr=learning_rate)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    use_scheduler = config['trainer'].get('use_scheduler', True)
    if use_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=3)
    else: scheduler = None

    try:
        assert args.load_weights
        vicreg.load_state_dict(torch.load(model_save_path))
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
        args.load_weights = False # either it is already False or there was a problem loading the model

    if args.freeze:
        for param in vicreg.projector.parameters():
            param.requires_grad = False
        for param in vicreg.local_projector.parameters():
            param.requires_grad = False
    
    trainer = VICReg_Trainer(vicreg, train_loader, val_loader, ign_val_loader,
                             optimizer, scheduler, true_loader, early_stopping, config)
    
    compute_accuracy = config['trainer'].get('compute_accuracy', False)
    save_checkpoint = config['trainer'].get('save_checkpoint', 0)
    save_optimizer = config['trainer'].get('save_optimizer', False)
    localize_before_train = config['trainer'].get('localize_before_train', False)
    validation_on_loc = config['trainer'].get('validation_on_loc', False)
    keep_top_n_losses = config['trainer'].get('keep_top_n_losses', 1)
    largest = config['trainer'].get('largest', False)
    adaptive_augmentations = config['trainer'].get('adaptive_augmentations', False)
    
    if validation_on_loc:
        print('Using localization values to perform validation.')
    if keep_top_n_losses != 1 and not isinstance(keep_top_n_losses, str):
        print(f'Keeping {100*keep_top_n_losses:.2f}% of the {"largest" if largest else "lowest"} losses for backpropagation.')
    if adaptive_augmentations:
        print('Using adaptive augmentations.')
    
    train_losses, val_losses = trainer.train_(num_epochs, 
                                             model_save_path=model_save_path,
                                             compute_accuracy=compute_accuracy,
                                             dir_save_losses=losses_save_dir_path, 
                                             force_save=False,
                                             save_checkpoint=save_checkpoint,
                                             save_optimizer=save_optimizer,
                                             localize_before_train=localize_before_train,
                                             validation_on_loc=validation_on_loc,
                                             keep_top_n_losses=keep_top_n_losses,
                                             adaptive_augmentations=adaptive_augmentations)
    
    
    np.save(os.path.join(losses_save_dir_path, 'train_losses.npy'), train_losses)
    np.save(os.path.join(losses_save_dir_path, 'val_losses.npy'), val_losses)
        
                  
    stop = time.time()
    time_spent = format_time(stop - start)
    print(f'It took {time_spent} to run.')