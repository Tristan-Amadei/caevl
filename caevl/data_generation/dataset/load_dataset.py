import pickle
import numpy as np
from glob import glob
import os

import torch
from torch.utils.data import DataLoader

from caevl.data_generation.dataset.custom_dataset import CustomDataset


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


def list_images(dir_):
    images =  glob(os.path.join(dir_, '*tif*'))
    images += glob(os.path.join(dir_, '*jpg'))
    images += glob(os.path.join(dir_, '*JPG'))
    images += glob(os.path.join(dir_, '*png'))
    images += glob(os.path.join(dir_, '*PNG'))
    return images


def load_dataset(config, canny_edges, return_both_images=False):
    image_size = tuple(config['image_size'])

    images_directory = config['Dataset']['images_directory']
    val_images_directory = config['Dataset']['val_images_directory']
    use_PIL = config['Dataset']['use_PIL']
    apply_transform = config['Dataset']['apply_transform']
    apply_normalization = config['Dataset']['apply_normalization']
    mean = config['Dataset'].get('mean')
    std = config['Dataset'].get('std')
    grayscale = config['Dataset'].get('grayscale', True)
    return_locations = config['Dataset'].get('return_locations', False)
    
    path_dict_coordinates = config['Dataset']['path_dict_coordinates']
    
    train_image_paths = list_images(images_directory)       
    train_image_paths.sort()
    
    with open(path_dict_coordinates, 'rb') as f:
        dict_coordinates = pickle.load(f)

    val_image_paths = list_images(val_images_directory)
    val_image_paths.sort()
    
    train_dataset = CustomDataset(image_paths=train_image_paths,
                               dict_coordinates=dict_coordinates,
                               use_PIL=use_PIL,
                               apply_transform=apply_transform,
                               apply_normalization=apply_normalization,
                               return_both_images=return_both_images,
                               output_size=image_size,
                               mean=mean,
                               std=std,
                               canny_edges=canny_edges,
                               return_locations=return_locations,
                               return_index=True,
                               grayscale=grayscale)
    
    val_dataset = CustomDataset(image_paths=val_image_paths,
                             dict_coordinates=dict_coordinates,
                             use_PIL=use_PIL,
                             apply_transform=False,
                             apply_normalization=apply_normalization,
                             return_both_images=return_both_images,
                             output_size=image_size,
                             mean=mean,
                             std=std,
                             canny_edges=canny_edges,
                             return_locations=return_locations,
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
    
    return train_loader, val_loader