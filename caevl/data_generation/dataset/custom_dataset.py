import numpy as np
from PIL import Image
from glob import glob
import os
import pickle
import json
import tifffile

import torch
from torchvision import transforms
from torch.utils.data import Dataset

from caevl.data_generation.dataset.utils.compute_mean_accuracy import compute_mean_std_online
import caevl.data_generation.dataset.augmentations as aug


class CustomDataset(Dataset):
    """IGN Dataset object."""

    def __init__(self, image_paths: list,
                       path_dict_coordinates: str=None,
                       dict_coordinates: dict=None,
                       use_PIL: bool=True,
                       apply_transform: bool=False,
                       return_index: bool=False,
                       return_both_images: bool=False,
                       apply_normalization: bool=False,
                       output_size: tuple=(256, 256),
                       mean: float=None,
                       std: float=None,
                       canny_edges: bool=False,
                       add_vignetting: bool=False,
                       grayscale: bool=True,
                       return_locations: bool=False):
        """Dataset class for the IGN dataset.

        Parameters
        ----------
        image_paths : list
            List of paths to all images.
        path_dict_coordinates : str, optional
            Path to the file that links image names and their coordinates, default is None.
        dict_coordinates : dict, optional
            Dictionary linking image names and their coordinates, default is None.
        use_PIL : bool, optional
            If True, uses PIL to open images; otherwise, uses the matrix data, default is True.
        apply_transform : bool, optional
            Whether to apply transformations to the images, default is False.
        return_index : bool, optional
            Whether to return the index of the fetched elements, default is False.
        return_both_images : bool, optional
            Whether to return both the original and transformed images, default is False.
        apply_normalization : bool, optional
            Whether to apply normalization to the images, default is False.
        output_size : tuple, optional
            Desired output size of the images, default is (256, 256).
        mean : float, optional
            Mean for normalization. If None, and apply_normalization is True, computes
            the mean and std of the passed datased. Default is None.
        std : float, optional
            Standard deviation for normalization. If None, and apply_normalization is True, computes the meand and std of
            the passed dataset. Default is None.
        canny_edges : bool, optional
            Whether to add Canny edges to the images. Default is False.
        add_vignetting : bool, optional
            Whether to add vignetting to the images, default is False.
        grayscale : bool, optional
            Whether to convert images to grayscale, default is True.
        return_locations : bool, optional
            Whether to return locations of the images when using geometric transformations, default is False.
        """

        super(CustomDataset, self).__init__()
        self.image_paths = image_paths
        self.image_names = [os.path.split(image_path)[-1] for image_path in self.image_paths]  # keep the extension

        self.use_PIL = use_PIL
        self.output_size = output_size
        self.return_index = return_index
        self.return_both_images = return_both_images
        self.return_locations = return_locations

        self.add_vignetting = add_vignetting
        self.grayscale = grayscale

        # Open dictionary that has image names as keys and their coordinates as values
        self.path_dict_coordinates = path_dict_coordinates
        self.dict_coordinates = dict()
        if path_dict_coordinates is not None:
            with open(path_dict_coordinates, 'rb') as handle:
                self.dict_coordinates = pickle.load(handle)
        elif dict_coordinates is not None:
            self.dict_coordinates = dict_coordinates
        self.remove_extensions

        if canny_edges:
            apply_normalization = False
        self.apply_normalization = apply_normalization

        if apply_normalization:
            if mean is None or std is None:
                self.mean, self.std = self.compute_mean_std()
            else:
                self.mean, self.std = mean, std

        self.apply_transform = apply_transform
        self.canny_edges = canny_edges
        self.set_transform()

    def remove_extensions(self):
        """Remove file extensions from the keys in the dictionary of coordinates."""

        keys = list(self.dict_coordinates.keys())
        for key in keys:
            value = self.dict_coordinates.pop(key)
            key_no_extension = os.path.splitext(key)[0]
            self.dict_coordinates[key_no_extension] = value

    def set_transform(self, rot_degrees=30, translate=(50, 50), blur_kernel_size=5, brightness_factor=1.,
                      contrast_factor=1., noise_mean=0, noise_std=1, zoom_in_prob=0.8, blur_prob=0.5,
                      brightness_prob=0.5, contrast_prob=0.5, noise_prob=0.25, vignetting_prob=0.9, masking_prob=1,
                      vignetting_val=70):

        if self.canny_edges:
            self.edges_mask = aug.CannyMask(low_threshold=70, high_threshold=200, apertureSize=3, L2gradient=False)

        # vignetting_val = 80 if self.output_size[0] == self.output_size[1] else 70
        self.to_tensor = transforms.Compose(
            ([aug.Resize(self.output_size)])
            + ([aug.Grayscale(self.grayscale)])
            + ([aug.RandomVignetting(sigma=vignetting_val, probability=1)] if (self.add_vignetting) else [])
            + ([self.edges_mask] if self.canny_edges else [])
            + ([aug.ToTensor(process_independently=False)])
            + ([aug.RandomMasking(probability=1)] if self.add_vignetting else [])
            + ([transforms.Normalize(self.mean, self.std)] if self.apply_normalization else [])
        )

        if self.apply_transform:
            self.transform = transforms.Compose(
                ([aug.Resize(self.output_size)])
                + ([aug.Grayscale(self.grayscale)])
                + [
                    aug.RandomZoomIn(output_size=self.output_size, degrees=rot_degrees, translate=translate,
                                     return_locations=self.return_locations, probability=zoom_in_prob),
                    aug.RandomGaussianBlur(kernel_size=blur_kernel_size, blur_probability=blur_prob),
                    aug.RandomBrightnessContrast(brightness_factor=brightness_factor, contrast_factor=contrast_factor,
                                                 brightness_probability=brightness_prob, contrast_probability=contrast_prob),
                ]
                + ([aug.RandomGaussianNoise(mean=noise_mean, std=noise_std, clip=True, probability=noise_prob)])
                + ([aug.RandomVignetting(sigma=vignetting_val, probability=vignetting_prob)])
                + ([self.edges_mask] if self.canny_edges else [])
                + ([aug.ToTensor(process_independently=False),
                    aug.RandomMasking(probability=masking_prob),
                    ])
                + ([aug.CannyMaskNormalize(self.mean, self.std)] if self.apply_normalization else [])
            )

        else:
            self.transform = None

    def compute_mean_std(self):
        """Compute the mean and standard deviation of the dataset.

        Returns
        -------
        mean : float
            Mean of the dataset.
        std : float
            Standard deviation of the dataset.
        """

        ### look for already computed stats
        try:
            dir_, _ = os.path.split(self.image_paths[0])
            stats_file_path = glob(os.path.join(dir_, '*.json'))[0]
            with open(stats_file_path, 'r') as f:
                stats_file = json.load(f)
            mean = stats_file['mean']
            std = stats_file['std']
            return mean, std
        except:
            pass

        print("Computing mean and std for dataset.")
        if not self.use_PIL:
            mean = self.matrix_data.mean()
            std = self.matrix_data.std()
            return mean, std

        image_dir, _ = os.path.split(self.image_paths[0])
        mean, std = compute_mean_std_online(image_dir)
        return mean, std

    def __len__(self):
        return len(self.image_names)

    def open_image(self, image_path):
        try:
            image = Image.open(image_path)
        except:
            image = tifffile.imread(image_path).astype(np.float32)
        return image

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        full_image_name = self.image_names[idx]
        image_name, extension = os.path.splitext(full_image_name)

        image_path = self.image_paths[idx]
        image = self.open_image(image_path)

        try:
            coordinates = self.dict_coordinates[image_name][:2]
        except KeyError:
            coordinates = self.dict_coordinates[full_image_name][:2]
        except Exception as e:
            raise e

        if not self.return_both_images:
            image_ = self.to_tensor(image) if self.transform is None else self.transform(image)

            if self.return_index:
                return image_, coordinates, idx
            return image_, coordinates

        tensor_im = self.to_tensor(image)

        if self.return_locations:
            augmented_im, locations = self.transform(image) if self.transform is not None else (tensor_im, None)
            locations = torch.tensor(locations)
            locations = _location_to_NxN_grid(locations, N=7)
            if self.return_index:
                return tensor_im, augmented_im, coordinates, idx, locations
            return tensor_im, augmented_im, coordinates, locations

        augmented_im = self.transform(image) if self.transform is not None else tensor_im
        if self.return_index:
            return tensor_im, augmented_im, coordinates, idx
        return tensor_im, augmented_im, coordinates

    def get_coordinates(self, idx, return_angle=False):
        """Get coordinates of the image at the given index.

        Parameters
        ----------
        idx : int or list of int
            Index or list of indices.
        return_angle : bool, optional
            Whether to return the angle.

        Returns
        -------
        coordinates : np.ndarray
            Coordinates of the image(s).
        """

        nb_elements = 2 if not return_angle else 3
        if isinstance(idx, int):
            return self.dict_coordinates[os.path.splitext(self.image_names[idx])[0]][:nb_elements]
        coordinates = np.zeros((len(idx), 2))
        for i, index in enumerate(idx):
            coordinates[i] = self.dict_coordinates[os.path.splitext(self.image_names[index])[0]][:nb_elements]
        return coordinates

    def restrict_coordinates(self, min_x, max_x, min_y, max_y):
        """Restrict the dataset to images within the given coordinate range.

        Parameters
        ----------
        min_x : float
            Minimum x-coordinate.
        max_x : float
            Maximum x-coordinate.
        min_y : float
            Minimum y-coordinate.
        max_y : float
            Maximum y-coordinate.
        """

        print(f'Before restriction, there were {len(self.image_names)} images, ', end='')
        if min_x is None:
            min_x = 0
        if max_x is None:
            max_x = np.inf
        if min_y is None:
            min_y = 0
        if max_y is None:
            max_y = np.inf

        image_paths_to_keep = []
        image_names_to_keep = []
        for i, image in enumerate(self.image_names):
            coordinates = self.dict_coordinates[os.path.splitext(image)[0]]
            if min_x <= coordinates[0] and coordinates[0] <= max_x and min_y <= coordinates[1] and coordinates[1] <= max_y:
                image_names_to_keep.append(image)
                image_paths_to_keep.append(self.image_paths[i])

        self.image_paths = image_paths_to_keep
        self.image_names = image_names_to_keep
        print(f'there are {len(image_names_to_keep)} after restriction.')

    def change_tiling(self, tiling):
        """Change the tiling of the dataset.

        Parameters
        ----------
        tiling : list of tuples
            List of coordinates for the new tiling.
        """

        print(f'Before restriction, there were {len(self.image_names)} images, ', end='')

        all_tiling_coordinates = set([
            tuple(coord[:2]) for coord in tiling
        ])
        image_paths_to_keep = []
        image_names_to_keep = []
        for i, image in enumerate(self.image_names):
            coordinates = self.dict_coordinates[os.path.splitext(image)[0]]
            if tuple(coordinates[:2]) in all_tiling_coordinates:
                image_names_to_keep.append(image)
                image_paths_to_keep.append(self.image_paths[i])

        self.image_paths = image_paths_to_keep
        self.image_names = image_names_to_keep
        print(f'there are {len(image_names_to_keep)} after restriction.')


def _locations_to_NxN_grid(locations, N=7):
    locations_N_N = torch.zeros((len(locations), N, N, 2))
    for i in range(len(locations)):
        locations_N_N[i] = _location_to_NxN_grid(locations[i], N=N)
    locations_N_N = locations_N_N.reshape(-1, N * N, 2)
    return locations_N_N.to(locations.device)


def _location_to_NxN_grid(location, N=8, flip=False):
    i, j, h, w, H, W = location
    size_h_case = h / N
    size_w_case = w / N
    half_size_h_case = size_h_case / 2
    half_size_w_case = size_w_case / 2
    final_grid_x = torch.zeros(N, N)
    final_grid_y = torch.zeros(N, N)

    final_grid_x[0][0] = i + half_size_h_case
    final_grid_y[0][0] = j + half_size_w_case
    for k in range(1, N):
        final_grid_x[k][0] = final_grid_x[k - 1][0] + size_h_case
        final_grid_y[k][0] = final_grid_y[k - 1][0]
    for l in range(1, N):
        final_grid_x[0][l] = final_grid_x[0][l - 1]
        final_grid_y[0][l] = final_grid_y[0][l - 1] + size_w_case
    for k in range(1, N):
        for l in range(1, N):
            final_grid_x[k][l] = final_grid_x[k - 1][l] + size_h_case
            final_grid_y[k][l] = final_grid_y[k][l - 1] + size_w_case

    final_grid = torch.stack([final_grid_x, final_grid_y], dim=-1)
    final_grid = final_grid.reshape(N * N, 2)

    return final_grid
