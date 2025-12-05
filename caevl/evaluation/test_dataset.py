import os
from glob import glob
from PIL import Image
import json
import pickle
import yaml
import numpy as np

import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors

import caevl.data_generation.dataset.augmentations as aug


def read_images_paths(dataset_folder, sort_by_name=False):
    """Find images within 'dataset_folder'. If the file
    'dataset_folder'_images_paths.txt exists, read paths from such file.
    Otherwise, use glob(). Keeping the paths in the file speeds up computation,
    because using glob over large folders might be slow.

    Parameters
    ----------
    dataset_folder : str, folder containing JPEG images

    Returns
    -------
    images_paths : list[str], paths of JPEG images within dataset_folder
    """

    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Folder {dataset_folder} does not exist")

    print(f"Searching test images in {dataset_folder} with glob()")
    images_paths = sorted(glob(f"{dataset_folder}/**/*.jpg", recursive=True))
    images_paths += sorted(glob(f"{dataset_folder}/**/*.JPG", recursive=True))
    images_paths += sorted(glob(f"{dataset_folder}/**/*.tif*", recursive=True))
    images_paths += sorted(glob(f"{dataset_folder}/**/*png", recursive=True))
    if len(images_paths) == 0:
        raise FileNotFoundError(f"Directory {dataset_folder} does not contain any JPEG or tiff images")
    if sort_by_name:
        # case where there is no coords dict and the coordinates are stored in the filenames
        images_paths = sorted(images_paths, key=lambda x: x.split('@')[-1])
    return images_paths


def load_dict(path):

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[-1].lower()

    # ------------------- JSON -------------------
    if ext == ".json":
        with open(path, "r") as f:
            return json.load(f)

    # ------------------- PICKLE -------------------
    if ext in [".pkl", ".pickle"]:
        with open(path, "rb") as f:
            return pickle.load(f)

    # ------------------- YAML -------------------
    if ext in [".yaml", ".yml"]:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # ------------------- Unsupported -------------------
    raise ValueError(f"Unsupported file type: '{ext}' for file {path}")


def paths_to_coords(paths, coord_dict):
    """
    Given a list of image paths and a dictionary mapping image names (or full paths)
    to coordinates (e.g., [x, y]), return a list of coordinates in the same order.
    """
    coords = []
    for p in paths:
        name = os.path.basename(p)
        if p in coord_dict:
            coords.append(coord_dict[p][:2])
        elif name in coord_dict:
            coords.append(coord_dict[name][:2])
        else:
            raise KeyError(f"No coordinate found for image: {p}")
    return coords


def map_name_to_coordsName(paths):
    """
        For paths named @{utm_east}@{utm_north}@filename, maps 'filename' to full name
    """
    mapping = dict()
    for path in paths:
        name = os.path.basename(path)
        filename = name.split('@')[-1]
        mapping[filename] = name
    return mapping


class TestDataset(data.Dataset):
    def __init__(self, database_folder, queries_folder,
                 database_coords_path, queries_coords_path,
                 positive_dist_threshold=[25], image_size=None,
                 is_caevl=True):
        """Dataset with images from database and queries, used for validation and test.
        Parameters
        ----------
        dataset_folder : str, should contain the path to the val or test set,
            which contains the folders {database_folder} and {queries_folder}.
        database_folder : str, name of folder with the database.
        queries_folder : str, name of folder with the queries.
        positive_dist_threshold : int, distance in meters for a prediction to
            be considered a positive.
        """
        super().__init__()

        self.database_paths = read_images_paths(database_folder, sort_by_name=database_coords_path is None)
        self.queries_paths = read_images_paths(queries_folder, sort_by_name=queries_coords_path is None)
        self.images_paths = list(self.database_paths) + list(self.queries_paths)

        self.num_database = len(self.database_paths)
        self.num_queries = len(self.queries_paths)

        database_coord_dict = load_dict(database_coords_path) if database_coords_path is not None else None
        queries_coord_dict = load_dict(queries_coords_path) if queries_coords_path is not None else None

        # Map each image path to its coordinates, in correct order
        if database_coord_dict is not None:
            self.database_utms = paths_to_coords(self.database_paths, database_coord_dict)
        else:
            self.database_utms = np.array(
                [(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]
            ).astype(float)
        if queries_coord_dict is not None:
            self.queries_utms = paths_to_coords(self.queries_paths, queries_coord_dict)
        else:
            self.queries_utms = np.array(
                [(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]
            ).astype(float)

        self.database_utms = np.array(self.database_utms)
        self.queries_utms = np.array(self.queries_utms)
        self.utms = np.concatenate([self.database_utms, self.queries_utms], axis=0)

        # Find positives_per_query, which are within positive_dist_threshold (default 25 meters)
        self.positive_dist_threshold = positive_dist_threshold
        self.positives_per_query = []
        for i, dist in enumerate(positive_dist_threshold):
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.database_utms)
            self.positives_per_query.append(knn.radius_neighbors(
                self.queries_utms, radius=dist, return_distance=False
            ))

        self.is_caevl = is_caevl  # used to determine which set of transforms to use

        self.edges_mask = aug.CannyMask(low_threshold=70, high_threshold=200, apertureSize=3, L2gradient=False)
        self.transform = transforms.Compose(
            ([aug.Resize(image_size, antialias=True)] if image_size is not None else [])
            + ([transforms.Grayscale() if self.is_caevl else []])
            + ([self.edges_mask] if self.is_caevl else [])
            + ([transforms.ToTensor()])
        )

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        pil_img = Image.open(image_path)

        normalized_img = self.transform(pil_img)
        utm = self.utms[index]

        return normalized_img, index, utm

    def __len__(self):
        return len(self.images_paths)

    def __repr__(self):
        return f"< #queries: {self.num_queries}; #database: {self.num_database} >"

    def get_positives(self):
        return self.positives_per_query
