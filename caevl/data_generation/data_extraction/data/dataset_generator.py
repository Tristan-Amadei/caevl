import numpy as np
import os
from tqdm import tqdm
import multiprocessing
from multiprocessing.pool import Pool
import pickle

from caevl.data_generation.data_extraction.data.ign_frame_grabber import IgnFrameGrabber


def make_dir(path):
    """Create a directory (and all its parents) at the given path if not already existing.

    Parameters
    ----------
    path : str
        Path to create the directory to.
    """
    
    if not os.path.isdir(path):
        os.makedirs(path)   


class DatasetGenerator():
    def __init__(self, search_zone_points: np.ndarray, 
                       ign_frame_grabbers: list, 
                       output_dir: str):
        """Create an instance of DatasetGenerator.

        Parameters
        ----------
        search_zone_points : np.ndarray, shape (n, 3)
            Coordinates in Lambert93 system, as well as rotation angle, for each point in which to extract an image.
        ign_frame_grabbers : list
            List of instances of IgnFrameGrabber, containing all the IGN tiles from which to extract images.
        output_dir : str
            Path to the directory where to save extracted images.
        """
        
        self.search_zone_points = search_zone_points
        self.ign_frame_grabbers = ign_frame_grabbers
        
        self.output_dir = output_dir
        make_dir(output_dir)   
        
    
    def get_dict_indices_per_tile(self, ign_frame_grabber: IgnFrameGrabber):
        """Given an instance of IgnFrameGrabber, determines the indices of points from the search zone
        to extract from each IGN tile of the IgnFrameGrabber?

        Parameters
        ----------
        ign_frame_grabber : IgnFrameGrabber
            Instance of IgnFrameGrabber that contains the IGN tiles.

        Returns
        -------
        dict[str, list]
            Dictionary that contains the list of indices of points to extract for each IGN tile.
        """
        
        indices_per_tile = {}
        for index in tqdm(range(len(self.search_zone_points))):
            point = self.search_zone_points[index]
            coords = point[:2]
            tile = ign_frame_grabber.findTile(coords)
            if tile in ign_frame_grabber.tilelist:
                if indices_per_tile.get(tile) is None:
                    indices_per_tile[tile] = [index]
                else:
                    indices_per_tile[tile].append(index)
        return indices_per_tile
    
    
    def get_dict_indices_per_tile_across_all_ign_frame_grabbers(self):
        dict__indices_per_tile = {}
        for i, ign_frame_grabber in enumerate(self.ign_frame_grabbers):
            indices_per_tile = self.get_dict_indices_per_tile(ign_frame_grabber)
            dict__indices_per_tile[i] = indices_per_tile
        return dict__indices_per_tile
    
    
    def save_as_dataset(self, ign_frame_grabber_index, 
                              tile,
                              indices, 
                              search_zone_points, 
                              output_dir,
                              IGN_width, 
                              IGN_height, 
                              out_width, 
                              out_height, 
                              erase_existing,
                              crop):
        
        ign_frame_grabber = self.ign_frame_grabbers[ign_frame_grabber_index]
        idx_problems, dict_coordinates = ign_frame_grabber.saveAsDatasetTile(tile, indices, search_zone_points, output_dir,
                                                                             IGN_width, IGN_height, out_width, out_height, erase_existing, crop)
        return ign_frame_grabber_index, idx_problems, dict_coordinates
    

    def extract_all_IGN_images(self, IGN_width: int, 
                                     IGN_height: int, 
                                     out_width: int, 
                                     out_height: int, 
                                     erase_existing: bool=True,
                                     max_workers: float=None,
                                     dict_problems_directory: str=None,
                                     dict_problems_save_name: str=None,
                                     dict_coordinates_directory: str=None,
                                     dict_coordinates_save_name: str=None,
                                     crop: bool=True):
        """Extract all the images from all the instances of the IgnFrameGrabber.

        Parameters
        ----------
        IGN_width : int
            Width (in pixels) to extract from the IGN tile.
        IGN_height : int
            Height (in pixels) to extract from the IGN tile.
        out_width : int
            Width (in pixels) of the final extracted images. 
        out_height : int
            Height (in pixels) of the final extracted images.
        erase_existing : bool, optional
            Whether to erase already existing images with same filename, by default True
        max_workers : float, optional
            Max number of parallel processes to use, by default None
            If None: uses half of available workers
            If -1: use all available workers
            If int passed, uses the numbers of workers passed.
            If float passed, assumes it is a percentage of total available workers to use.
        dict_problems_directory : str, optional
            Directory in which should be saved the dictionary with indices that encountered a problem during extraction, by default None.
            If None, a new dictionary is created.
        dict_problems_save_name : str, optional
            Name of the file where should be saved the dictionary with indices that encountered a problem during extraction, by default None.
            If None, a new dictionary is created.
        dict_coordinates_directory : str, optional
            Directory in which should be saved the dictionary linking the images and their coordinates, by default None.
            If None, a new dictionary is created.
        dict_coordinates_save_name : str, optional
            Name of the file where should be saved the dictionary linking the images and their coordinates, by default None.
            If None, a new dictionary is created.

        Returns
        -------
        dict[str, list]
            Dictionary that contains, for each ign_frame_grabber, 
            the list of indices of points that had an issue during extraction.
        dict[str, tuple]
            Dictionary that contains, for each saved image, their coordinates.
        """

        print(f"Get correspondence locations - IGN tiles.")
        dict__indices_per_tile = self.get_dict_indices_per_tile_across_all_ign_frame_grabbers()
        
        args = []
        for ign_frame_grabber_index in dict__indices_per_tile.keys():
            tiles = list(dict__indices_per_tile[ign_frame_grabber_index].keys())
            args += [
                (ign_frame_grabber_index, tile, dict__indices_per_tile[ign_frame_grabber_index][tile], self.search_zone_points, self.output_dir,
                IGN_width, IGN_height, out_width, out_height, erase_existing, crop) for i, tile in enumerate(tiles)
            ]
        
        if max_workers is None:
            max_workers = int(np.ceil(multiprocessing.cpu_count()/2))
        elif max_workers == -1:
            max_workers = multiprocessing.cpu_count()
        elif 0 < max_workers and max_workers <= 1:
            max_workers = int(np.ceil(multiprocessing.cpu_count() * max_workers))
        elif 1 < max_workers and max_workers <= multiprocessing.cpu_count():
            max_workers = int(max_workers)
        else:
            max_workers = 1       
        
        if not erase_existing:
            try:
                with open(f'{dict_problems_directory}/{dict_problems_save_name}', 'rb') as handle:
                    dict_points_with_problem = pickle.load(handle)
            except:
                dict_points_with_problem = {}
        else: 
            dict_points_with_problem = {}
        for i, ign_frame_grabber in enumerate(self.ign_frame_grabbers):
            if dict_points_with_problem.get(i) is None:
                dict_points_with_problem[i] = []
        
        if not erase_existing: 
            try:
                with open(f'{dict_coordinates_directory}/{dict_coordinates_save_name}', 'rb') as handle:
                    dict_coordinates = pickle.load(handle)
            except:
                dict_coordinates = {} 
        else:
            dict_coordinates = {}
               
        print(f'Extracting images, working with maximum {max_workers} workers and {len(args)} different tiles to deal with.')
        indices_tiles_done = []
        with tqdm(total=len(args)) as pbar:
            with Pool(processes=max_workers) as parallel_executor:
                async_results = [parallel_executor.apply_async(
                    self.save_as_dataset, args=arg) for arg in args
                    ]
                
                while len(indices_tiles_done) < len(args):
                    for index in range(len(args)):
                        async_result = async_results[index]
                        if not async_result.ready() or index in indices_tiles_done:
                            pass
                        else:
                            ign_frame_grabber_index, indices_points_with_problem__tile, dict_coordinates__tile = async_result.get()
                            dict_points_with_problem[ign_frame_grabber_index].append(indices_points_with_problem__tile)
                            dict_coordinates.update(dict_coordinates__tile)
                            
                            if dict_problems_directory is not None and dict_problems_save_name is not None:
                                with open(f'{dict_problems_directory}/{dict_problems_save_name}', 'wb') as f:
                                    pickle.dump(dict_points_with_problem, f) 
                
                            if dict_coordinates_directory is not None and dict_coordinates_save_name is not None:
                                with open(f'{dict_coordinates_directory}/{dict_coordinates_save_name}', 'wb') as f:
                                    pickle.dump(dict_coordinates, f)
                            
                            pbar.update(1)
                            indices_tiles_done.append(index)
        parallel_executor.close()
        return dict_points_with_problem, dict_coordinates
