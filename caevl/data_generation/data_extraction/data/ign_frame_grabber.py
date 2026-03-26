import numpy as np
import math
import os
from tqdm import tqdm

from PIL import Image
import rasterio
from rasterio.plot import reshape_as_image
from rasterio.enums import Resampling

from caevl.data_generation.data_extraction.data.utils import *
from caevl.data_generation.data_extraction.data.ign_multiframe_grabber import *


class IgnFrameGrabber():
    """
    Class to retrieve images from a database of IGN images, corresponding to a given year and department.
    """

    def __init__(self, ign_databases, no_filling=False):
        self.dir_ign_databases = '/local_ssd/STI_STAG/2024/GNSS_DENIED/IGN/'
        self.ign_databases = ign_databases
        self.tilelist = self.compute_tile_list()
        self.tile_coords = [(int(t.split('-')[2]), int(t.split('-')[3])) for t in self.tilelist]
        self.correspondence_tilenames_databases = self.compute_dictionary_correspondence_tilenames_databases()
        self.bound_list = []
        self.bounds_global = {"left":0.0 , "right":0.0, "bottom":0.0, "top":0.0}
        self.buildBdorthoLists()
        if not no_filling:
            self.dict_filling_tiles = self.compute_dictionary_filling_tiles()
        else:
            self.dict_filling_tiles = dict()
        
        
    def compute_tile_list(self):
        """Compute the list of all tiles in all the databases given.
        If 2 tiles have the same coordinates, we keep the one with the older date (for instance, 2016 is older than 2017).

        Returns
        -------
        list
            List of all the names of tiles kept.
        """
        if isinstance(self.ign_databases, str):
            # only one ign database was given
            tilelist = os.listdir(self.ign_databases)
            return [tile for tile in tilelist if '.jp2' in tile]
        
        tilelist = os.listdir(self.ign_databases[0])
        tilelist = [tile for tile in tilelist if '.jp2' in tile]
        tile_coords = [(int(t.split('-')[2]), int(t.split('-')[3])) for t in tilelist]
        for i in range(1, len(self.ign_databases)):
            # we add tiles that do not superpose with another, and for those that do, we keep the oldest one 
            # (i.e if one from 2020 superposes with one from 2021, we keep the one from 2020)
            tilelist_to_add = os.listdir(self.ign_databases[i])
            tilelist_to_add = [tile for tile in tilelist_to_add if '.jp2' in tile]
            tile_coords_to_add = [(int(t.split('-')[2]), int(t.split('-')[3])) for t in tilelist_to_add]
            
            indices_unique = [idx for idx, coord in enumerate(tile_coords_to_add) if coord not in tile_coords]
            # indices of tiles with coordinates not already present in the list of tiles
            tilelist += [tilelist_to_add[i] for i in indices_unique]
            tile_coords = [(int(t.split('-')[2]), int(t.split('-')[3])) for t in tilelist]
            
            indices_not_unique = [idx for idx, coord in enumerate(tile_coords_to_add) if coord in tile_coords]
            for index in indices_not_unique:
                department_tile_to_add, year_tile_to_add = self.get_department_and_year_from_tile(tilelist_to_add[index])
                
                coordinates = tile_coords_to_add[index]
                index_tilelist = tile_coords.index(coordinates)
                department_tile_in_list, year_tile_in_list = self.get_department_and_year_from_tile(tilelist[index_tilelist])
                if int(year_tile_in_list) > int(year_tile_to_add):
                    tilelist[index_tilelist] = tilelist_to_add[index]
                    
            tile_coords = [(int(t.split('-')[2]), int(t.split('-')[3])) for t in tilelist]

        return tilelist
    
    
    def compute_dictionary_correspondence_tilenames_databases(self):
        """Compute the dictionary that links each tile to its IGN database.

        Returns
        -------
        dict [str, str]
            Dictionary with tiles as keys and ign_databases as values.
        """
        
        correspondence_tilenames_databases = dict()
        for tile in self.tilelist:
            for ign_database in self.ign_databases:
                if tile in os.listdir(ign_database):
                    correspondence_tilenames_databases[tile] = ign_database
        return correspondence_tilenames_databases
    
    
    def compute_dictionary_filling_tiles(self):
        """Compute the dictionary that links each tile with some missing parts to the IGN tile that can fill the missing parts up.

        Returns
        -------
        dict [str, str]
            Dictionary with tiles with missing parts as keys and path to tiles to fill them up as values.
        """
        
        dict_filling_tiles = dict() 
        for i in tqdm(range(len(self.tilelist))):
            tile_name = self.tilelist[i]
            ### Check if tile is full or has some missing parts
            has_missing_parts = check_white_pixels(os.path.join(self.correspondence_tilenames_databases[tile_name], tile_name))
            if not has_missing_parts:
                path_tile_to_fill_with = None
            else:
                _, year = self.get_department_and_year_from_tile(tile_name)
                path_tile_to_fill_with = find_tile_to_fill_with(year=int(year),
                                                                coords=(int(tile_name.split('-')[2]), int(tile_name.split('-')[3])),
                                                                dir_ign_databases=self.dir_ign_databases)
            dict_filling_tiles[tile_name] = path_tile_to_fill_with
        return dict_filling_tiles
                 

    def buildBdorthoLists(self):
        """Get all coordinates of IGN images in the base, as well as the global bounds of the whole base."""
        
        self.x_main = [1000*int(f.split('-')[2]) for f in self.tilelist]
        self.y_main = [1000*int(f.split('-')[3]) for f in self.tilelist]
        self.x_main = sorted(list(set(self.x_main)))
        self.y_main = sorted(list(set(self.y_main)))
        self.bounds_global['left']   = min(self.x_main)
        self.bounds_global['right']  = max(self.x_main)
        self.bounds_global['bottom'] = min(self.y_main)
        self.bounds_global['top']    = max(self.y_main)
        
        
    def get_department_and_year_from_tile(self, tile: str):
        """Get department number and year from a tile name.

        Parameters
        ----------
        tile : str
            Name of the tile in question.

        Returns
        -------
        tuple [str, str]
            The daprtment and year of the given tile.
        """
        
        split_ = tile.split('-')
        department = split_[0]
        year = split_[1]
        return department, year


    def findTile(self, coordinates_lambert93_pointToLocate: tuple):
        """Find the name of the ign image where the point at the given coordinates is located.

        Parameters
        ----------
        coordinates_lambert93_pointToLocate : tuple
            Coodinates of the point to locate, in Lambert 93 system.

        Returns
        -------
        str
            Name of the tile which contains the coordinates.
        """
        
        x,y = coordinates_lambert93_pointToLocate[0], coordinates_lambert93_pointToLocate[1]
        x_bound = int(getBoundValueX(x, self.x_main))
        y_bound = int(getBoundValueY(y, self.y_main))
        try:
            index = self.tile_coords.index((x_bound, y_bound))
            tile_containing_point = self.tilelist[index]
            return tile_containing_point
        except Exception as e:
            return ''


    def findPointInTile(self, coordinates_lambert93: tuple,
                        tile_name: str,
                        tile=None):
        """Find the coordinates within the IGN image of the given point with real coordinates in Lambert93.

        Parameters
        ----------
        coordinates_lambert93 : tuple
            Coodinates of the point to locate, in Lambert 93 system.
        tile_name : str
            Name of the IGN tile which contains the given coordinates. 
        tile : rasterio.io.DatasetReader, optional
            IGN tile (alreay open) which contains the given coordinates, by default None

        Returns
        -------
        tuple
            Coordinates within the IGN image of the given point.
        """
        
        if tile is None:
            tile = rasterio.open(os.path.join(self.correspondence_tilenames_databases[tile_name], tile_name))
        t = tile.transform
        t_np = np.array(t).reshape(3,3)
        t_inv = np.linalg.inv(t_np)
        coordinates_img_np = np.matmul(t_inv, np.array([coordinates_lambert93[0], coordinates_lambert93[1], 1]))
        coordinates_img = (coordinates_img_np[0], coordinates_img_np[1]) # in the OpenCV format (x,y)
        
        if tile is None:
            tile.close()
        return coordinates_img


    def find_adjacent_tile_names(self, tile):
        """Find and open all IGN tiles adjacent to a given IGN tile.

        Parameters
        ----------
        tile : str
            Name of reference tile.

        Returns
        -------
        dict[str, rasterio.io.DatasetReader]
            Dictionary with all adjacent tiles.
        """
        
        x_tile, y_tile = get_tile_position(tile)
        def open_tile(tile_name):
            if not tile_name in self.tilelist:
                coordinates = (int(tile_name.split('-')[2]), int(tile_name.split('-')[3]))
                if not coordinates in self.tile_coords:
                    return None
                index_in_tile_coords = self.tile_coords.index(coordinates)
                tile_with_right_coordinates = self.tilelist[index_in_tile_coords]
                return rasterio.open(os.path.join(self.correspondence_tilenames_databases[tile_with_right_coordinates], tile_with_right_coordinates))

            return rasterio.open(os.path.join(self.correspondence_tilenames_databases[tile_name], tile_name))
            
        adjacent_tiles = {'left'        : open_tile(change_position_in_tile_name(tile, x_tile - 5, y_tile)),
                          'right'       : open_tile(change_position_in_tile_name(tile, x_tile + 5, y_tile)),
                          'bottom'      : open_tile(change_position_in_tile_name(tile, x_tile, y_tile - 5)),
                          'top'         : open_tile(change_position_in_tile_name(tile, x_tile, y_tile + 5)),
                          'top_left'    : open_tile(change_position_in_tile_name(tile, x_tile - 5, y_tile + 5)),
                          'top_right'   : open_tile(change_position_in_tile_name(tile, x_tile + 5, y_tile + 5)),
                          'bottom_left' : open_tile(change_position_in_tile_name(tile, x_tile - 5, y_tile - 5)),
                          'bottom_right': open_tile(change_position_in_tile_name(tile, x_tile + 5, y_tile - 5)),
                          }
        return adjacent_tiles  
    
    
    def find_adjacent_filling_tiles(self, tile):
        """Find and open all IGN tiles to fill potential missing parts in tiles adjacent to a given IGN tile.

        Parameters
        ----------
        tile : rasterio.io.DatasetReader
            Reference adjacent tile.

        Returns
        -------
        dict[str, rasterio.io.DatasetReader]
            Dictionary with all adjacent tiles.
        """
        
        x_tile, y_tile = get_tile_position(tile)
        def open_filling_tile(tile_name):
            if not tile_name in self.tilelist:
                coordinates = (int(tile_name.split('-')[2]), int(tile_name.split('-')[3]))
                if not coordinates in self.tile_coords:
                    return None
                index_in_tile_coords = self.tile_coords.index(coordinates)
                tile_with_right_coordinates = self.tilelist[index_in_tile_coords]
            else:
                tile_with_right_coordinates = tile_name
            path_filling_tile = self.dict_filling_tiles.get(tile_with_right_coordinates)
            if path_filling_tile is None:
                return None
            return rasterio.open(path_filling_tile)

        adjacent_filling_tiles = {'left'        : open_filling_tile(change_position_in_tile_name(tile, x_tile - 5, y_tile)),
                                  'right'       : open_filling_tile(change_position_in_tile_name(tile, x_tile + 5, y_tile)),
                                  'bottom'      : open_filling_tile(change_position_in_tile_name(tile, x_tile, y_tile - 5)),
                                  'top'         : open_filling_tile(change_position_in_tile_name(tile, x_tile, y_tile + 5)),
                                  'top_left'    : open_filling_tile(change_position_in_tile_name(tile, x_tile - 5, y_tile + 5)),
                                  'top_right'   : open_filling_tile(change_position_in_tile_name(tile, x_tile + 5, y_tile + 5)),
                                  'bottom_left' : open_filling_tile(change_position_in_tile_name(tile, x_tile - 5, y_tile - 5)),
                                  'bottom_right': open_filling_tile(change_position_in_tile_name(tile, x_tile + 5, y_tile - 5)),
                                }
        return adjacent_filling_tiles        


    def saveAsDatasetTile(self, tile_name: str,
                                indexes, 
                                list_all_coordinates_Lambert93, 
                                output_dir: str, 
                                IGN_width: int=1000, 
                                IGN_height: int=1000, 
                                out_width: int=1000, 
                                out_height: int=1000,  
                                erase_existing: bool=True,
                                crop: bool=True):
        """Extract and save all images at given coordinates at the given indices.

        Parameters
        ----------
        tile_name : str
            Name of the IGN tile from which to extract all the images.
        indexes : array-like, shape (m,)
            Indices of the coordinates to consider from the list of all coordinates.
        list_all_coordinates_Lambert93 : array-like, shape (n, 3)
            List of all coordinates in Lambert 93 systems, as well as rotation angles.
        output_dir : str
            Path of the directory where to save all the extracted images.
        IGN_width : int, optional
            Width (in pixels) to extracto from the IGN tile, by default 1000
        IGN_height : int, optional
            Height (in pixels) to extract from the IGN tile, by default 1000
        out_width : int, optional
            Width (in pixels) of the final extracted images, by default 1000
        out_height : int, optional
            Height (in pixels) of the final extracted images, by default 1000
        erase_existing : bool, optional
            Whether to erase already existing images with same filename, by default True

        Returns
        -------
        list
            List of all indices which encountered an issue, not allowing the extraction of the image.
        dict [str, tuple]
            Dictionary that takes as keys the filenames of the saved images, and as values their coordinates.
        """

        # indices of images for which the extraction won't work
        idx_problems = []
        
        # dictionary that links images to their coordinates
        dict_coordinates = dict()

        # load the tile
        tile = rasterio.open(os.path.join(self.correspondence_tilenames_databases[tile_name], tile_name))
        t = tile.transform
        t_np = np.array(t).reshape(3,3)
        t_inv = np.linalg.inv(t_np)
        
        # load the 8 adjacent tiles
        adjacent_tiles = self.find_adjacent_tile_names(tile_name)
        
        # load tiles to fill potential missing parts in adjacent tiles
        adjacent_filling_tiles = self.find_adjacent_filling_tiles(tile_name)
        
        #### Introduction of factor to allow clean rotation augmentation
        crop_factor = np.sqrt(2)
        
        department, year = self.get_department_and_year_from_tile(tile_name)
        
        ### Check if tile is full or has some missing parts
        path_tile_to_fill_with = self.dict_filling_tiles[tile_name]
        if path_tile_to_fill_with is None:
            tile_to_fill_with = None
        else:
            tile_to_fill_with = rasterio.open(path_tile_to_fill_with)

        for i in range(len(indexes)):
            idx = indexes[i]
            
            filename = f'{idx:010d}_{department}_{year}.jpg'
            save_path = os.path.join(output_dir, filename)
            
            if os.path.exists(save_path) and not erase_existing:
                coordinates_lambert93 = list_all_coordinates_Lambert93[idx]
                dict_coordinates[filename] = coordinates_lambert93
            
            else:
                #### Get position of the tile in the IGN raw tile
                coordinates_lambert93 = list_all_coordinates_Lambert93[idx]
                rot_angle = coordinates_lambert93[-1] # cap in degrees
                rot_angle = -(90 - rot_angle) # match the cap with the rotation basis of rasterio
                coordinates_in_img_np = np.matmul(t_inv, np.array([coordinates_lambert93[0], coordinates_lambert93[1], 1]))
                coordinates_in_img = (coordinates_in_img_np[0], coordinates_in_img_np[1])
                try:
                    roi = self.getRoiDataFromOpenTile(tile,
                                                      tile_to_fill_with,
                                                      adjacent_tiles,
                                                      adjacent_filling_tiles,
                                                      coordinates_in_img,
                                                      crop_factor*IGN_width,
                                                      crop_factor*IGN_height,
                                                      math.floor(crop_factor*out_width), 
                                                      math.floor(crop_factor*out_height),
                                                      rot_angle)

                    if crop:
                        crop_height = int(roi.shape[0] / crop_factor) + 1
                        crop_width = int(roi.shape[1] / crop_factor) + 1
                        start_row = int((roi.shape[0] - crop_height) / 2)
                        start_col = int((roi.shape[1] - crop_width) / 2)
                        roi = roi[start_row:start_row+crop_height, start_col:start_col+crop_width]
                    data = Image.fromarray(roi)
                    data = data.resize((out_width, out_height), Image.ANTIALIAS)
                    data.save(save_path)
                    dict_coordinates[filename] = coordinates_lambert93

                except Exception as e:
                    # some issue was encountered during the extraction
                    idx_problems.append([idx, repr(e)])

        tile.close()
        if tile_to_fill_with is not None:
            tile_to_fill_with.close()
        for adj_tile in adjacent_tiles.values():
            if adj_tile is not None:
                adj_tile.close()

        return idx_problems, dict_coordinates


    def saveAsDatasetSample(self, coordinates_lambert93, 
                                  output_dir: str, 
                                  filename: str, 
                                  IGN_width: int=1000, 
                                  IGN_height: int=1000, 
                                  out_width: int=1000, 
                                  out_height: int=1000, 
                                  rot_angle: float=0):
        """Get ROI (Region of Interest) at given real-life coordinates in Lambert93 and save the extracted image.

        Parameters
        ----------
        coordinates_lambert93 : array-like or tuple, shape (2,)
            Coordinates of the center of the image to extract, in the Lambert 93 system.
        output_dir: str
            Path of the directory in which to save the extracted image.
        filename: str
            Name of the file to save.
        IGN_width : int, optional
            Width (in pixels) to extract from the IGN tile, by default 1000
        IGN_height : int, optional
            Height (in pixels) to extract from the IGN tile, by default 1000
        out_width : int, optional
            Width (in pixels) of the final extracted images, by default 1000
        out_height : int, optional
            Height (in pixels) of the final extracted images, by default 1000
        rot_angle : float, optional
            Rotate the extracted image by this value, by default 0

        Returns
        -------
        np.ndarray, shape (out_height, out_width, 3)
            Image extracted around the given coordinates.
        """

        try:
            roi = self.getRoiData(coordinates_lambert93, 
                                IGN_width, 
                                IGN_height, 
                                out_width, 
                                out_height,
                                rot_angle)
            data = Image.fromarray(roi)
            data.save(os.path.join(output_dir, filename))
            return True
        except:
            return False


    def getRoiDataFromOpenTile(self, tile, 
                                     tile_to_fill_with,
                                     adjacent_tiles: dict, 
                                     adjacent_filling_tiles: dict,
                                     coordinates_in_img, 
                                     IGN_width: int=1000, 
                                     IGN_height: int=1000, 
                                     out_width: int=1000, 
                                     out_height: int=1000, 
                                     rot_angle: float=0):
        """Get ROI (Region of Interest) at given coordinates within a given (already open) IGN tile.

        Parameters
        ----------
        tile : rasterio.io.DatasetReader
            IGN tile from which to extract the image.
        tile_to_fill_with : rasterio.io.DatasetReader
            Tile with wich to fill the reference tile if it has some missing parts.
        adjacent_tiles : dict
            8 IGN tiles adjacent to reference tile (tile).
        adjacent_filling_tiles : dict
            Tiles to fill potential missing parts in adjacent tiles.
        coordinates_in_img : array-like or tuple, shape (2,)
            Coordinates within the image of the center of the image to extract.
        IGN_width : int, optional
            Width (in pixels) to extract from the IGN tile, by default 1000
        IGN_height : int, optional
            Height (in pixels) to extract from the IGN tile, by default 1000
        out_width : int, optional
            Width (in pixels) of the final extracted images, by default 1000
        out_height : int, optional
            Height (in pixels) of the final extracted images, by default 1000
        rot_angle : float, optional
            Rotate the extracted image by this value, by default 0

        Returns
        -------
        np.ndarray, shape (out_height, out_width, 3)
            Image extracted around the given coordinates.
        """
        
        # compute window that contains the area of the IGN tile to extract
        window = ((coordinates_in_img[1]-3*IGN_height//4, coordinates_in_img[1]+3*IGN_height//4), (coordinates_in_img[0]-3*IGN_width//4, coordinates_in_img[0]+3*IGN_width//4))

        ### WINDOW IS WITHIN THE TILE ###
        if not (np.any(np.array(list(window)) < 0) or np.any(np.array(list(window)) > 25000)):   
            # print('direct')
            img = tile.read([1,2,3],
                            window=window,
                            out_shape=(tile.count, 3*out_height//2, 3*out_width//2),
                            resampling=Resampling.cubic)
            # img = Image.fromarray(reshape_as_image(img))
            img = reshape_as_image(img)
            
            if tile_to_fill_with is not None:
                img_to_fill_with = tile_to_fill_with.read([1,2,3],
                                                        window=window,
                                                        out_shape=(tile.count, 3*out_height//2, 3*out_width//2),
                                                        resampling=Resampling.cubic)
                img_to_fill_with = reshape_as_image(img_to_fill_with)

                fill_image(img, img_to_fill_with)
                
            img = Image.fromarray(img)
            
            if rot_angle is not None and rot_angle != 0:
                img = img.rotate(rot_angle)
            img = img.crop((out_width//4, out_height//4, 5*out_width//4, 5*out_height//4))
            
            return np.array(img, dtype=np.uint8)
        
        ### WINDOW IS WITHIN 4 TILES ###
        window_flatten = np.array(window).reshape(4)
        if (np.sum((window_flatten < 0) | (window_flatten > 25000)) == 2):
            if window_flatten[0] < 0 and window_flatten[2] < 0:
                # OVERLAP TOP LEFT CORNER
                # print('top left')
                img = regenerateLeftTop(window, tile, adjacent_tiles['left'], adjacent_tiles['top'], adjacent_tiles['top_left'], 
                                        tile_to_fill_with, adjacent_filling_tiles['left'], adjacent_filling_tiles['top'], adjacent_filling_tiles['top_left'],
                                        out_height, out_width, rot_angle)

            elif window_flatten[0] < 0 and window_flatten[3] > 25000:
                # OVERLAP TOP RIGHT CORNER
                # print('top right')
                img = regenerateTopRight(window, tile, adjacent_tiles['right'], adjacent_tiles['top'], adjacent_tiles['top_right'], 
                                         tile_to_fill_with, adjacent_filling_tiles['right'], adjacent_filling_tiles['top'], adjacent_filling_tiles['top_right'],
                                         out_height, out_width, rot_angle)
                
            elif window_flatten[1] > 25000 and window_flatten[2] < 0:
                # OVERLAP BOTTOM LEFT CORNER
                # print('bottom left')
                img = regenerateLeftBot(window, tile, adjacent_tiles['left'], adjacent_tiles['bottom'], adjacent_tiles['bottom_left'], 
                                        tile_to_fill_with, adjacent_filling_tiles['left'], adjacent_filling_tiles['bottom'], adjacent_filling_tiles['bottom_left'],
                                        out_height, out_width, rot_angle)
                
            else:
                # OVERLAP BOTTOM RIGHT CORNER
                # print('bottom right')
                img = regenerateBotRight(window, tile, adjacent_tiles['right'], adjacent_tiles['bottom'], adjacent_tiles['bottom_right'], 
                                         tile_to_fill_with, adjacent_filling_tiles['right'], adjacent_filling_tiles['bottom'], adjacent_filling_tiles['bottom_right'],
                                         out_height, out_width, rot_angle)

            return np.array(img, dtype=np.uint8)
        
        ### WINDOW IS WITHIN 2 TILES ###
        
        if window[0][0] < 0:
            # OVERLAP TOP EDGE
            # print('top')
            img = regenerateVerticalTop(tile, adjacent_tiles['top'], tile_to_fill_with, adjacent_filling_tiles['top'], window, out_height, out_width, rot_angle)
        
        elif window[0][1] > 25000:
            # OVERLAP BOTTOM EDGE
            # print('bottom')
            img = regenerateVerticalBottom(tile, adjacent_tiles['bottom'], tile_to_fill_with, adjacent_filling_tiles['bottom'], window, out_height, out_width, rot_angle)
        
        elif window[1][0] < 0:
            # OVERLAP LEFT EDGE
            # print('left')
            img = regenerateHorizontalLeft(tile, adjacent_tiles['left'], tile_to_fill_with, adjacent_filling_tiles['left'], window, out_height, out_width, rot_angle)
        
        else:
            # OVERLAP RIGHT EDGE
            # print('right')
            img = regenerateHorizontalRight(tile, adjacent_tiles['right'], tile_to_fill_with, adjacent_filling_tiles['right'], window, out_height, out_width, rot_angle)

        return np.array(img, dtype=np.uint8)
        

    def getRoiDataFromTile(self, tile_name: str, 
                                 coordinates_in_img, 
                                 IGN_width: int=1000, 
                                 IGN_height: int=1000, 
                                 out_width=1000, 
                                 out_height=1000,
                                 rot_angle=0, 
                                 crop: bool=True):
        """Get ROI (Region of Interest) at given coordinates within a given (not already open) IGN tile.

        Parameters
        ----------
        tile : str
            Name of the IGN tile from which to extract the ROI.
        coordinates_in_img : array-like or tuple, shape (2,)
            Coordinates within the image of the center of the image to extract.
        IGN_width : int, optional
            Width (in pixels) to extract from the IGN tile, by default 1000
        IGN_height : int, optional
            Height (in pixels) to extract from the IGN tile, by default 1000
        out_width : int, optional
            Width (in pixels) of the final extracted images, by default 1000
        out_height : int, optional
            Height (in pixels) of the final extracted images, by default 1000
        rot_angle : float, optional
            Rotate the extracted image by this value, by default 0

        Returns
        -------
        np.ndarray, shape (out_height, out_width, 3)
            Image extracted around the given coordinates.
        """
        
        tile = rasterio.open(os.path.join(self.correspondence_tilenames_databases[tile_name], tile_name))
        
        # load the 8 adjacent tiles
        adjacent_tiles = self.find_adjacent_tile_names(tile_name)
        adjacent_filling_tiles = self.find_adjacent_filling_tiles(tile_name)
        
        #### INTRODUCTION OF FACTOR FOR ROTATION
        crop_factor = np.sqrt(2)
        
        ### Check if tile is full or has some missing parts
        path_tile_to_fill_with = self.dict_filling_tiles.get(tile_name)
        if path_tile_to_fill_with is None:
            tile_to_fill_with = None
        else:
            tile_to_fill_with = rasterio.open(path_tile_to_fill_with)
        
        roi = self.getRoiDataFromOpenTile(tile,
                                          tile_to_fill_with,
                                          adjacent_tiles,
                                          adjacent_filling_tiles,
                                          coordinates_in_img,
                                          crop_factor*IGN_width,
                                          crop_factor*IGN_height,
                                          math.floor(crop_factor*out_width), 
                                          math.floor(crop_factor*out_height),
                                          rot_angle)              
            
        if crop:       
            crop_height = int(roi.shape[0] / crop_factor) + 1
            crop_width = int(roi.shape[1] / crop_factor) + 1
            start_row = int((roi.shape[0] - crop_height) / 2)
            start_col = int((roi.shape[1] - crop_width) / 2)
            roi = roi[start_row: start_row + crop_height, start_col: start_col + crop_width]
        data = Image.fromarray(roi)
        data = data.resize((out_width, out_height), Image.ANTIALIAS)
        data = np.array(data, dtype=np.uint8)
        
        tile.close()
        for adj_tile in adjacent_tiles.values():
            if adj_tile is not None:
                adj_tile.close()
                
        return data
    
    
    def getRoiData(self, coordinates_lambert93, 
                         IGN_width: int=1000, 
                         IGN_height: int=1000, 
                         out_width=1000, 
                         out_height=1000,
                         rot_angle=0,
                         crop: bool=True):  
        """Get ROI (Region of Interest) at given real-life coordinates in Lambert93.

        Parameters
        ----------
        coordinates_lambert93 : array-like or tuple, shape (2,)
            Coordinates of the center of the image to extract, in the Lambert 93 system.
        IGN_width : int, optional
            Width (in pixels) to extract from the IGN tile, by default 1000
        IGN_height : int, optional
            Height (in pixels) to extract from the IGN tile, by default 1000
        out_width : int, optional
            Width (in pixels) of the final extracted images, by default 1000
        out_height : int, optional
            Height (in pixels) of the final extracted images, by default 1000
        rot_angle : float, optional
            Rotate the extracted image by this value, by default 0

        Returns
        -------
        np.ndarray, shape (out_height, out_width, 3)
            Image extracted around the given coordinates.
        """
        
        ign_tile = self.findTile(coordinates_lambert93)
        coordinates_in_img = self.findPointInTile(coordinates_lambert93, ign_tile)
        
        roi = self.getRoiDataFromTile(ign_tile, 
                                      coordinates_in_img, 
                                      IGN_width, 
                                      IGN_height, 
                                      out_width, 
                                      out_height,
                                      rot_angle,
                                      crop)
        return roi
