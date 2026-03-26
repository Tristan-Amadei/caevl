import argparse
import yaml
import numpy as np
import pickle

from pathlib import Path
import sys
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from caevl.data_generation.data_extraction.search_tiling.search_zone import extract_coordinates, extend_coordinates
from caevl.data_generation.data_extraction.search_tiling.tiling import get_corner_square_around_trajectory, get_tiling_from_corners
from caevl.data_generation.data_extraction.data.dataset_generator import DatasetGenerator, make_dir
from caevl.data_generation.data_extraction.data.ign_frame_grabber import IgnFrameGrabber

import time


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



if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(description='Extract IGN images from coordinates')
    parser.add_argument('-c', '--config',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='config_IGN_vol09.yml')
    args = parser.parse_args()
    
    ## path to the config file
    path = args.filename
    if not path.endswith('.yml'):
        path += '.yml'
    
    with open(path, 'r') as config_file:
        config = yaml.safe_load(config_file)
        
    ## path to the file that contains the points defining the search zone
    path_search_zone_points = config['path_search_zone_points']
    search_zone_points = np.load(path_search_zone_points)
    
    flight_path = config['data_flight_params']['path']
    name_column_latitude = config['data_flight_params']['name_column_latitude']
    name_column_longitude = config['data_flight_params']['name_column_longitude']
    search_width = config['search_params']['search_width']
    frequency_sampling_points = config['search_params']['frequency_sampling_points']
    start_index = config['search_params']['start_index']
    stop_index = config['search_params']['stop_index']
    fine_step_x = config['search_params']['step_x']
    fine_step_y = config['search_params']['step_y']
    
    ### REDEFINE A REGULAR TILING SO AS TO SUBSAMPLE THE FINE TILING ###
    coordinates, flight_informations = extract_coordinates(flight_path, step=frequency_sampling_points, 
                                                           start_index=start_index, stop_index=stop_index, 
                                                           name_column_latitude=name_column_latitude, 
                                                           name_column_longitude=name_column_longitude)
    coordinates = extend_coordinates(coordinates, distance=search_width/2)
    
    coarse_step_x = config['dimensions']['coarse_step_x']
    coarse_step_y = config['dimensions']['coarse_step_y']
    coarse_frequency_sampling = config['dimensions']['coarse_frequency_sampling']
    min_x, min_y, max_x, max_y = get_corner_square_around_trajectory(coordinates)
    margin = max(search_width/2, fine_step_x, fine_step_y)

    fine_tiling = get_tiling_from_corners(min_x, min_y, max_x, max_y, fine_step_x, fine_step_y, margin)
    skip_frequency_step_x = int(coarse_step_x/fine_step_x)
    skip_frequency_step_y = int(coarse_step_y/fine_step_y)
    
    coarse_tiling = fine_tiling[::skip_frequency_step_x, ::skip_frequency_step_y]
    
    ### ONLY KEEP ELEMENTS FROM THE FINE TILING THAT ARE IN THE COARSE TILING ###

    coarse_tiling = {tuple(coord) for coord in coarse_tiling.reshape(-1, 2)}
    print(f'Fine-grained tiling: shape {search_zone_points.shape}')
    search_zone_points = np.array([point for point in search_zone_points if tuple(point[0:2]) in coarse_tiling])
    print(f'Coarse-grained tiling: shape {search_zone_points.shape}')
    
    
    ign_frame_grabbers = []
    nb_FrameGrabbers = len(config['FrameGrabbers'])
    print(f'There are {nb_FrameGrabbers} IgnFrameGrabbers.')
    print(f'There will be {nb_FrameGrabbers * search_zone_points.shape[0]} images extracted in total.')
    
    print('Initializing IgnFrameGrabbers.')
    for i in range(nb_FrameGrabbers):
        nb_paths = len(config['FrameGrabbers'][f'FrameGrabber_{i}'])
        print(f'IgnFrameGrabber {i} contains {nb_paths} {"databases" if nb_paths > 1 else "database"}.')
        paths = []
        for j in range(nb_paths):
            path = config['FrameGrabbers'][f'FrameGrabber_{i}'][f'path_{j}']
            paths.append(path)
        ign_frame_grabbers.append(
            IgnFrameGrabber(ign_databases=paths)
            )
        
    output_dir = config['output_dir']
    
    dataset_generator = DatasetGenerator(search_zone_points=search_zone_points, 
                                         ign_frame_grabbers=ign_frame_grabbers, 
                                         output_dir=output_dir)
    
    out_w = config['dimensions']['out_w']
    out_h = config['dimensions']['out_h']
    w = config['dimensions']['w']
    h = config['dimensions']['h']
    
    camera_depth = config['camera_depth']
    mean_altitude = flight_informations[config['data_flight_params']['name_column_altitude']].mean()

    if w is None:
        w = int(np.ceil(2 * mean_altitude * np.tan(camera_depth/2 * np.pi/180) * (1/0.2)))
        print(f'IGN width: {w}')
        
    if h is None:
        h = int(np.ceil(2 * mean_altitude * np.tan(camera_depth/2 * np.pi/180) * out_h/out_w * (1/0.2)))
        print(f'IGN height: {h}')
    
    erase_existing = config['erase_existing']
    print(f"Erase already existing files: {erase_existing}")
    crop_surplus_margin = config.get('crop_surplus_margin', True)
    print(f'Crop the surplus margin: {crop_surplus_margin}')
    max_workers = config['max_workers']
    
    dict_problems_directory = config['dict_problems_directory']
    dict_problems_save_name = config['dict_problems_save_name']
    make_dir(dict_problems_directory)
    
    dict_coordinates_directory = config['dict_coordinates_directory']
    dict_coordinates_save_name = config['dict_coordinates_save_name']
    make_dir(dict_coordinates_directory)

    dict_points_with_problem, dict_coordinates = dataset_generator.extract_all_IGN_images(IGN_width=w,
                                                                        IGN_height=h,
                                                                        out_width=out_w,
                                                                        out_height=out_h,
                                                                        erase_existing=erase_existing,
                                                                        max_workers=max_workers,
                                                                        dict_problems_directory=dict_problems_directory,
                                                                        dict_problems_save_name=dict_problems_save_name,
                                                                        dict_coordinates_directory=dict_coordinates_directory,
                                                                        dict_coordinates_save_name=dict_coordinates_save_name,
                                                                        crop=crop_surplus_margin)
    
    with open(f'{dict_problems_directory}/{dict_problems_save_name}', 'wb') as f:
        pickle.dump(dict_points_with_problem, f) 
        
    with open(f'{dict_coordinates_directory}/{dict_coordinates_save_name}', 'wb') as f:
        pickle.dump(dict_coordinates, f)
        
    stop = time.time()
    time_spent = format_time(stop - start)
    print(f'It took {time_spent} to run.')
