import argparse
import yaml
import numpy as np

from pathlib import Path
import sys
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from caevl.data_generation.data_extraction.search_tiling.search_zone import find_all_points_of_search_zone

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Extract tiling points within the trajectory of a given flight.')
    parser.add_argument('-c', '--config',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='config_tiling_vol09.yml')
    args = parser.parse_args()
    
    path = args.filename
    if not path.endswith('.yml'):
        path += '.yml'
    
    with open(path, 'r') as config_file:
        config = yaml.safe_load(config_file)
        
    path_to_DAI = config['data_params']['path']
    flight_number = config['data_params']['flight_number']
    name_column_latitude = config['data_params']['name_column_latitude']
    name_column_longitude = config['data_params']['name_column_longitude']
    name_column_caps = config['data_params']['name_column_caps']
    
    search_width = config['search_params']['search_width']
    frequency_sampling_points = config['search_params']['frequency_sampling_points']
    start_index = config['search_params']['start_index']
    stop_index = config['search_params']['stop_index']
    
    step_x = config['tiling_params']['step_x']
    step_y = config['tiling_params']['step_y']
    
    print(f'Tiling for flight n°{flight_number}')
    print(f'Search width: {search_width}')
    print(f'Frequency image sampling: {frequency_sampling_points}')
    print(f'Tiling step: {step_x}')
    
    search_zone = find_all_points_of_search_zone(path_to_csv_file=path_to_DAI,
                                                 search_width=search_width,
                                                 step_x=step_x,
                                                 step_y=step_y, 
                                                 frequency_sampling_points=frequency_sampling_points,
                                                 start_index=start_index,
                                                 stop_index=stop_index,
                                                 name_column_latitude=name_column_latitude,
                                                 name_column_longitude=name_column_longitude,
                                                 name_column_caps=name_column_caps
                                                 )
    save_name = f'./data/tiling/tiling_vol{flight_number}_{search_width}_{frequency_sampling_points}_{step_x}.npy'
    np.save(save_name, search_zone)
