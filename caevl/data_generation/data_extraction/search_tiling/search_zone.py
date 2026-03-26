import numpy as np
import pandas as pd
from tqdm import tqdm

from caevl.data_generation.data_extraction.search_tiling.utils import getLambert93Coord, compute_vector, add_to_array
from caevl.data_generation.data_extraction.search_tiling.tiling import get_corner_square_around_trajectory, get_tiling_from_corners, find_points_to_consider_in_tiling
from caevl.data_generation.data_extraction.rotation_angle.transform_angle_rotation import transform_rot_angle


def extract_coordinates(path_to_csv_file: str,
                        step: int=1,
                        start_index: int=0,
                        stop_index: int=None,
                        name_column_latitude: str='lat_rad',
                        name_column_longitude: str='long_rad'):
    """Read the csv file at the path given, and extract coordinates (latitude, longitude).

    Parameters
    ----------
    path_to_csv_file : str
        Path to the csv file to read.
    step : int
        Sampling step.
    start_index: int, optional
        Index of first image to consider, by default 0
    stop_index: int, optional
        Index of last image to consider, by default None
    name_column_latitude : str, optional
        Name of the column in the file containing the latitudes, by default 'lat_rad'.
    name_column_longitude : str, optional
        Name of the column in the file containing the longitudes, by default 'long_rad'.

    Returns
    -------
    coordinates : ndarray, shape (n, 2)
        2D array with (latitude, longitude) for each recorded point during a flight. 
    flight_informations : pd.DataFrame
        Pandas Dataframe at specified path. 
    """

    flight_informations = pd.read_csv(path_to_csv_file)
    coordinates = flight_informations[start_index: stop_index: step]
    coordinates = coordinates[
        [name_column_latitude, name_column_longitude]
        ].values
    coordinates_in_Lambert93 = getLambert93Coord(coordinates[:, 0], coordinates[:, 1])
    return coordinates_in_Lambert93, flight_informations


def extend_coordinates(coordinates: np.ndarray,
                       distance: float):
    """Extend coordinates.
    We want to extend the search zone at the first and last point coordinates,
    so as not to brutally stop said search zone. 
    To do so, we create 'false' coordinates before the first one and after the last one.

    Parameters
    ----------
    coordinates : np.ndarray shape (n, 2)
        Coordinates of recorded points during a flight.
    distance: float
        Distance to create the 'false' points at from the true points. 
        
    Returns
    -------
    ndarray, shape (n+2, 2)
        Extended coordinates.
    """

    ### CREATE AND INSERT 'FALSE' POINT BEFORE FIRST POINT ###
    vector_between_first_and_second_point = compute_vector(coordinates[0], coordinates[1], normalize=True)
    vector_between_first_and_second_point *= distance
    point_0 = coordinates[0] - vector_between_first_and_second_point
    coordinates = np.insert(coordinates, 0, point_0, axis=0) #insert before first point coordinate
    
    ### CREATE AND INSERT 'FALSE' POINT AFTER LAST POINT ###
    vector_between_secondtolast_and_last_point = compute_vector(coordinates[-2], coordinates[-1], normalize=True)
    vector_between_secondtolast_and_last_point *= distance
    point_plus_1 = coordinates[-1] + vector_between_secondtolast_and_last_point
    coordinates = np.insert(coordinates, len(coordinates), point_plus_1, axis=0) # insert after last point coordinate
    
    return coordinates


def find_all_points_of_search_zone(path_to_csv_file: str,
                                   search_width: float,
                                   step_x: float,
                                   step_y: float=None, 
                                   frequency_sampling_points: int=1,
                                   start_index: int=0,
                                   stop_index: int=None,
                                   name_column_latitude: str='lat_rad',
                                   name_column_longitude: str='long_rad',
                                   name_column_caps: str='psi_convZYX_camSSA_rad'):
    """Compute the coordinates and orientations of IGN to sample, from
    a given trajectory in a csv file at the given path.

    Parameters
    ----------
    path_to_csv_file : str
        Path to the csv file to read.
    search_width : float
        Total size of search, i.e how far should the orthogonal points be from the coordinates.
    step_x : float
        Tiling step on the x coordinates.
    step_y : float, optional
        Tiling step on the y coordinates, by default None.
        If None, is equal to step_x.
    frequency_sampling_points : int, optional
        Frequency of points kept along the trajectory, by default 1
    start_index: int, optional
        Index of first image to consider, by default 0
    stop_index: int, optional
        Index of last image to consider, by default None
    name_column_latitude : str, optional
        Name of the column in the file containing the latitudes, by default 'lat_rad'.
    name_column_longitude : str, optional
        Name of the column in the file containing the longitudes, by default 'long_rad'.
    name_column_caps : str, optional
        Name of the column in the file containing the caps, by default 'psi_convZYX_camSSA_rad'

    Returns
    -------
    np.ndarray, shape (n_kept, 3)
        Coordinates and cap of each point to consider when extracting IGN images.
    """
    
    coordinates, flight_informations = extract_coordinates(path_to_csv_file, step=frequency_sampling_points, 
                                                           start_index=start_index, stop_index=stop_index, 
                                                           name_column_latitude=name_column_latitude, 
                                                           name_column_longitude=name_column_longitude)
    coordinates = extend_coordinates(coordinates, distance=search_width/2)
    
    caps_along_trajectory = flight_informations[start_index: stop_index: frequency_sampling_points].apply(
        lambda x: transform_rot_angle(x.psi_convZYX_camSSA_rad, x.theta_convZYX_camSSA_rad, x.phi_convZYX_camSSA_rad)[0], axis=1
        )
    caps_along_trajectory = 180 * np.array(caps_along_trajectory) / np.pi
    
    # extend caps to match the extended coordinates
    extended_caps_along_trajectory = np.zeros(len(caps_along_trajectory)+2)
    extended_caps_along_trajectory[0] = caps_along_trajectory[0]
    extended_caps_along_trajectory[1:len(caps_along_trajectory)+1] = caps_along_trajectory
    extended_caps_along_trajectory[-1] = caps_along_trajectory[-1]
    
    caps_along_trajectory = extended_caps_along_trajectory
    
    ### COMPUTE WHOLE RECTANGULAR REGULAR TILING ENVELOPPING WHOLE TRAJECTORY ###
    min_x, min_y, max_x, max_y = get_corner_square_around_trajectory(coordinates)
    margin = max(search_width/2, step_x, step_y)
    regular_tiling = get_tiling_from_corners(min_x, min_y, max_x, max_y, step_x, step_y, margin)
    
    ### ITERATE OVER EACH COORDINATE VECTOR TO FIND POINTS TO KEEP ###
    length_zone = int(0.33 * regular_tiling.shape[0] * regular_tiling.shape[1])
    # we don't know in advance how many points will be kept at the end
    # to avoid having to extend the list of points everytime, which is computationally expensive
    # we create a big list of points that should (hopefully) contain all considered points
    # we then either extend the list if it is not big enough, or restrain it at the end to remove the unnecessary part
    all_points_of_search_zone = np.zeros((length_zone, 3))
    current_index = 0
    
    for i in tqdm(range(len(coordinates)-1)):
        coordinate_i = coordinates[i]
        try:
            coordinate_i_plus_1 = 0.9 * coordinates[i+1] + 0.1 * coordinates[i+2]
        except:
            coordinate_i_plus_1 = coordinates[i+1]
        # we go a bit further than the next coordinate point to avoid avoid a few tile points
        # here and there being missed
        cap = extended_caps_along_trajectory[i]
        tiling_vector_i = find_points_to_consider_in_tiling(coordinate_i, coordinate_i_plus_1,
                                                            search_width, regular_tiling, cap)
        all_points_of_search_zone, current_index = add_to_array(
            array_to_add=tiling_vector_i, array_to_add_to=all_points_of_search_zone, index_to_start_adding_from=current_index
        )
    all_points_of_search_zone = all_points_of_search_zone[:current_index]
    ## remove potential duplicates
    all_points_of_search_zone = np.unique(all_points_of_search_zone, axis=0)
    
    return all_points_of_search_zone