import numpy as np
import pyproj


def getLambert93Coord(lat_rad: np.ndarray, 
                      long_rad: np.ndarray):
    """Convert coordinates to Lambert93 system.

    Parameters
    ----------
    lat_rad : np.ndarray, shape (n, )
        Latitudes to convert.
    long_rad : np.ndarray, shape (n, )
        Longitudes to convert.

    Returns
    -------
    np.ndarray, shape (n, 2)
        Array containing the converted coordinates, stacked in columns.
    """
    gps = pyproj.CRS('EPSG:4326')  
    lambert93 = pyproj.CRS('EPSG:2154') 
    transformer = pyproj.Transformer.from_crs(gps, lambert93, always_xy=True)

    long_la93, lat_la93 = [], []
    for (lat, long) in zip(lat_rad, long_rad):
        lat_deg = 180*lat/np.pi
        long_deg = 180*long/np.pi
        x, y = transformer.transform(long_deg, lat_deg)
        long_la93.append(x)
        lat_la93.append(y)
    long_la93 = np.array(long_la93)
    lat_la93 = np.array(lat_la93)
    stacked_array = np.array(list(zip(long_la93, lat_la93))) 
    return stacked_array


def normalize_vector(vector: np.ndarray):
    """Return a normalized vector.

    Parameters
    ----------
    vector : np.ndarray
        Vector to normalize. 

    Returns
    -------
    np.ndarray
        Normalized vector, shape unchanged.
    """
    
    norm = np.linalg.norm(vector)
    vector = vector if norm == 0 else vector / norm
    return vector


def compute_vector(x, y, normalize=False):
    """Compute vector between x and y.

    Parameters
    ----------
    x : tuple, list or np.ndarray, shape (2,)
        Coordinates of point x.
    y : tuple, list or np.ndarray, shape (2,)
        Coordinates of point y.
    normalize: bool
        If True, return a normalized vector.

    Returns
    -------
    np.ndarray, shape (2,)
        Vector between x and y.
    """

    vector = np.array([y[0] - x[0], y[1] - x[1]])
    if normalize:
        vector = normalize_vector(vector)
    return vector


def add_to_array(array_to_add: np.ndarray, 
                 array_to_add_to: np.ndarray,
                 index_to_start_adding_from: int):
    """
    Insert elements from `array_to_add` into `array_to_add_to` at the specified index.

    If there is sufficient space in `array_to_add_to` to accommodate
    `array_to_add` starting from `index_to_start_adding_from`, the elements
    are inserted directly. Otherwise, `array_to_add_to` is extended by
    concatenating `array_to_add` to it.

    Parameters
    ----------
    array_to_add : numpy.ndarray
        The array with elements to be added to `array_to_add_to`.
    array_to_add_to : numpy.ndarray
        The array into which `array_to_add` will be inserted.
    index_to_start_adding_from : int
        The starting index in `array_to_add_to` where addition will begin.

    Returns
    -------
    modified_array : numpy.ndarray
        The `array_to_add_to` after insertion of `array_to_add`.
    new_index : int
        The index in `modified_array` immediately following the last
        inserted element of `array_to_add`.
    """
    
    if index_to_start_adding_from + len(array_to_add) < len(array_to_add_to):
        # the array that receives is big enough to receive the whole array to add
        array_to_add_to[index_to_start_adding_from: index_to_start_adding_from+len(array_to_add)] = array_to_add
        index_to_start_adding_from += len(array_to_add)
        return array_to_add_to, index_to_start_adding_from
    
    # otherwise, we need to extend the array
    array_to_add_to = array_to_add_to[:index_to_start_adding_from]
    array_to_add_to = np.concatenate([
        array_to_add_to, array_to_add
    ])
    index_to_start_adding_from = len(array_to_add_to)
    return array_to_add_to, index_to_start_adding_from
