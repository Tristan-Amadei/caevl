import numpy as np
from numba import njit, prange

from caevl.data_generation.data_extraction.search_tiling.utils import compute_vector, normalize_vector


def get_corner_square_around_trajectory(trajectory: np.ndarray):
    """Find corners of the smallest square that contains all points of a given trajectory.

    Parameters
    ----------
    trajectory : np.ndarray, shape (n, 2)
        Coordinates of each point along a trajectory.

    Returns
    -------
    tuple (float, float, float, float)
        Returns the four corners of the square.
    """
    
    all_x_coords = trajectory[:, 0]
    all_y_coords = trajectory[:, 1]
    min_x, min_y = all_x_coords.min(), all_y_coords.min()
    max_x, max_y = all_x_coords.max(), all_y_coords.max()
    return min_x, min_y, max_x, max_y


def get_tiling_from_corners(min_x: float,
                            min_y: float,
                            max_x: float,
                            max_y: float,
                            step_x: float,
                            step_y: float=None,
                            margin: float=0):
    """Compute regular tiling from square corners.

    Parameters
    ----------
    min_x : float
        x dimension of lower corner point.
    min_y : float
        y dimension of lower corner point.
    max_x : float
        x dimension of upper corner point.
    max_y : float
        y dimension of upper corner point.
    step_x : float
        Tiling step on the x coordinates.
    step_y : float, optional
        Tiling step on the y coordinates, by default None.
        If None, is equal to step_x.
    margin : float, optional
        Margin to add on the borders of the tiling, by default 0.

    Returns
    -------
    np.ndarray, shape (n_y, n_x, 2)
        Coordinates of each point of the regular tiling within the given square.
    """
    
    step_y = step_y if step_y is not None else step_x
    tiling_x = np.arange(start=min_x - margin, stop=max_x + margin + step_x, step=step_x)
    tiling_y = np.arange(start=min_y - margin, stop=max_y + margin + step_y, step=step_y)
    tiling = np.zeros((len(tiling_x), len(tiling_y), 2))
    tiling[:, :, 0] = tiling_x[:, np.newaxis]
    tiling[:, :, 1] = tiling_y[np.newaxis, :]
    return tiling


@njit
def point_in_polygon(x: float, y: float, polygon_points: np.ndarray):
    """Check whether a point (x, y) is inside a polygon.
    This method uses the ray-casting algorithm.

    Parameters
    ----------
    x : float
        X coordinate.
    y : float
        Y coordinate.
    polygon_points : np.ndarray, shape (n, 2)
        Points forming the 'polygon' (might not be a completly correct one)

    Returns
    -------
    bool
        True if (x, y) is inside the polygon.
    """
    
    n = len(polygon_points)
    inside = False
    px, py = polygon_points[0]
    # px, py are the coordinates of 'preceding' point
    for i in range(n+1):
        sx, sy = polygon_points[i % n]
        # sx, sy are the coordinates of the 'current' point
        if y > min(py, sy): 
            if y <= max(py, sy):
                # y is between the coordinates of py and sy
                if x <= max(px, sx):
                    # x is on the left of segment created by 'preceding' and 'current' point
                    if py != sy:
                        # compute x-coordinate of intersection of horizontal ray with segment of the polygon
                        xinters = (y - py) * (sx - px) / (sy - py) + px
                    if px == sx or x <= xinters:
                        # if odd number of intersections, point is inside
                        inside = not inside
        px, py = sx, sy
    return inside


@njit(parallel=True)
def find_proper_tiling(tiling: np.ndarray, 
                       polygon_points: np.ndarray):
    """Check which points from a regular rectangular tiling are within a polygon,
    formed by polygon_points.

    Parameters
    ----------
    tiling : np.ndarray, shape (n_tiling, m_tiling, 2)
        Rectangular regular tiling wrapping the whole trajectory.
    polygon_points : np.ndarray, shape (n, 2)
        Points forming the 'polygon' (might not be a correct one)

    Returns
    -------
    np.ndarray, shape (n_inside, 2)
        Coordinates of tiling points within the polygon.
    """

    points = tiling.reshape(-1, 2)
    n = points.shape[0]
    points_within = np.zeros(n, dtype=np.bool_)
    
    for i in prange(n):
        points_within[i] = point_in_polygon(points[i, 0], points[i, 1], polygon_points)
    return points[points_within]


def find_points_to_consider_in_tiling(coordinates_i: np.ndarray, 
                                      coordinates_i_plus_1: np.ndarray,
                                      search_width: float,
                                      regular_tiling: np.ndarray,
                                      cap: float):
    """Between points i and (i+1) along a given trajectory, compute the points
    from a given regular tiling to consider inside a search polygon.

    Parameters
    ----------
    coordinates_i : np.ndarray, shape (2,)
        Coordinates (x, y) of point i along a trajectory.
    coordinates_i_plus_1 : np.ndarray, shape (2,)
        Coordinates (x, y) of point (i+1) along a trajectory.
    search_width : float
        Total size of search, i.e how far should the orthogonal points be from the coordinates.
    tiling : np.ndarray, shape (n_tiling, m_tiling, 2)
        Rectangular regular tiling envelopping the whole trajectory.
    cap: float
        Cap of the drone at point i in the trajectory.

    Returns
    -------
    np.ndarray, shape (n_kept, 3)
        Coordinates + cap of tiling points within the search polygon.
    """
    
    vector_between_coordinates = compute_vector(coordinates_i, coordinates_i_plus_1, normalize=False)
    orthogonal_vector = np.array([-vector_between_coordinates[1], vector_between_coordinates[0]])
    orthogonal_vector = normalize_vector(orthogonal_vector)
    
    ### COMPUTE THE 4 CORNERS OF THE SEARCH ZONE BETWEEN POINTS i AND i+1 ###
    corners = np.array([
        coordinates_i + search_width/2 * orthogonal_vector,
        coordinates_i_plus_1 + search_width/2 * orthogonal_vector,
        coordinates_i_plus_1 - search_width/2 * orthogonal_vector,
        coordinates_i - search_width/2 * orthogonal_vector,
    ])
    
    ### RESTRICT TILING AROUND ZONE TO CONSIDER ###
    # instead of considering all points in the tiling, even those very far from the zone
    # defined by the 4 corners, we restrict the tiling to search to a zone around those 4 corners
    x_min_corners, x_max_corners = corners[:, 0].min(), corners[:, 0].max()
    y_min_corners, y_max_corners = corners[:, 1].min(), corners[:, 1].max()
    
    indices_x_restriction, indices_y_restriction = np.where(
        (   
            (regular_tiling[:, :, 0] >= x_min_corners) & 
            (regular_tiling[:, :, 0] <= x_max_corners) & 
            (regular_tiling[:, :, 1] >= y_min_corners) & 
            (regular_tiling[:, :, 1] <= y_max_corners)
            )
        )
    restricted_regular_tiling = regular_tiling[indices_x_restriction, indices_y_restriction]
    
    ### CHECK WHICH POINTS FROM THE TILING ARE INSIDE THE POLYGON ###
    tiling_inside_polygone = find_proper_tiling(restricted_regular_tiling, corners)
    caps = np.ones((len(tiling_inside_polygone))) * cap
    tiling_inside_polygone = np.hstack([tiling_inside_polygone, caps[:, np.newaxis]])
    # we add the cap information to each point
    return tiling_inside_polygone
