import numpy as np
import cv2

from PIL import Image
from rasterio.plot import reshape_as_image
from rasterio.enums import Resampling

from caevl.data_generation.data_extraction.data.utils import *


########## BETWEEN 2 IGM FRAMES ##########

def read_IGN_tile(tile, filling_tile, window, out_h, out_w):
    img = tile.read([1,2,3],
                    window=window,
                    out_shape=(tile.count, 3*out_h//2, 3*out_w//2),
                    resampling=Resampling.cubic)
    img = np.array(img, dtype=np.uint8)
    img = img.transpose(1, 2, 0)
    
    if filling_tile is not None:
        img_to_fill_with = filling_tile.read([1,2,3],
                                             window=window,
                                             out_shape=(tile.count, 3*out_h//2, 3*out_w//2),
                                             resampling=Resampling.cubic)
        img_to_fill_with = np.array(img_to_fill_with, dtype=np.uint8)
        img_to_fill_with = img_to_fill_with.transpose(1, 2, 0)
        fill_image(img, img_to_fill_with)
    
    return img


def rotate_and_crop(img, rot_angle, out_h, out_w):
    img = Image.fromarray(reshape_as_image(img.transpose(2, 0, 1)))
    if rot_angle is not None and rot_angle != 0:
        img = img.rotate(rot_angle)
    img = img.crop((out_w//4, out_h//4, 5*out_w//4, 5*out_h//4))
    img = np.array(img, dtype=np.uint8)
    return img


def regenerateVerticalTop(tile_ref, top_tile, filling_ref, filling_top, window, out_h, out_w, rot_angle):
    """Reconstruct correct image from reference IGN tile and the tile atop.

    Parameters
    ----------
    tile_ref : rasterio.io.DatasetReader
        IGN tile where the center of the image is located.
    top_tile : rasterio.io.DatasetReader
        IGN tile on top of tile_ref, onto which the image is overflowing.
    filling_ref : rasterio.io.DatasetReader
        IGN tile to fill the mssing parts of the reference tile, if any.
    filling_top : rasterio.io.DatasetReader
        IGN tile to fill the mssing parts of the tile on top of tile_ref, if any.
    window : tuple
        Edges of the window to extract from the IGN images(s).
    out_h : int
        Height of the final returned image.
    out_w : int
        Width of the final returned image
    rot_angle : float
        Angle of rotation to apply to the final image.

    Returns
    -------
    np.ndarray, shape (out_h, out_width, tile.count)
        Reconstructed image.
    """
    
    # DEAL MISSING PART PARAMETERS
    # step_missing : desired part missing
    step_missing = abs(window[0][0])
    # step_lateral : step of a frame
    step_lateral = abs(window[0][1] - window[0][0])
    # length of the frame IGN
    n = 25000
    # proportion of missing step to concat and reshape images
    proportion_missing = step_missing / step_lateral

    # WINDOW of EACH TILE
    window_ref = ((window[0][0] + step_missing, window[0][1] + step_missing), 
                    (window[1][0], window[1][1]))
    window_top = ((n-step_lateral, n), 
                    (window[1][0], window[1][1]))
    
    # READ IMAGES 
    img_ref = read_IGN_tile(tile_ref, filling_ref, window_ref, out_h, out_w)
    image_top = read_IGN_tile(top_tile, filling_top, window_top, out_h, out_w)

    # GENERATE CORRECT IMAGE
    # number of pixels to remove
    idx_missing = int(proportion_missing*img_ref.shape[1])
    # remove translated surplus from reference image
    img_ref = img_ref[:(img_ref.shape[0] - idx_missing), :, :]
    # remove translated surplus from other image
    image_top = image_top[(image_top.shape[1] - idx_missing):, :, :]
    img = cv2.vconcat([image_top, img_ref])

    img = rotate_and_crop(img, rot_angle, out_h, out_w)
    return img


def regenerateVerticalBottom(tile_ref, bottom_tile, filling_ref, filling_bottom, window, out_h, out_w, rot_angle):
    """Reconstruct correct image from reference IGN tile and the tile underneath.

    Parameters
    ----------
    tile_ref : rasterio.io.DatasetReader
        IGN tile where the center of the image is located.
    bottom_tile : rasterio.io.DatasetReader
        IGN tile at bottom of tile_ref, onto which the image is overflowing.
    filling_ref : rasterio.io.DatasetReader
        IGN tile to fill the mssing parts of the reference tile, if any.
    filling_bottom : rasterio.io.DatasetReader
        IGN tile to fill the mssing parts of the tile at the bottom of tile_ref, if any.
    window : tuple
        Edges of the window to extract from the IGN images(s).
    out_h : int
        Height of the final returned image.
    out_w : int
        Width of the final returned image
    rot_angle : float
        Angle of rotation to apply to the final image.

    Returns
    -------
    np.ndarray, shape (out_h, out_width, tile.count)
        Reconstructed image.
    """
    
    n = 25_000
    step_missing = abs(window[0][1]) - n
    step_lateral = abs(window[1][1] - window[1][0])
    proportion_missing = step_missing / step_lateral

    # WINDOW of EACH TILE
    window_ref = ((window[0][0] - step_missing, window[0][1] + step_missing), 
                     (window[1][0], window[1][1]))
    window_bottom = ((0, step_lateral), 
                    (window[1][0], window[1][1]))
    
    # READ IMAGES 
    img_ref = read_IGN_tile(tile_ref, filling_ref, window_ref, out_h, out_w)
    img_bottom = read_IGN_tile(bottom_tile, filling_bottom, window_bottom, out_h, out_w)

    # GENERATE CORRECT IMAGE
    idx_missing = int(proportion_missing*img_ref.shape[1])
    img_ref = img_ref[idx_missing:, :, :]
    img_bottom = img_bottom[:idx_missing, :, :]
    img = cv2.vconcat([img_ref, img_bottom])

    img = rotate_and_crop(img, rot_angle, out_h, out_w)
    return img


def regenerateHorizontalLeft(tile_ref, left_tile, filling_ref, filling_left, window, out_h, out_w, rot_angle):
    """Reconstruct correct image from reference IGN tile and the tile on its left.

    Parameters
    ----------
    tile_ref : rasterio.io.DatasetReader
        IGN tile where the center of the image is located.
    left_tile : rasterio.io.DatasetReader
        IGN tile on the left of tile_ref, onto which the image is overflowing.
    filling_ref : rasterio.io.DatasetReader
        IGN tile to fill the mssing parts of the reference tile, if any.
    filling_left : rasterio.io.DatasetReader
        IGN tile to fill the mssing parts of the tile on the left of tile_ref, if any.
    window : tuple
        Edges of the window to extract from the IGN images(s).
    out_h : int
        Height of the final returned image.
    out_w : int
        Width of the final returned image
    rot_angle : float
        Angle of rotation to apply to the final image.

    Returns
    -------
    np.ndarray, shape (out_h, out_width, tile.count)
        Reconstructed image.
    """
    
    n = 25_000
    step_missing = abs(window[1][0])
    step_lateral = abs(window[1][1] - window[1][0])
    proportion_missing = step_missing / step_lateral

    # WINDOW of EACH TILE
    window_ref = ((window[0][0], window[0][1]), 
                    (window[1][0] + step_missing, window[1][1] + step_missing))
    window_left = ((window[0][0], window[0][1]), 
                        (n-step_lateral, n))
    
    # READ IMAGES 
    img_ref = read_IGN_tile(tile_ref, filling_ref, window_ref, out_h, out_w)
    img_left = read_IGN_tile(left_tile, filling_left, window_left, out_h, out_w)

    # GENERATE CORRECT IMAGE
    idx_missing = int(proportion_missing*img_ref.shape[0])
    img_ref = img_ref[:, :(img_ref.shape[0] - idx_missing), :]
    img_left = img_left[:, (img_left.shape[0] - idx_missing):, :]
    img = cv2.hconcat([img_left, img_ref])

    img = rotate_and_crop(img, rot_angle, out_h, out_w)
    return img


def regenerateHorizontalRight(tile_ref, right_tile, filling_ref, filling_right, window, out_h, out_w, rot_angle):
    """Reconstruct correct image from reference IGN tile and the tile on its right.

    Parameters
    ----------
    tile_ref : rasterio.io.DatasetReader
        IGN tile where the center of the image is located.
    right_tile : rasterio.io.DatasetReader
        IGN tile on the right of tile_ref, onto which the image is overflowing.
    filling_ref : rasterio.io.DatasetReader
        IGN tile to fill the mssing parts of the reference tile, if any.
    filling_right : rasterio.io.DatasetReader
        IGN tile to fill the mssing parts of the tile on the right of tile_ref, if any.
    window : tuple
        Edges of the window to extract from the IGN images(s).
    out_h : int
        Height of the final returned image.
    out_w : int
        Width of the final returned image
    rot_angle : float
        Angle of rotation to apply to the final image.

    Returns
    -------
    np.ndarray, shape (out_h, out_width, tile.count)
        Reconstructed image.
    """
    
    n = 25_000
    step_missing = abs(window[1][1]) - n
    step_lateral = abs(window[1][1] - window[1][0])
    proportion_missing = step_missing / step_lateral

    # WINDOW of EACH TILE
    window_ref = ((window[0][0], window[0][1]), 
                    (window[1][0] - step_missing, window[1][1] - step_missing))
    window_right = ((window[0][0], window[0][1]), (0, step_lateral))
    
    # READ IMAGES 
    img_ref = read_IGN_tile(tile_ref, filling_ref, window_ref, out_h, out_w)
    img_right = read_IGN_tile(right_tile, filling_right, window_right, out_h, out_w)

    # GENERATE CORRECT IMAGE
    idx_missing = int(proportion_missing*img_ref.shape[0])
    img_ref = img_ref[:, idx_missing:, :]
    img_right = img_right[:, :idx_missing, :]
    img = cv2.hconcat([img_ref, img_right])

    img = rotate_and_crop(img, rot_angle, out_h, out_w)
    return img


########## BETWEEN 4 IGM FRAMES ##########

def regenerateLeftTop(window, tile_ref, tile_left, tile_top, tile_top_left, 
                      filling_ref, filling_left, filling_top, filling_top_left,
                      out_h, out_w, rot_angle):
    """Reconstruct correct image from reference IGN tile and the tiles atop, on the left and top left corner.

    Parameters
    ----------
    window : tuple
        Edges of the window to extract from the IGN images(s).
    tile_ref : rasterio.io.DatasetReader
        IGN tile where the center of the image is located
    tile_left : rasterio.io.DatasetReader
        IGN tile on the left of tile_ref, onto which the image is overflowing.
    tile_top : rasterio.io.DatasetReader
        IGN tile on top of tile_ref, onto which the image is overflowing.
    tile_top_left : rasterio.io.DatasetReader
        IGN tile on the top left corner of tile_ref, onto which the image is overflowing.
    filling_ref : rasterio.io.DatasetReader
        IGN tile to fill the mssing parts of the reference tile, if any.
    filling_left : rasterio.io.DatasetReader
        IGN tile to fill the mssing parts of the tile on the left of tile_ref, if any.
    filling_top : rasterio.io.DatasetReader
        IGN tile to fill the mssing parts of the tile on the top of tile_ref, if any.
    filling_top_left : rasterio.io.DatasetReader
        IGN tile to fill the mssing parts of the tile on the top left of tile_ref, if any.
    out_h : int
        Height of the final returned image.
    out_w : int
        Width of the final returned image
    rot_angle : float
        Angle of rotation to apply to the final image.

    Returns
    -------
    np.ndarray, shape (out_h, out_width, tile.count)
        Reconstructed image.
    """
    
    ### WINDOW
    # parameters of the IGN tiles
    n = 25000
    step_lateral = abs(window[0][1] - window[0][0])

    # missing part
    step_missing_top = abs(window[0][0])
    step_missing_left = abs(window[1][0])

    # initialize windows
    window_ref = ((0, step_lateral), (0, step_lateral))
    window_top = ((n - step_lateral, n), (0, step_lateral))
    window_left = ((0, step_lateral), (n-step_lateral, n))
    window_top_left = ((n-step_lateral, n), (n-step_lateral, n))

    # missing proportions
    proportion_missing_top = step_missing_top / step_lateral
    proportion_missing_left = step_missing_left / step_lateral

    ### OPEN WINDOWS
    # REFERENCE TILE
    image_ref = read_IGN_tile(tile_ref, filling_ref, window_ref, out_h, out_w)
    
    # OTHER TILE TOP
    image_top = read_IGN_tile(tile_top, filling_top, window_top, out_h, out_w)
    
    # OTHER TILE : LEFT OF REFERENCE
    image_left = read_IGN_tile(tile_left, filling_left, window_left, out_h, out_w)
    
    # OTHER TILE : LEFT OF TOP
    image_top_left = read_IGN_tile(tile_top_left, filling_top_left, window_top_left, out_h, out_w)

    # CROP DESIRED POSITION
    m = image_ref.shape[0]
    idx_missing_top = int(proportion_missing_top*m)
    idx_missing_left = int(proportion_missing_left*m)
    image_ref = image_ref[:(m-idx_missing_top), :(m-idx_missing_left), :]
    image_top = image_top[(m-idx_missing_top):, :(m-idx_missing_left), :]
    image_left = image_left[:(m-idx_missing_top), (m-idx_missing_left):, :]
    image_top_left = image_top_left[(m-idx_missing_top):, (m-idx_missing_left):, :]

    # CONCATENATE TO MAKE A FULL IMAGE
    image_bottom = np.concatenate((image_left, image_ref), axis=1)
    image_top = np.concatenate((image_top_left, image_top), axis=1)
    image_full = np.concatenate((image_top, image_bottom), axis=0)

    img = rotate_and_crop(image_full, rot_angle, out_h, out_w)

    return img


def regenerateLeftBot(window, tile_ref, tile_left, tile_bottom, tile_bottom_left, 
                      filling_ref, filling_left, filling_bottom, filling_bottom_left,
                      out_h, out_w, rot_angle):
    """Reconstruct correct image from reference IGN tile and the tiles atop, on the left and bottom left corner.

    Parameters
    ----------
    window : tuple
        Edges of the window to extract from the IGN images(s).
    tile_ref : rasterio.io.DatasetReader
        IGN tile where the center of the image is located
    tile_left : rasterio.io.DatasetReader
        IGN tile on the left of tile_ref, onto which the image is overflowing.
    tile_bottom : rasterio.io.DatasetReader
        IGN tile at bottom of tile_ref, onto which the image is overflowing.
    tile_bottom_left : rasterio.io.DatasetReader
        IGN tile on the bottom left corner of tile_ref, onto which the image is overflowing.
    filling_ref : rasterio.io.DatasetReader
        IGN tile to fill the mssing parts of the reference tile, if any.
    filling_left : rasterio.io.DatasetReader
        IGN tile to fill the mssing parts of the tile on the left of tile_ref, if any.
    filling_bottom : rasterio.io.DatasetReader
        IGN tile to fill the mssing parts of the tile at the bottom of tile_ref, if any.
    filling_bottom_left : rasterio.io.DatasetReader
        IGN tile to fill the mssing parts of the tile on the bottom left of tile_ref, if any.
    out_h : int
        Height of the final returned image.
    out_w : int
        Width of the final returned image
    rot_angle : float
        Angle of rotation to apply to the final image.

    Returns
    -------
    np.ndarray, shape (out_h, out_width, tile.count)
        Reconstructed image.
    """
    
    ### WINDOW
    # parameters of the IGN tiles
    n = 25000
    step_lateral = abs(window[0][1] - window[0][0])

    # missing part
    step_missing_bot = abs(window[0][1]) - n
    step_missing_left = abs(window[1][0])

    # initialize windows
    window_ref = ((n-step_lateral, n), (0, step_lateral))
    window_bot = ((0, step_lateral), (0, step_lateral))
    window_left = ((n-step_lateral, n), (n-step_lateral, n))
    window_bot_left = ((0, step_lateral), (n-step_lateral, n))

    # missing proportions
    proportion_missing_bot = step_missing_bot / step_lateral
    proportion_missing_left = step_missing_left / step_lateral

    ### OPEN WINDOWS
    # REFERENCE TILE
    image_ref = read_IGN_tile(tile_ref, filling_ref, window_ref, out_h, out_w)
    
    # OTHER TILE TOP
    image_bot = read_IGN_tile(tile_bottom, filling_bottom, window_bot, out_h, out_w)
    
    # OTHER TILE : LEFT OF REFERENCE
    image_left = read_IGN_tile(tile_left, filling_left, window_left, out_h, out_w)
    
    # OTHER TILE : LEFT OF TOP
    image_bot_left = read_IGN_tile(tile_bottom_left, filling_bottom_left, window_bot_left, out_h, out_w)

    # CROP DESIRED POSITION
    m = image_ref.shape[0]
    idx_missing_bot = int(proportion_missing_bot*m)
    idx_missing_left = int(proportion_missing_left*m)
    image_ref = image_ref[idx_missing_bot:, :(m-idx_missing_left), :]
    image_bot = image_bot[:idx_missing_bot, :(m-idx_missing_left), :]
    image_bot_left = image_bot_left[:idx_missing_bot, (m-idx_missing_left):, :]
    image_left = image_left[idx_missing_bot:, (m-idx_missing_left):, :]

    # CONCATENATE TO MAKE A FULL IMAGE
    image_bottom = np.concatenate((image_bot_left, image_bot), axis=1)
    image_top = np.concatenate((image_left, image_ref), axis=1)
    image_full = np.concatenate((image_top, image_bottom), axis=0)

    img = rotate_and_crop(image_full, rot_angle, out_h, out_w)

    return img


def regenerateBotRight(window, tile_ref, tile_right, tile_bottom, tile_bottom_right, 
                       filling_ref, filling_right, filling_bottom, filling_bottom_right,
                       out_h, out_w, rot_angle):
    """Reconstruct correct image from reference IGN tile and the tiles at the bottom, on the right and bottom right corner.

    Parameters
    ----------
    window : tuple
        Edges of the window to extract from the IGN images(s).
    tile_ref : rasterio.io.DatasetReader
        IGN tile where the center of the image is located
    tile_right : rasterio.io.DatasetReader
        IGN tile on the right of tile_ref, onto which the image is overflowing.
    tile_bottom : rasterio.io.DatasetReader
        IGN tile at bottom of tile_ref, onto which the image is overflowing.
    tile_bottom_right : rasterio.io.DatasetReader
        IGN tile on the bottom right corner of tile_ref, onto which the image is overflowing.
    filling_ref : rasterio.io.DatasetReader
        IGN tile to fill the mssing parts of the reference tile, if any.
    filling_right : rasterio.io.DatasetReader
        IGN tile to fill the mssing parts of the tile on the right of tile_ref, if any.
    filling_bottom : rasterio.io.DatasetReader
        IGN tile to fill the mssing parts of the tile at the bottom of tile_ref, if any.
    filling_bottom_right : rasterio.io.DatasetReader
        IGN tile to fill the mssing parts of the tile on the bottom right of tile_ref, if any.
    out_h : int
        Height of the final returned image.
    out_w : int
        Width of the final returned image
    rot_angle : float
        Angle of rotation to apply to the final image.

    Returns
    -------
    np.ndarray, shape (out_h, out_width, tile.count)
        Reconstructed image.
    """
    
    ### WINDOW
    # parameters of the IGN tiles
    n = 25000
    step_lateral = abs(window[0][1] - window[0][0])

    # missing part
    step_missing_bot = abs(window[0][1]) - n
    step_missing_right = abs(window[1][1]) - n

    # initialize windows
    window_ref = ((n-step_lateral, n), (n-step_lateral, n))
    window_bot = ((0, step_lateral), (n-step_lateral, n))
    window_right = ((n-step_lateral, n), (0, step_lateral))
    window_bot_right = ((0, step_lateral), (0, step_lateral))

    # missing proportions
    proportion_missing_bot = step_missing_bot / step_lateral
    proportion_missing_right = step_missing_right / step_lateral

    ### OPEN WINDOWS
    # REFERENCE TILE
    image_ref = read_IGN_tile(tile_ref, filling_ref, window_ref, out_h, out_w)
    
    # OTHER TILE TOP
    image_bot = read_IGN_tile(tile_bottom, filling_bottom, window_bot, out_h, out_w)
    
    # OTHER TILE : LEFT OF REFERENCE
    image_right = read_IGN_tile(tile_right, filling_right, window_right, out_h, out_w)
    
    # OTHER TILE : LEFT OF TOP
    image_bot_right = read_IGN_tile(tile_bottom_right, filling_bottom_right, window_bot_right, out_h, out_w)

    # CROP DESIRED POSITION
    m = image_ref.shape[0]
    idx_missing_bot = int(proportion_missing_bot*m)
    idx_missing_right = int(proportion_missing_right*m)
    image_ref = image_ref[idx_missing_bot:, idx_missing_right:, :]
    image_bot = image_bot[:idx_missing_bot, idx_missing_right:, :]
    image_bot_right = image_bot_right[:idx_missing_bot, :idx_missing_right, :]
    image_right = image_right[idx_missing_bot:, :idx_missing_right, :]

    # CONCATENATE TO MAKE A FULL IMAGE
    # DISPLAY FULL IMAGE
    image_bottom = np.concatenate((image_bot, image_bot_right), axis=1)
    image_top = np.concatenate((image_ref, image_right), axis=1)
    image_full = np.concatenate((image_top, image_bottom), axis=0)

    img = rotate_and_crop(image_full, rot_angle, out_h, out_w)

    return img


def regenerateTopRight(window, tile_ref, tile_right, tile_top, tile_top_right, 
                       filling_ref, filling_right, filling_top, filling_top_right,
                       out_h, out_w, rot_angle):
    """Reconstruct correct image from reference IGN tile and the tiles atop, on the right and top right corner.

    Parameters
    ----------
    window : tuple
        Edges of the window to extract from the IGN images(s).
    tile_ref : rasterio.io.DatasetReader
        IGN tile where the center of the image is located
    tile_right : rasterio.io.DatasetReader
        IGN tile on the right of tile_ref, onto which the image is overflowing.
    tile_top : rasterio.io.DatasetReader
        IGN tile on top of tile_ref, onto which the image is overflowing.
    tile_top_right : rasterio.io.DatasetReader
        IGN tile on the top right corner of tile_ref, onto which the image is overflowing.
    filling_ref : rasterio.io.DatasetReader
        IGN tile to fill the mssing parts of the reference tile, if any.
    filling_right : rasterio.io.DatasetReader
        IGN tile to fill the mssing parts of the tile on the right of tile_ref, if any.
    filling_top : rasterio.io.DatasetReader
        IGN tile to fill the mssing parts of the tile on top of tile_ref, if any.
    filling_top_right : rasterio.io.DatasetReader
        IGN tile to fill the mssing parts of the tile on the top right of tile_ref, if any.
    out_h : int
        Height of the final returned image.
    out_w : int
        Width of the final returned image
    rot_angle : float
        Angle of rotation to apply to the final image.

    Returns
    -------
    np.ndarray, shape (out_h, out_width, tile.count)
        Reconstructed image.
    """
    
    ### WINDOW
    # parameters of the IGN tiles
    n = 25000
    step_lateral = abs(window[0][1] - window[0][0])

    # missing part
    step_missing_top = abs(window[0][0])
    step_missing_right = abs(window[1][1]) - n

    # initialize windows
    window_ref = ((0, step_lateral), (n-step_lateral, n))
    window_top = ((n-step_lateral, n), (n-step_lateral, n))
    window_right = ((0, step_lateral), (0, step_lateral))
    window_top_right = ((n-step_lateral, n), (0, step_lateral))

    # missing proportions
    proportion_missing_top = step_missing_top / step_lateral
    proportion_missing_right = step_missing_right / step_lateral

    ### OPEN WINDOWS
    # REFERENCE TILE
    image_ref = read_IGN_tile(tile_ref, filling_ref, window_ref, out_h, out_w)
    
    # OTHER TILE TOP
    image_top = read_IGN_tile(tile_top, filling_top, window_top, out_h, out_w)
    
    # OTHER TILE : LEFT OF REFERENCE
    image_right = read_IGN_tile(tile_right, filling_right, window_right, out_h, out_w)
    
    # OTHER TILE : LEFT OF TOP
    image_top_right = read_IGN_tile(tile_top_right, filling_top_right, window_top_right, out_h, out_w)

    # CROP DESIRED POSITION
    m = image_ref.shape[0]
    idx_missing_top = int(proportion_missing_top*m)
    idx_missing_right = int(proportion_missing_right*m)
    image_ref = image_ref[:(m-idx_missing_top), idx_missing_right:, :]
    image_top = image_top[(m-idx_missing_top):, idx_missing_right:, :]
    image_top_right = image_top_right[(m-idx_missing_top):, :idx_missing_right, :]
    image_right = image_right[:(m-idx_missing_top), :idx_missing_right, :]

    # CONCATENATE TO MAKE A FULL IMAGE
    image_bottom = np.concatenate((image_ref, image_right), axis=1)
    image_top = np.concatenate((image_top, image_top_right), axis=1)
    image_full = np.concatenate((image_top, image_bottom), axis=0)

    img = rotate_and_crop(image_full, rot_angle, out_h, out_w)

    return img
