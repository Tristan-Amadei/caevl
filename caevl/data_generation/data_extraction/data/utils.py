import rasterio
import numpy as np
from glob import glob
import os


def genImageFilename(u):
        return str(u).zfill(10) + '.jpg'


def getBoundValueX(val, l):
    if val > max(l) or val < min(l):
        return 0
    for i in range(1, len(l)):
        if l[i-1] <= val <= l[i]:
            return f'{l[i-1]//1000:04d}'


def getBoundValueY(val, l):
    if val > max(l) or val < min(l):
        return 0
    for i in range(1, len(l)):
        if l[i-1] <= val <= l[i]:
            return f'{l[i]//1000:04d}'
        

def get_tile_position(tile):
    split_ = tile.split('-')
    x = int(split_[2])
    y = int(split_[3])
    return x, y


def change_position_in_tile_name(tile_name, updated_x, updated_y):
    split_ = tile_name.split('-')
    updated_tile_name = str(split_[0]) + '-' + \
                str(split_[1]) + '-'+ \
                f'{updated_x:04d}' + '-'+ \
                f'{updated_y:04d}' + '-'+ \
                str(split_[4]) + '-' + \
                str(split_[5]) + '-' + \
                str(split_[6]) 
    return updated_tile_name


def check_white_pixels_im(image, threshold=1/25):
    image_np = np.array(image)
    white_pixel_count = np.sum(image_np == [255, 255, 255])
    white_pixel_ratio = white_pixel_count / (image_np.shape[0] * image_np.shape[1])

    return white_pixel_ratio >= threshold


def check_white_pixels(path, threshold=1/25):
    im = rasterio.open(path)
    im = rasterio.plot.reshape_as_image(im.read([1, 2, 3], out_shape=(3, 2048, 2048)))
    return check_white_pixels_im(im, threshold)


def fill_image(im_missing_parts, im_full):
    im_missing_parts[im_missing_parts == [255, 255, 255]] = im_full[im_missing_parts == [255, 255, 255]]
    
    
def find_index_in_list(element, list_):
    for i in range(len(list_)):
        if list_[i].startswith(element):
            return i
    return -1
    

def list_tiles(ign_database):
    cur_path = os.path.join(ign_database, 'ORTHOHR')
    index_donnees_livraison = find_index_in_list('1', list(os.walk(cur_path))[0][1])
    donnees_livraison = list(os.walk(cur_path))[0][1][index_donnees_livraison]
    cur_path = os.path.join(cur_path, donnees_livraison)
    dir_images = list(os.walk(cur_path))[0][1][0]
    cur_path = os.path.join(cur_path, dir_images)
    tiles = os.listdir(cur_path)
    tiles = [tile for tile in tiles if '.jp2' in tile]
    tile_coords = [(int(t.split('-')[2]), int(t.split('-')[3])) for t in tiles]
    return cur_path, tiles, tile_coords
    

def find_tile_to_fill_with(year, coords, dir_ign_databases):
    cur_delta_year = 0
    nb_databases_visited = 0
    nb_total_databases = len(glob(os.path.join(dir_ign_databases,'*0M20*')))
    while nb_databases_visited < nb_total_databases:
        if cur_delta_year == 0:
            ign_databases_in_delta = glob(os.path.join(dir_ign_databases,f'*0M20*{year}*'))
        else:
            ign_databases_in_delta = glob(os.path.join(dir_ign_databases,f'*0M20*{year-cur_delta_year}*')) + glob(os.path.join(dir_ign_databases,f'*0M20*{year+cur_delta_year}*'))
        
        for ign_database in ign_databases_in_delta:
            path_images, tiles, tile_coords = list_tiles(ign_database)
            for i, coords_ in enumerate(tile_coords):
                if coords == coords_:
                    path_tile_with_same_coords = os.path.join(path_images, tiles[i])
                    if not check_white_pixels(path_tile_with_same_coords):
                        return path_tile_with_same_coords
            nb_databases_visited += 1
            
        cur_delta_year += 1
    return None

