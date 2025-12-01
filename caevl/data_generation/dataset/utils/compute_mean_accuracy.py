import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import json


def compute_mean_std_online(image_dir):
    
    psum = 0
    psum_sq = 0
    image_size = None
    # Load images from directory
    with tqdm(total=len(os.listdir(image_dir))) as pbar:
        for file_name in os.listdir(image_dir):
            if file_name.endswith(('jpg', 'jpeg', 'png', '.tif', '.tiff')):
                img = Image.open(os.path.join(image_dir, file_name)).convert('L')
                img_array = np.array(img) / 255.
                psum += img_array.sum()
                psum_sq += (img_array**2).sum()
                image_size = img.size
                pbar.update(1)
    
    count = (len(os.listdir(image_dir)) * image_size[0] * image_size[1])
    
    # Compute mean and standard deviation
    mean = psum/ count
    var  = (psum_sq / count) - (mean ** 2)
    std  = var**(1/2)
    
    return mean, std

def update_json_file(json_path, mean, std):
    data = {}
    
    # Read the existing JSON file if it exists
    if os.path.exists(json_path):
        return
    
    # Update the JSON data with mean and std
    data.update({
        'mean': mean,
        'std': std
    })
    
    # Write the updated data back to the JSON file
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
