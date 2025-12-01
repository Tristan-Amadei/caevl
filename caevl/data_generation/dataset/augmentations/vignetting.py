import torch
import torchvision.transforms.functional as TF
import random
import numpy as np


class RandomVignetting(torch.nn.Module):
    def __init__(self, sigma=70, probability=0.5):
        super(RandomVignetting, self).__init__()
        self.sigma = sigma
        self.probability = probability
        
    def forward(self, x):
        if not isinstance(x, tuple):
            return self.apply(x)
        return (self.apply(x[0]),) + x[1:]

    def apply(self, x):
        if random.random() < self.probability:
            return apply_vignetting(x, self.sigma)
        return x


def getGaussianFilters(ksize, sigma, double=False):
    coords = torch.arange(ksize)
    
    filters = torch.exp(-(coords - (ksize - 1) / 2)**2 / (2*sigma**2))
    return filters / torch.sum(filters)


def create_circular_mask(image_size, radius):
    center = (image_size - 1) / 2  # Assuming square image
    y, x = np.ogrid[-center:image_size-center, -center:image_size-center]
    distance_from_center = np.sqrt(x*x + y*y)
    mask = distance_from_center > radius
    return mask.astype(np.int16)


def get_vignetting_mask(width, height, sigma):
    x_filters = getGaussianFilters(height, sigma)
    y_filters = getGaussianFilters(width, sigma)
    
    kernels = x_filters[:, None] @ y_filters[:, None].T
    
    vignetting_mask = torch.clip(255 * kernels / torch.linalg.norm(kernels), min=0, max=1)
    return vignetting_mask


def apply_vignetting_torch(image, sigma):
    height, width = image.shape[-2], image.shape[-1]
    device = image.device
    
    factor = 1.75 if height != width else 1.65
    circular_mask = create_circular_mask(max(height, width), max(height, width)/factor)
    circular_mask = circular_mask[(circular_mask.shape[0]-height)//2: (circular_mask.shape[0]-height)//2+height,
                                  (circular_mask.shape[1]-width)//2: (circular_mask.shape[1]-width)//2+width]
    circular_mask = torch.tensor(circular_mask).to(device)
    
    vignetting_mask = get_vignetting_mask(width, height, sigma)
    strong_vignetting_mask = get_vignetting_mask(width, height, sigma/1.5)
    
    vignetting_mask = vignetting_mask.to(device)
    strong_vignetting_mask = strong_vignetting_mask.to(device)
    
    new_image = image * strong_vignetting_mask * (circular_mask) + image * vignetting_mask * (1 - circular_mask)
    return new_image

def apply_vignetting_PIL(image, sigma):
    height, width = image.height, image.width
    
    factor = 1.75 if height != width else 1.65
    circular_mask = create_circular_mask(max(height, width), max(height, width)/factor)
    circular_mask = circular_mask[(circular_mask.shape[0]-height)//2: (circular_mask.shape[0]-height)//2+height,
                                  (circular_mask.shape[1]-width)//2: (circular_mask.shape[1]-width)//2+width]
    
    vignetting_mask = np.array(get_vignetting_mask(width, height, sigma))
    strong_vignetting_mask = np.array(get_vignetting_mask(width, height, sigma/1.5))
    
    new_image = image * strong_vignetting_mask * (circular_mask) + image * vignetting_mask * (1 - circular_mask)
    return new_image


def apply_vignetting(image, sigma):
    if torch.is_tensor(image):
        return apply_vignetting_torch(image, sigma)
    return apply_vignetting_PIL(image, sigma)