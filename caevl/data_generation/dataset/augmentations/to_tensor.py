import torch
import torchvision
import numpy as np


class ToTensor(torchvision.transforms.ToTensor):
    def __init__(self, process_independently=False):
        super(ToTensor, self).__init__()
        self.process_independently = process_independently    
    
    def __call__(self, x):
        if not isinstance(x, tuple):
            return self.to_tensor(x)
        return (self.to_tensor(x[0]),) + x[1:]
    
    def rearrange_channels(self, img):
        if img.shape[-3] < img.shape[-2] and img.shape[-3] < img.shape[-1]:
            return img
        return img.permute(2, 0, 1)
    
    def to_tensor(self, img):
        if self.process_independently and img.ndim >= 3:
            return torch.cat([self.to_tensor(img[:, :, i]) for i in range(img.shape[-1])], dim=0)
        
        if torch.is_tensor(img):
            return self.rearrange_channels(img)
        return super().__call__(img)