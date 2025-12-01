import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import sys
sys.path.append('..')

from caevl.models.pretrained.dinov2.vision_transformer import DinoVisionTransformer


class Grayscale1D_to_Grayscale3D_Shift:
    def __call__(self, img):
        # Check if the input image is a grayscale image with shape (H, W)
        if img.ndim == 2:
            img = img.unsqueeze(0)
        if img.ndim == 3:
            img = img.unsqueeze(1)
        if img.shape[1] > 1:
            return img
        
        # shift the image one byte to the left
        left_shift = torch.cat((img[:, :, 1:, :], img[:, :, 0:1, :]), dim=2)

        # shift the image one byte to the right
        right_shift = torch.cat((img[:, :, -1, :][:, :, None], img[:, :, :-1, :]), dim=2)
        
        stacked_img = img.repeat(1, 3, 1, 1)
        stacked_img[:, 1:2, :, :] = left_shift
        stacked_img[:, 2:3, :, :] = right_shift
        return stacked_img
    
class Grayscale1D_to_Grayscale3D_noShift:
    def __call__(self, img):
        # Check if the input image is a grayscale image with shape (H, W)
        if img.ndim == 2:
            img = img.unsqueeze(0)
        if img.ndim == 3:
            img = img.unsqueeze(1)
        if img.shape[1] > 1:
            return img
        
        return img.repeat(1, 3, 1, 1)
    
class ResizeClosest:
    def __call__(self, img):
        height, width = img.shape[-2], img.shape[-1]
        return transforms.functional.resize(img, (height//14*14, width//14*14), antialias=True)
    
class IdentityTransform:
    def __call__(self, img):
        return img
    
class PerceptualTransform:
    def __init__(self, grayscale_shift, no_grayscale=False):
        if no_grayscale:
            to_gray = IdentityTransform()
        else:
            to_gray = Grayscale1D_to_Grayscale3D_Shift() if grayscale_shift else Grayscale1D_to_Grayscale3D_noShift()
        self.transform = transforms.Compose([to_gray,
                                             ResizeClosest()
                                             ])
        
    def __call__(self, x):
        return self.transform(x)

class DinoV2(nn.Module):
    def __init__(self, device, load_pretrained_weights=True, requires_grad=False, 
                 add_head=False, grayscale_shift=True, no_grayscale=False):
        super(DinoV2, self).__init__()
        self.device = device

        self.transform = PerceptualTransform(grayscale_shift, no_grayscale)
        
        self.model = DinoVisionTransformer(img_size=224, 
                                           patch_size=14, 
                                           in_chans=3, 
                                           init_values=1, 
                                           num_register_tokens=0, 
                                           ffn_layer='mlp')
        
        if load_pretrained_weights:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            path_weights = os.path.join(module_dir, 'dinov2_vitb14_pretrain.pth')

            weights = torch.load(path_weights)
            keys = list(weights.keys())
            for key in keys:
                if key.startswith('blocks'):
                    split_ = key.split(".")
                    split_[0] = 'blocks.0'
                    new_key = '.'.join(split_)
                    weights[new_key] = weights.pop(key, None)
            self.model.load_state_dict(weights)
        
        self.model = self.model.to(self.device)
        self.requires_grad = requires_grad
        if not self.requires_grad:
            for param in self.model.parameters():
                param.requires_grad = False
            
        self.latent_dim = 768
        self.add_head = add_head
        if self.add_head:
            self.non_linearity = nn.GELU() 
            self.latent_dim = 500
            self.head = nn.Linear(self.model.norm.normalized_shape[0], self.latent_dim)
    
    def forward(self, images):
        dino_embeddings = self.model(self.transform(images).to(self.device))
        if self.add_head:
            dino_embeddings = self.non_linearity(dino_embeddings)
            dino_embeddings = self.head(dino_embeddings)
        return dino_embeddings
        
