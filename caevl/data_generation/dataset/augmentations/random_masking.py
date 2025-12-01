import numpy as np
import torch
import random
import torchvision.transforms.functional as F

import os
module_dir = os.path.dirname(os.path.abspath(__file__))
binary_mask_path = os.path.join(module_dir, 'binary_mask.npy')

class RandomMasking(torch.nn.Module):
    def __init__(self, probability=0.5):
        super(RandomMasking, self).__init__()
        self.probability = probability
        self.mask = torch.tensor(np.load(binary_mask_path)).to(torch.float32)
        
    def forward(self, x):
        if not isinstance(x, tuple):
            return self.apply(x)
        return (self.apply(x[0]),) + x[1:]

    def apply(self, x):
        if random.random() < self.probability:
            device = x.device

            height, width = x.shape[-2], x.shape[-1]
            mask = F.resize(self.mask.unsqueeze(0), (height, width), antialias=True)
            mask = torch.where(mask >= 0.5, 1, 0)
            mask = mask.repeat(x.shape[0], 1, 1) # if x has more than 1 channel, mask all channels at given pixels
            return x * mask.to(device)

        return x