import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import random
import numpy as np

class RandomGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=1., probability=0.5, clip=True, apply_softmax=False):
        super(RandomGaussianNoise, self).__init__()
        self.mean = mean
        self.std = std
        self.probability = probability
        self.clip = clip
        self.apply_softmax = apply_softmax
        
    def forward(self, x):
        if not isinstance(x, tuple):
            return self.apply(x)
        return (self.apply(x[0]),) + x[1:]
    
    def apply(self, x):
        if isinstance(x, np.ndarray):
            return self.apply_noise(torch.from_numpy(x))
        if torch.is_tensor(x):
            return self.apply_noise(x)
        return self.apply_noise(TF.pil_to_tensor(x))
        

    def apply_noise(self, x):
        if random.random() < self.probability:
            noise = self.mean + torch.randn_like(x) * self.std
            noisy_x = x + noise
            if self.clip:
                noisy_x = torch.clamp(noisy_x, 0, 1)
            if self.apply_softmax:
                noisy_x = self._softmax(x)
            return noisy_x
        return x
    
    def _softmax(self, x):
        if x.shape[0] < x.shape[1] and x.shape[0] < x.shape[2]:
            # shape of type (c, h, w)
            return F.softmax(x, dim=0)
        # shape of type (h, w, c)
        return F.softmax(x, dim=-1)
