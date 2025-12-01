import torch
import torchvision.transforms.functional as F

class Grayscale(torch.nn.Module):
    def __init__(self, grayscale):
        super(Grayscale, self).__init__()
        self.grayscale = grayscale
        
    def forward(self, x):
        if not isinstance(x, tuple):
            return self.apply(x)
        return (self.apply(x[0]),) + x[1:]

    def apply(self, x):
        if self.grayscale:
            return F.rgb_to_grayscale(x, num_output_channels=1)
        
        if len(x.getbands()) >= 3:
            return x
        return x.convert('RGB')
