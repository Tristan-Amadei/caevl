import torchvision.transforms.functional as TF
import random

class RandomGaussianBlur:
    def __init__(self, kernel_size=3, blur_probability=0.5, sigma=1):
        self.kernel_size = kernel_size
        self.blur_probability = blur_probability
        self.sigma = sigma

    def __call__(self, x):
        random_ = random.random()
        if not isinstance(x, tuple):
            if random_ < self.blur_probability:
                return TF.gaussian_blur(x, self.kernel_size, sigma=self.sigma)
            return x
        return (self(x[0]),) + x[1:]
