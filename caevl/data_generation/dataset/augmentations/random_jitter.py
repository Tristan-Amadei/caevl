import torchvision.transforms.functional as TF
import random


class RandomBrightnessContrast:
    def __init__(self, brightness_factor=1, contrast_factor=2, brightness_probability=0.5, contrast_probability=0.5):
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.brightness_probability = brightness_probability
        self.contrast_probability = contrast_probability

    def __call__(self, x):
        if not isinstance(x, tuple):
            return self.apply(x)
        return (self.apply(x[0]),) + x[1:]

    def apply(self, x):
        # Apply contrast adjustment with probability contrast_probability
        if random.random() < self.contrast_probability:
            contrast_change = random.uniform(max(1 - self.contrast_factor, 0.1), 1 + self.contrast_factor)
            x = TF.adjust_contrast(x, contrast_change)

        # Apply brightness adjustment with probability brightness_probability
        if random.random() < self.brightness_probability:
            brightness_change = random.uniform(1, 1 + self.brightness_factor)
            x = TF.adjust_brightness(x, brightness_change)

        return x
