import torchvision.transforms as transforms
import numpy as np
import torch


class Resize:
    def __init__(self, target_size, antialias=True):
        self.target_size = target_size
        self.target_height, self.target_width = target_size
        self.antialias = antialias

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            resized = self.resize(torch.from_numpy(image.astype(np.float32)))
            if resized.ndim >= 3 and resized.shape[0] > 3:
                return torch.nn.functional.softmax(resized, dim=0)
            return resized
        return self.resize(image)

    def resize(self, image):

        try:
            # image is Pillow
            original_height, original_width = image.height, image.width
        except:
            # image is tensor
            original_height, original_width = image.shape[-2], image.shape[-1]

        # Calculate the aspect ratios
        original_aspect_ratio = original_width / original_height
        target_aspect_ratio = self.target_width / self.target_height

        if original_width == self.target_width and original_height == self.target_height:
            return image
        if original_aspect_ratio == target_aspect_ratio:
            resized = transforms.functional.resize(image, self.target_size, antialias=self.antialias)
            return resized
        else:
            if original_aspect_ratio > target_aspect_ratio:
                new_height = original_height
                new_width = int(new_height * target_aspect_ratio)
            else:
                new_width = original_width
                new_height = int(new_width / target_aspect_ratio)

            # Center crop
            left = (original_width - new_width) // 2
            top = (original_height - new_height) // 2
            cropped_image = transforms.functional.crop(image, top=top, left=left, height=new_height, width=new_width)
            resized = transforms.functional.resize(cropped_image, self.target_size, antialias=self.antialias)
            return resized
