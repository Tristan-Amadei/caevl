import torchvision


class Normalize(torchvision.transforms.Normalize):
    def __call__(self, x):
        if not isinstance(x, tuple):
            return super().__call__(x)
        return (super().__call__(x[0]),) + x[1:]
