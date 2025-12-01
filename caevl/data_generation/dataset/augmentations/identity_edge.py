import numpy as np

class IdentityMask:
    def __init__(self):
        pass


    def identity_edge_detector(self, image):
        """
        Return same image, aligned with other edge detectors.
        """
        
        image = np.array(image).astype(np.uint8)
        image = image.squeeze()

        return image
    
    def __call__(self, x):
        if not isinstance(x, tuple):
            return self.identity_edge_detector(x)
        return (self.identity_edge_detector(x[0]),) + x[1:]