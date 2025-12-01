import cv2
import numpy as np

import numpy as np

cv2.setNumThreads(1)

class CannyMask:
    def __init__(self, low_threshold=100, high_threshold=200, 
                 apertureSize=3, L2gradient=False):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.apertureSize = apertureSize
        self.L2gradient = L2gradient

    def get_edges(self, img):
        
        # Apply Canny edge detection
        img = np.array(img).astype(np.uint8)
        img = img.squeeze()
        edges = cv2.Canny(img, self.low_threshold, self.high_threshold, 
                          apertureSize=self.apertureSize, L2gradient=self.L2gradient)
        
        # Binarize the image (edges will be 255, rest will be 0)
        binary_img = (255 * (edges > 0)).astype(np.uint8)
                
        return binary_img
    
    def get_heatmap(self, img):
        img = np.array(img).astype(np.uint8)
        img = img.squeeze()
        heatmap = canny_heatmap(img, sigma=self.sigma).astype(np.float32)
        return heatmap
    
    def apply(self, img):
        return self.get_edges(img)
    
    def __call__(self, x):
        if not isinstance(x, tuple):
            return self.apply(x)
        return (self.apply(x[0]),) + x[1:]
    


def _preprocess(image, mask, sigma, mode, cval):

    import scipy.ndimage as ndi
    from skimage._shared.filters import gaussian
    from skimage._shared.utils import _supported_float_type

    gaussian_kwargs = dict(sigma=sigma, mode=mode, cval=cval, preserve_range=False)
    compute_bleedover = mode == 'constant' or mask is not None
    float_type = _supported_float_type(image.dtype)
    if mask is None:
        if compute_bleedover:
            mask = np.ones(image.shape, dtype=float_type)
        masked_image = image

        eroded_mask = np.ones(image.shape, dtype=bool)
        eroded_mask[:1, :] = 0
        eroded_mask[-1:, :] = 0
        eroded_mask[:, :1] = 0
        eroded_mask[:, -1:] = 0

    else:
        mask = mask.astype(bool, copy=False)
        masked_image = np.zeros_like(image)
        masked_image[mask] = image[mask]

        # Make the eroded mask. Setting the border value to zero will wipe
        # out the image edges for us.
        s = ndi.generate_binary_structure(2, 2)
        eroded_mask = ndi.binary_erosion(mask, s, border_value=0)

    if compute_bleedover:
        # Compute the fractional contribution of masked pixels by applying
        # the function to the mask (which gets you the fraction of the
        # pixel data that's due to significant points)
        bleed_over = (
            gaussian(mask.astype(float_type, copy=False), **gaussian_kwargs)
            + np.finfo(float_type).eps
        )

    # Smooth the masked image
    smoothed_image = gaussian(masked_image, **gaussian_kwargs)

    # Lower the result by the bleed-over fraction, so you can
    # recalibrate by dividing by the function on the mask to recover
    # the effect of smoothing from just the significant pixels.
    if compute_bleedover:
        smoothed_image /= bleed_over

    return smoothed_image, eroded_mask


def canny_heatmap(
    image,
    sigma=1.0,
    mask=None,
    mode='constant',
    cval=0.0,
):

    import scipy.ndimage as ndi
    from skimage._shared.utils import check_nD
    from skimage.util.dtype import dtype_limits

    if np.issubdtype(image.dtype, np.int64) or np.issubdtype(image.dtype, np.uint64):
        raise ValueError("64-bit integer images are not supported")

    check_nD(image, 2)
    dtype_max = dtype_limits(image, clip_negative=False)[1]

    # Image filtering
    smoothed, eroded_mask = _preprocess(image, mask, sigma, mode, cval)

    # Gradient magnitude estimation
    jsobel = ndi.sobel(smoothed, axis=1)
    isobel = ndi.sobel(smoothed, axis=0)
    magnitude = isobel * isobel
    magnitude += jsobel * jsobel
    np.sqrt(magnitude, out=magnitude)
    return magnitude

