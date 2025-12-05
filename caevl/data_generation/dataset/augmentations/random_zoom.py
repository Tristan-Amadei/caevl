import random
import math
import numpy as np

import torchvision
import torchvision.transforms.functional as TF


def get_height_width(x):
    try:
        height = x.shape[-2]
        width = x.shape[-1]
    except:
        height, width = x.height, x.width
    return height, width


class RandomZoomIn(torchvision.transforms.RandomResizedCrop):
    def __init__(self, output_size, degrees, translate, probability=0.5,
                 ratio=(1, 1), scale=(0.7, 1), return_locations=False, locations=None):
        super(RandomZoomIn, self).__init__(output_size, ratio=ratio, scale=scale)
        self.probability = probability
        self.output_size = output_size
        self.degrees = degrees
        self.translate = translate
        self.return_locations = return_locations

        self.rotation_object = RandomRotationWithProbability(degrees=self.degrees, rotation_probability=1,
                                                             return_locations=self.return_locations)
        self.translation_object = TranslateAndCrop(translate=self.translate, probability=1, return_locations=self.return_locations)

    def _resized_crop(self, img):
        height, width = get_height_width(img)
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        cropped_img = TF.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        if self.return_locations:
            return cropped_img, (i, j, h, w, height, width)
        return cropped_img

    def forward(self, img):
        if random.random() < self.probability:
            draw = random.random()
            if draw < 1 / 3:
                ## rotation + crop + resize
                return self.rotation_object(img)
            if draw < 2 / 3:
                return self.translation_object(img)
            return self._resized_crop(img)
        if self.return_locations:
            height, width = get_height_width(img)
            return img, (0, 0, height, width, height, width)
        return img


### RANDOM ROTATION & CROP ###


def _largest_rotated_rect(w, h, angle):

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return bb_w - 2 * x, bb_h - 2 * y


class RandomRotationWithProbability:
    def __init__(self, degrees, rotation_probability=0.5, return_locations=False):
        self.degrees = degrees
        self.rotation_probability = rotation_probability
        self.interpolation = TF.InterpolationMode.BILINEAR
        self.return_locations = return_locations

    def __call__(self, x):
        output_height, output_width = get_height_width(x)

        if random.random() < self.rotation_probability:
            angle = random.uniform(-self.degrees, self.degrees)

            # rotation and crop
            image, i_crop, j_crop, h_crop, w_crop = self._rotate_and_crop(x, angle, output_height, output_width)

            # compute (i, j) in the original image by reversing the rotation
            i, j = self._reverse_rotation(i_crop, j_crop, angle)

            if self.return_locations:
                return image, (i, j, h_crop, w_crop, output_height, output_width)
            return image
        if self.return_locations:
            return x, (0, 0, output_height, output_width, output_height, output_width)
        return x

    def _rotate_and_crop(self, x, rotation_degree, output_height, output_width):
        image = TF.rotate(x, rotation_degree, interpolation=self.interpolation)

        # get the largest rectangle that fits in the rotated image
        # center crop to ommit black noise on the edges
        width_to_crop, height_to_crop = _largest_rotated_rect(output_width, output_height, math.radians(rotation_degree))
        width_to_crop, height_to_crop = abs(round(width_to_crop, 0)), abs(round(height_to_crop, 0))

        image = TF.center_crop(image, [int(height_to_crop), int(width_to_crop)])
        image = TF.resize(image, (output_height, output_width))

        # get the top-left corner of the crop in the rotated image
        i_crop = (output_height - height_to_crop) // 2
        j_crop = (output_width - width_to_crop) // 2

        return image, i_crop, j_crop, int(height_to_crop), int(width_to_crop)

    def _reverse_rotation(self, i_crop, j_crop, angle):
        """
        Reverse the rotation to find the original coordinates (i, j).
        """
        theta = math.radians(-angle)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        # apply the reverse rotation matrix to (i_crop, j_crop)
        i = (i_crop * cos_theta) - (j_crop * sin_theta)
        j = (i_crop * sin_theta) + (j_crop * cos_theta)

        return round(i), round(j)


### RANDOM TRANSLATION & CROP ###


class TranslateAndCrop:
    def __init__(self, translate=(0, 0), probability=0.5, probability_translation_x=0.75, probability_translation_y=0.75,
                 return_locations=False):
        self.translate = translate
        self.probability = probability
        self.probability_translation_x = probability_translation_x
        self.probability_translation_y = probability_translation_y
        self.return_locations = return_locations

    def __call__(self, img):
        output_height, output_width = get_height_width(img)

        if random.random() < self.probability_translation_x:
            translation_x = random.uniform(-self.translate[0], self.translate[0])
        else:
            translation_x = 0

        if random.random() < self.probability_translation_y:
            translation_y = random.uniform(-self.translate[1], self.translate[1])
        else:
            translation_y = 0

        translation = (translation_x, translation_y)
        # Translate the image
        img = TF.affine(img, angle=0, translate=translation, scale=1, shear=0, fill=0)

        # Find coordinates where the image is non-zero
        non_zero_coords = np.where(np.array(img).squeeze() != 0)
        non_zero_coords = np.column_stack(non_zero_coords)

        if non_zero_coords.shape[0] == 0:
            # If all pixels are zero, return the original image
            return img

        x_min = non_zero_coords[:, 0].min()
        x_max = non_zero_coords[:, 0].max()
        y_min = non_zero_coords[:, 1].min()
        y_max = non_zero_coords[:, 1].max()

        height = output_height if x_min == x_max else x_max - x_min
        width = output_width if y_min == y_max else y_max - y_min

        # Crop the image to the bounding box of non-zero regions
        img = TF.crop(img=img, top=x_min, left=y_min, height=height, width=width)
        img = TF.resize(img, (output_height, output_width))

        if self.return_locations:
            return img, (x_min, y_min, height, width, output_height, output_width)
        return img
