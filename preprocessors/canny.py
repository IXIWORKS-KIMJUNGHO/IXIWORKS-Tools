import cv2
import numpy as np
from PIL import Image

from .util import HWC3, resize_image


class CannyDetector:
    def __call__(self, input_image, low_threshold=100, high_threshold=200,
                 detect_resolution=512, image_resolution=512, **kwargs):
        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image)
        img = HWC3(input_image)
        img = resize_image(img, detect_resolution)
        result = cv2.Canny(img, low_threshold, high_threshold)
        result = HWC3(result)
        result = resize_image(result, image_resolution)
        return Image.fromarray(result)
