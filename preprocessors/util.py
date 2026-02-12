import cv2
import numpy as np


def HWC3(x):
    """Ensure image is in HWC 3-channel format."""
    assert x.dtype == np.uint8
    if x.ndim == 2:
        return np.stack([x] * 3, axis=2)
    if x.shape[2] == 1:
        return np.concatenate([x] * 3, axis=2)
    if x.shape[2] == 4:
        color = x[:, :, :3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        result = (color * alpha + 255.0 * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
        return result
    return x


def resize_image(input_image, resolution):
    """Resize image maintaining aspect ratio, rounding to 64px boundaries."""
    H, W = input_image.shape[:2]
    k = float(resolution) / min(H, W)
    H_new = int(np.round(H * k / 64.0)) * 64
    W_new = int(np.round(W * k / 64.0)) * 64
    if max(H_new, W_new) > max(H, W):
        interp = cv2.INTER_LANCZOS4
    else:
        interp = cv2.INTER_AREA
    return cv2.resize(input_image, (W_new, H_new), interpolation=interp)
