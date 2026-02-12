import numpy as np
import torch
from PIL import Image

from .util import HWC3, resize_image


class MidasDetector:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path=None):
        from transformers import DPTForDepthEstimation, DPTImageProcessor
        model_id = "Intel/dpt-hybrid-midas"
        processor = DPTImageProcessor.from_pretrained(model_id)
        model = DPTForDepthEstimation.from_pretrained(model_id)
        model.eval()
        return cls(model, processor)

    @torch.no_grad()
    def __call__(self, input_image, detect_resolution=512,
                 image_resolution=512, **kwargs):
        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image)
        img = HWC3(input_image)
        img = resize_image(img, detect_resolution)
        pil_img = Image.fromarray(img)

        device = next(self.model.parameters()).device
        inputs = self.processor(images=pil_img, return_tensors="pt").to(device)
        outputs = self.model(**inputs)
        depth = outputs.predicted_depth

        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max - depth_min > 1e-6:
            depth = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth = np.zeros_like(depth)
        depth = (depth * 255.0).clip(0, 255).astype(np.uint8)

        result = HWC3(depth)
        result = resize_image(result, image_resolution)
        return Image.fromarray(result)
