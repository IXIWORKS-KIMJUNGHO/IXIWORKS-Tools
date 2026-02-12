import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .util import HWC3, resize_image


class ResidualBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, n_residual_blocks=9, sigmoid=True):
        super().__init__()
        # Initial block
        model0 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        # Downsampling
        model1 = []
        for i in range(2):
            mult = 2 ** i
            model1 += [
                nn.Conv2d(64 * mult, 64 * mult * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(64 * mult * 2),
                nn.ReLU(inplace=True),
            ]
        # Residual blocks
        model2 = []
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(256)]
        # Upsampling
        model3 = []
        for i in range(2):
            mult = 2 ** (2 - i)
            model3 += [
                nn.ConvTranspose2d(64 * mult, 64 * mult // 2, 3,
                                   stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(64 * mult // 2),
                nn.ReLU(inplace=True),
            ]
        # Final
        model4 = [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]

        self.model0 = nn.Sequential(*model0)
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)

    def forward(self, x):
        out = self.model4(self.model3(self.model2(self.model1(self.model0(x)))))
        return out


class LineartDetector:
    def __init__(self, model):
        self.model = model

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path=None):
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download("lllyasviel/Annotators", "sk_model.pth")
        model = Generator(3, 1, 3)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()
        return cls(model)

    @torch.no_grad()
    def __call__(self, input_image, coarse=False, detect_resolution=512,
                 image_resolution=512, **kwargs):
        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image)
        img = HWC3(input_image)
        img = resize_image(img, detect_resolution)

        device = next(self.model.parameters()).device
        tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        output = self.model(tensor)[0, 0].cpu().numpy()
        output = (output * 255.0).clip(0, 255).astype(np.uint8)
        output = 255 - output

        result = HWC3(output)
        result = resize_image(result, image_resolution)
        return Image.fromarray(result)
