# M-LSD: Real-Time and Light-Weight Line Segment Detector
# Copyright 2021-present NAVER Corp. (Apache License v2.0)
# PyTorch version by lihaoweicv

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .util import HWC3, resize_image

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------


class _BlockTypeA(nn.Module):
    def __init__(self, in_c1, in_c2, out_c1, out_c2, upscale=True):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c2, out_c2, kernel_size=1),
            nn.BatchNorm2d(out_c2),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c1, out_c1, kernel_size=1),
            nn.BatchNorm2d(out_c1),
            nn.ReLU(inplace=True),
        )
        self.upscale = upscale

    def forward(self, a, b):
        b = self.conv1(b)
        a = self.conv2(a)
        if self.upscale:
            b = F.interpolate(b, scale_factor=2.0, mode='bilinear', align_corners=True)
        return torch.cat((a, b), dim=1)


class _BlockTypeB(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x) + x
        x = self.conv2(x)
        return x


class _BlockTypeC(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=5, dilation=5),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
        )
        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class _ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        self.channel_pad = out_planes - in_planes
        self.stride = stride
        if stride == 2:
            padding = 0
        else:
            padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )
        self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        if self.stride == 2:
            x = F.pad(x, (0, 1, 0, 1), "constant", 0)
        for module in self:
            if not isinstance(module, nn.MaxPool2d):
                x = module(x)
        return x


class _InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(_ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            _ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class _MobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        input_channel = 32
        width_mult = 1.0
        round_nearest = 8
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
        ]
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        features = [_ConvBNReLU(4, input_channel, stride=2)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(_InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        self.features = nn.Sequential(*features)
        self.fpn_selected = [1, 3, 6, 10, 13]

    def forward(self, x):
        fpn_features = []
        for i, f in enumerate(self.features):
            if i > self.fpn_selected[-1]:
                break
            x = f(x)
            if i in self.fpn_selected:
                fpn_features.append(x)
        c1, c2, c3, c4, c5 = fpn_features
        return c1, c2, c3, c4, c5


class _MobileV2_MLSD_Large(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _MobileNetV2()
        self.block15 = _BlockTypeA(in_c1=64, in_c2=96, out_c1=64, out_c2=64, upscale=False)
        self.block16 = _BlockTypeB(128, 64)
        self.block17 = _BlockTypeA(in_c1=32, in_c2=64, out_c1=64, out_c2=64)
        self.block18 = _BlockTypeB(128, 64)
        self.block19 = _BlockTypeA(in_c1=24, in_c2=64, out_c1=64, out_c2=64)
        self.block20 = _BlockTypeB(128, 64)
        self.block21 = _BlockTypeA(in_c1=16, in_c2=64, out_c1=64, out_c2=64)
        self.block22 = _BlockTypeB(128, 64)
        self.block23 = _BlockTypeC(64, 16)

    def forward(self, x):
        c1, c2, c3, c4, c5 = self.backbone(x)
        x = self.block15(c4, c5)
        x = self.block16(x)
        x = self.block17(c3, x)
        x = self.block18(x)
        x = self.block19(c2, x)
        x = self.block20(x)
        x = self.block21(c1, x)
        x = self.block22(x)
        x = self.block23(x)
        x = x[:, 7:, :, :]
        return x


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _decode_output(tpMap, topk_n=200, ksize=5):
    b, c, h, w = tpMap.shape
    assert b == 1
    displacement = tpMap[:, 1:5, :, :][0]
    center = tpMap[:, 0, :, :]
    heat = torch.sigmoid(center)
    hmax = F.max_pool2d(heat, (ksize, ksize), stride=1, padding=(ksize - 1) // 2)
    keep = (hmax == heat).float()
    heat = heat * keep
    heat = heat.reshape(-1)
    scores, indices = torch.topk(heat, topk_n, dim=-1, largest=True)
    yy = torch.floor_divide(indices, w).unsqueeze(-1)
    xx = torch.fmod(indices, w).unsqueeze(-1)
    ptss = torch.cat((yy, xx), dim=-1)
    ptss = ptss.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    displacement = displacement.detach().cpu().numpy()
    displacement = displacement.transpose((1, 2, 0))
    return ptss, scores, displacement


def _pred_lines(image, model, input_shape, score_thr=0.10, dist_thr=20.0):
    h, w, _ = image.shape
    device = next(iter(model.parameters())).device
    h_ratio, w_ratio = h / input_shape[0], w / input_shape[1]
    resized_image = np.concatenate([
        cv2.resize(image, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_AREA),
        np.ones([input_shape[0], input_shape[1], 1]),
    ], axis=-1)
    resized_image = resized_image.transpose((2, 0, 1))
    batch_image = np.expand_dims(resized_image, axis=0).astype('float32')
    batch_image = (batch_image / 127.5) - 1.0
    batch_image = torch.from_numpy(batch_image).float().to(device)
    outputs = model(batch_image)
    pts, pts_score, vmap = _decode_output(outputs, 200, 3)
    start = vmap[:, :, :2]
    end = vmap[:, :, 2:]
    dist_map = np.sqrt(np.sum((start - end) ** 2, axis=-1))
    segments_list = []
    for center, score in zip(pts, pts_score):
        y, x = center
        distance = dist_map[y, x]
        if score > score_thr and distance > dist_thr:
            disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[y, x, :]
            x_start = x + disp_x_start
            y_start = y + disp_y_start
            x_end = x + disp_x_end
            y_end = y + disp_y_end
            segments_list.append([x_start, y_start, x_end, y_end])
    lines = 2 * np.array(segments_list)
    lines[:, 0] = lines[:, 0] * w_ratio
    lines[:, 1] = lines[:, 1] * h_ratio
    lines[:, 2] = lines[:, 2] * w_ratio
    lines[:, 3] = lines[:, 3] * h_ratio
    return lines


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class MLSDdetector:
    def __init__(self, model):
        self.model = model

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path=None):
        from huggingface_hub import hf_hub_download
        repo = pretrained_model_or_path or "lllyasviel/Annotators"
        model_path = hf_hub_download(repo, "mlsd_large_512_fp32.pth")
        model = _MobileV2_MLSD_Large()
        model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
        model.eval()
        return cls(model)

    def to(self, device):
        self.model.to(device)
        return self

    def __call__(self, input_image, thr_v=0.1, thr_d=0.1, detect_resolution=512,
                 image_resolution=512, **kwargs):
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        img = input_image
        img_output = np.zeros_like(img)
        try:
            with torch.no_grad():
                lines = _pred_lines(img, self.model, [img.shape[0], img.shape[1]], thr_v, thr_d)
                for line in lines:
                    x_start, y_start, x_end, y_end = [int(val) for val in line]
                    cv2.line(img_output, (x_start, y_start), (x_end, y_end), [255, 255, 255], 1)
        except Exception:
            pass

        detected_map = img_output[:, :, 0]
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        return Image.fromarray(detected_map)
