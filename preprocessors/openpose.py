# OpenPose Body + Hand Detection
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# Adapted from controlnet_aux (Hzzone/pytorch-openpose + ControlNet edits)
# Licensed by CMU for non-commercial use only.

import colorsys
import math
from collections import OrderedDict
from typing import List, NamedTuple, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.ndimage import gaussian_filter, label

from .util import HWC3, resize_image

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class Keypoint(NamedTuple):
    x: float
    y: float
    score: float = 1.0
    id: int = -1


class BodyResult(NamedTuple):
    keypoints: List[Union[Keypoint, None]]
    total_score: float
    total_parts: int


HandResult = List[Keypoint]


class PoseResult(NamedTuple):
    body: BodyResult
    left_hand: Union[HandResult, None]
    right_hand: Union[HandResult, None]


# ---------------------------------------------------------------------------
# Network architectures
# ---------------------------------------------------------------------------

def _make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3], padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
    return nn.Sequential(OrderedDict(layers))


class _BodyPoseModel(nn.Module):
    def __init__(self):
        super().__init__()
        no_relu_layers = [
            'conv5_5_CPM_L1', 'conv5_5_CPM_L2',
            'Mconv7_stage2_L1', 'Mconv7_stage2_L2',
            'Mconv7_stage3_L1', 'Mconv7_stage3_L2',
            'Mconv7_stage4_L1', 'Mconv7_stage4_L2',
            'Mconv7_stage5_L1', 'Mconv7_stage5_L2',
            'Mconv7_stage6_L1', 'Mconv7_stage6_L1',
        ]
        blocks = {}
        block0 = OrderedDict([
            ('conv1_1', [3, 64, 3, 1, 1]), ('conv1_2', [64, 64, 3, 1, 1]),
            ('pool1_stage1', [2, 2, 0]),
            ('conv2_1', [64, 128, 3, 1, 1]), ('conv2_2', [128, 128, 3, 1, 1]),
            ('pool2_stage1', [2, 2, 0]),
            ('conv3_1', [128, 256, 3, 1, 1]), ('conv3_2', [256, 256, 3, 1, 1]),
            ('conv3_3', [256, 256, 3, 1, 1]), ('conv3_4', [256, 256, 3, 1, 1]),
            ('pool3_stage1', [2, 2, 0]),
            ('conv4_1', [256, 512, 3, 1, 1]), ('conv4_2', [512, 512, 3, 1, 1]),
            ('conv4_3_CPM', [512, 256, 3, 1, 1]),
            ('conv4_4_CPM', [256, 128, 3, 1, 1]),
        ])
        block1_1 = OrderedDict([
            ('conv5_1_CPM_L1', [128, 128, 3, 1, 1]),
            ('conv5_2_CPM_L1', [128, 128, 3, 1, 1]),
            ('conv5_3_CPM_L1', [128, 128, 3, 1, 1]),
            ('conv5_4_CPM_L1', [128, 512, 1, 1, 0]),
            ('conv5_5_CPM_L1', [512, 38, 1, 1, 0]),
        ])
        block1_2 = OrderedDict([
            ('conv5_1_CPM_L2', [128, 128, 3, 1, 1]),
            ('conv5_2_CPM_L2', [128, 128, 3, 1, 1]),
            ('conv5_3_CPM_L2', [128, 128, 3, 1, 1]),
            ('conv5_4_CPM_L2', [128, 512, 1, 1, 0]),
            ('conv5_5_CPM_L2', [512, 19, 1, 1, 0]),
        ])
        blocks['block1_1'] = block1_1
        blocks['block1_2'] = block1_2

        self.model0 = _make_layers(block0, no_relu_layers)

        for i in range(2, 7):
            blocks['block%d_1' % i] = OrderedDict([
                ('Mconv1_stage%d_L1' % i, [185, 128, 7, 1, 3]),
                ('Mconv2_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                ('Mconv3_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                ('Mconv4_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                ('Mconv5_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                ('Mconv6_stage%d_L1' % i, [128, 128, 1, 1, 0]),
                ('Mconv7_stage%d_L1' % i, [128, 38, 1, 1, 0]),
            ])
            blocks['block%d_2' % i] = OrderedDict([
                ('Mconv1_stage%d_L2' % i, [185, 128, 7, 1, 3]),
                ('Mconv2_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                ('Mconv3_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                ('Mconv4_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                ('Mconv5_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                ('Mconv6_stage%d_L2' % i, [128, 128, 1, 1, 0]),
                ('Mconv7_stage%d_L2' % i, [128, 19, 1, 1, 0]),
            ])

        for k in blocks:
            blocks[k] = _make_layers(blocks[k], no_relu_layers)

        self.model1_1 = blocks['block1_1']
        self.model2_1 = blocks['block2_1']
        self.model3_1 = blocks['block3_1']
        self.model4_1 = blocks['block4_1']
        self.model5_1 = blocks['block5_1']
        self.model6_1 = blocks['block6_1']

        self.model1_2 = blocks['block1_2']
        self.model2_2 = blocks['block2_2']
        self.model3_2 = blocks['block3_2']
        self.model4_2 = blocks['block4_2']
        self.model5_2 = blocks['block5_2']
        self.model6_2 = blocks['block6_2']

    def forward(self, x):
        out1 = self.model0(x)
        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2 = torch.cat([out1_1, out1_2, out1], 1)
        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = torch.cat([out2_1, out2_2, out1], 1)
        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = torch.cat([out3_1, out3_2, out1], 1)
        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5 = torch.cat([out4_1, out4_2, out1], 1)
        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6 = torch.cat([out5_1, out5_2, out1], 1)
        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)
        return out6_1, out6_2


class _HandPoseModel(nn.Module):
    def __init__(self):
        super().__init__()
        no_relu_layers = [
            'conv6_2_CPM', 'Mconv7_stage2', 'Mconv7_stage3',
            'Mconv7_stage4', 'Mconv7_stage5', 'Mconv7_stage6',
        ]
        block1_0 = OrderedDict([
            ('conv1_1', [3, 64, 3, 1, 1]), ('conv1_2', [64, 64, 3, 1, 1]),
            ('pool1_stage1', [2, 2, 0]),
            ('conv2_1', [64, 128, 3, 1, 1]), ('conv2_2', [128, 128, 3, 1, 1]),
            ('pool2_stage1', [2, 2, 0]),
            ('conv3_1', [128, 256, 3, 1, 1]), ('conv3_2', [256, 256, 3, 1, 1]),
            ('conv3_3', [256, 256, 3, 1, 1]), ('conv3_4', [256, 256, 3, 1, 1]),
            ('pool3_stage1', [2, 2, 0]),
            ('conv4_1', [256, 512, 3, 1, 1]), ('conv4_2', [512, 512, 3, 1, 1]),
            ('conv4_3', [512, 512, 3, 1, 1]), ('conv4_4', [512, 512, 3, 1, 1]),
            ('conv5_1', [512, 512, 3, 1, 1]), ('conv5_2', [512, 512, 3, 1, 1]),
            ('conv5_3_CPM', [512, 128, 3, 1, 1]),
        ])
        block1_1 = OrderedDict([
            ('conv6_1_CPM', [128, 512, 1, 1, 0]),
            ('conv6_2_CPM', [512, 22, 1, 1, 0]),
        ])
        blocks = {'block1_0': block1_0, 'block1_1': block1_1}
        for i in range(2, 7):
            blocks['block%d' % i] = OrderedDict([
                ('Mconv1_stage%d' % i, [150, 128, 7, 1, 3]),
                ('Mconv2_stage%d' % i, [128, 128, 7, 1, 3]),
                ('Mconv3_stage%d' % i, [128, 128, 7, 1, 3]),
                ('Mconv4_stage%d' % i, [128, 128, 7, 1, 3]),
                ('Mconv5_stage%d' % i, [128, 128, 7, 1, 3]),
                ('Mconv6_stage%d' % i, [128, 128, 1, 1, 0]),
                ('Mconv7_stage%d' % i, [128, 22, 1, 1, 0]),
            ])
        for k in blocks:
            blocks[k] = _make_layers(blocks[k], no_relu_layers)

        self.model1_0 = blocks['block1_0']
        self.model1_1 = blocks['block1_1']
        self.model2 = blocks['block2']
        self.model3 = blocks['block3']
        self.model4 = blocks['block4']
        self.model5 = blocks['block5']
        self.model6 = blocks['block6']

    def forward(self, x):
        out1_0 = self.model1_0(x)
        out1_1 = self.model1_1(out1_0)
        concat_stage2 = torch.cat([out1_1, out1_0], 1)
        out_stage2 = self.model2(concat_stage2)
        concat_stage3 = torch.cat([out_stage2, out1_0], 1)
        out_stage3 = self.model3(concat_stage3)
        concat_stage4 = torch.cat([out_stage3, out1_0], 1)
        out_stage4 = self.model4(concat_stage4)
        concat_stage5 = torch.cat([out_stage4, out1_0], 1)
        out_stage5 = self.model5(concat_stage5)
        concat_stage6 = torch.cat([out_stage5, out1_0], 1)
        out_stage6 = self.model6(concat_stage6)
        return out_stage6


# ---------------------------------------------------------------------------
# Resize / padding helpers
# ---------------------------------------------------------------------------

def _smart_resize(x, s):
    Ht, Wt = s
    if x.ndim == 2:
        Ho, Wo = x.shape
        Co = 1
    else:
        Ho, Wo, Co = x.shape
    if Co == 3 or Co == 1:
        k = float(Ht + Wt) / float(Ho + Wo)
        interp = cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4
        return cv2.resize(x, (int(Wt), int(Ht)), interpolation=interp)
    else:
        return np.stack([_smart_resize(x[:, :, i], s) for i in range(Co)], axis=2)


def _smart_resize_k(x, fx, fy):
    if x.ndim == 2:
        Ho, Wo = x.shape
        Co = 1
    else:
        Ho, Wo, Co = x.shape
    Ht, Wt = Ho * fy, Wo * fx
    if Co == 3 or Co == 1:
        k = float(Ht + Wt) / float(Ho + Wo)
        interp = cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4
        return cv2.resize(x, (int(Wt), int(Ht)), interpolation=interp)
    else:
        return np.stack([_smart_resize_k(x[:, :, i], fx, fy) for i in range(Co)], axis=2)


def _pad_right_down_corner(img, stride, pad_value):
    h, w = img.shape[:2]
    pad = [0, 0,
           0 if (h % stride == 0) else stride - (h % stride),
           0 if (w % stride == 0) else stride - (w % stride)]
    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :] * 0 + pad_value, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :] * 0 + pad_value, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + pad_value, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + pad_value, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)
    return img_padded, pad


def _transfer(model, model_weights):
    transfered = {}
    for name in model.state_dict().keys():
        transfered[name] = model_weights['.'.join(name.split('.')[1:])]
    return transfered


def _npmax(array):
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j


# ---------------------------------------------------------------------------
# Body estimation
# ---------------------------------------------------------------------------

class _Body:
    def __init__(self, model_path):
        self.model = _BodyPoseModel()
        model_dict = _transfer(self.model, torch.load(model_path, map_location="cpu"))
        self.model.load_state_dict(model_dict)
        self.model.eval()

    def to(self, device):
        self.model.to(device)
        return self

    def __call__(self, oriImg):
        device = next(iter(self.model.parameters())).device
        scale_search = [0.5]
        boxsize = 368
        stride = 8
        padValue = 128
        thre1 = 0.1
        thre2 = 0.05
        multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]
        heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
        paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

        for m in range(len(multiplier)):
            scale = multiplier[m]
            imageToTest = _smart_resize_k(oriImg, fx=scale, fy=scale)
            imageToTest_padded, pad = _pad_right_down_corner(imageToTest, stride, padValue)
            im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
            im = np.ascontiguousarray(im)
            data = torch.from_numpy(im).float().to(device)
            with torch.no_grad():
                Mconv7_stage6_L1, Mconv7_stage6_L2 = self.model(data)
            Mconv7_stage6_L1 = Mconv7_stage6_L1.cpu().numpy()
            Mconv7_stage6_L2 = Mconv7_stage6_L2.cpu().numpy()

            heatmap = np.transpose(np.squeeze(Mconv7_stage6_L2), (1, 2, 0))
            heatmap = _smart_resize_k(heatmap, fx=stride, fy=stride)
            heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            heatmap = _smart_resize(heatmap, (oriImg.shape[0], oriImg.shape[1]))

            paf = np.transpose(np.squeeze(Mconv7_stage6_L1), (1, 2, 0))
            paf = _smart_resize_k(paf, fx=stride, fy=stride)
            paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            paf = _smart_resize(paf, (oriImg.shape[0], oriImg.shape[1]))

            heatmap_avg += heatmap_avg + heatmap / len(multiplier)
            paf_avg += paf / len(multiplier)

        all_peaks = []
        peak_counter = 0
        for part in range(18):
            map_ori = heatmap_avg[:, :, part]
            one_heatmap = gaussian_filter(map_ori, sigma=3)
            map_left = np.zeros(one_heatmap.shape)
            map_left[1:, :] = one_heatmap[:-1, :]
            map_right = np.zeros(one_heatmap.shape)
            map_right[:-1, :] = one_heatmap[1:, :]
            map_up = np.zeros(one_heatmap.shape)
            map_up[:, 1:] = one_heatmap[:, :-1]
            map_down = np.zeros(one_heatmap.shape)
            map_down[:, :-1] = one_heatmap[:, 1:]
            peaks_binary = np.logical_and.reduce((
                one_heatmap >= map_left, one_heatmap >= map_right,
                one_heatmap >= map_up, one_heatmap >= map_down,
                one_heatmap > thre1))
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            peak_id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],) for i in range(len(peak_id))]
            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)

        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
                   [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
                   [1, 16], [16, 18], [3, 17], [6, 18]]
        mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22],
                  [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52],
                  [55, 56], [37, 38], [45, 46]]

        connection_all = []
        special_k = []
        mid_num = 10

        for k in range(len(mapIdx)):
            score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
            candA = all_peaks[limbSeq[k][0] - 1]
            candB = all_peaks[limbSeq[k][1] - 1]
            nA = len(candA)
            nB = len(candB)
            if nA != 0 and nB != 0:
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                        norm = max(0.001, norm)
                        vec = np.divide(vec, norm)
                        startend = list(zip(
                            np.linspace(candA[i][0], candB[j][0], num=mid_num),
                            np.linspace(candA[i][1], candB[j][1], num=mid_num)))
                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0]
                                          for I in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1]
                                          for I in range(len(startend))])
                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                            0.5 * oriImg.shape[0] / norm - 1, 0)
                        criterion1 = len(np.nonzero(score_midpts > thre2)[0]) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append(
                                [i, j, score_with_dist_prior,
                                 score_with_dist_prior + candA[i][2] + candB[j][2]])
                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if i not in connection[:, 3] and j not in connection[:, 4]:
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                        if len(connection) >= min(nA, nB):
                            break
                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])

        for k in range(len(mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = np.array(limbSeq[k]) - 1
                for i in range(len(connection_all[k])):
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1
                    if found == 1:
                        j = subset_idx[0]
                        if subset[j][indexB] != partBs[i]:
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2:
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])

        deleteIdx = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)
        return candidate, subset

    @staticmethod
    def format_body_result(candidate, subset):
        return [
            BodyResult(
                keypoints=[
                    Keypoint(x=candidate[ci][0], y=candidate[ci][1],
                             score=candidate[ci][2], id=candidate[ci][3])
                    if ci != -1 else None
                    for ci in person[:18].astype(int)
                ],
                total_score=person[18],
                total_parts=person[19],
            )
            for person in subset
        ]


# ---------------------------------------------------------------------------
# Hand estimation
# ---------------------------------------------------------------------------

class _Hand:
    def __init__(self, model_path):
        self.model = _HandPoseModel()
        model_dict = _transfer(self.model, torch.load(model_path, map_location="cpu"))
        self.model.load_state_dict(model_dict)
        self.model.eval()

    def to(self, device):
        self.model.to(device)
        return self

    def __call__(self, oriImgRaw):
        device = next(iter(self.model.parameters())).device
        scale_search = [0.5, 1.0, 1.5, 2.0]
        boxsize = 368
        stride = 8
        padValue = 128
        thre = 0.05
        multiplier = [x * boxsize for x in scale_search]
        wsize = 128
        heatmap_avg = np.zeros((wsize, wsize, 22))
        Hr, Wr, Cr = oriImgRaw.shape
        oriImg = cv2.GaussianBlur(oriImgRaw, (0, 0), 0.8)

        for m in range(len(multiplier)):
            scale = multiplier[m]
            imageToTest = _smart_resize(oriImg, (scale, scale))
            imageToTest_padded, pad = _pad_right_down_corner(imageToTest, stride, padValue)
            im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
            im = np.ascontiguousarray(im)
            data = torch.from_numpy(im).float().to(device)
            with torch.no_grad():
                output = self.model(data).cpu().numpy()
            heatmap = np.transpose(np.squeeze(output), (1, 2, 0))
            heatmap = _smart_resize_k(heatmap, fx=stride, fy=stride)
            heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            heatmap = _smart_resize(heatmap, (wsize, wsize))
            heatmap_avg += heatmap / len(multiplier)

        all_peaks = []
        for part in range(21):
            map_ori = heatmap_avg[:, :, part]
            one_heatmap = gaussian_filter(map_ori, sigma=3)
            binary = np.ascontiguousarray(one_heatmap > thre, dtype=np.uint8)
            if np.sum(binary) == 0:
                all_peaks.append([0, 0])
                continue
            label_img, label_numbers = label(binary)
            max_index = np.argmax([np.sum(map_ori[label_img == i]) for i in range(1, label_numbers + 1)]) + 1
            label_img[label_img != max_index] = 0
            map_ori[label_img == 0] = 0
            y, x = _npmax(map_ori)
            y = int(float(y) * float(Hr) / float(wsize))
            x = int(float(x) * float(Wr) / float(wsize))
            all_peaks.append([x, y])
        return np.array(all_peaks)


# ---------------------------------------------------------------------------
# Drawing functions
# ---------------------------------------------------------------------------

_BODY_LIMB_SEQ = [
    [2, 3], [2, 6], [3, 4], [4, 5],
    [6, 7], [7, 8], [2, 9], [9, 10],
    [10, 11], [2, 12], [12, 13], [13, 14],
    [2, 1], [1, 15], [15, 17], [1, 16],
    [16, 18],
]

_BODY_COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
    [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
    [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
    [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
    [255, 0, 170], [255, 0, 85],
]

_HAND_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15],
    [15, 16], [0, 17], [17, 18], [18, 19], [19, 20],
]

_EPS = 0.01


def _draw_bodypose(canvas, keypoints):
    H, W, C = canvas.shape
    stickwidth = 4
    for (k1_index, k2_index), color in zip(_BODY_LIMB_SEQ, _BODY_COLORS):
        kp1 = keypoints[k1_index - 1]
        kp2 = keypoints[k2_index - 1]
        if kp1 is None or kp2 is None:
            continue
        Y = np.array([kp1.x, kp2.x]) * float(W)
        X = np.array([kp1.y, kp2.y]) * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])
    for keypoint, color in zip(keypoints, _BODY_COLORS):
        if keypoint is None:
            continue
        x = int(keypoint.x * W)
        y = int(keypoint.y * H)
        cv2.circle(canvas, (x, y), 4, color, thickness=-1)
    return canvas


def _draw_handpose(canvas, keypoints):
    if not keypoints:
        return canvas
    H, W, C = canvas.shape
    n_edges = len(_HAND_EDGES)
    for ie, (e1, e2) in enumerate(_HAND_EDGES):
        k1 = keypoints[e1]
        k2 = keypoints[e2]
        if k1 is None or k2 is None:
            continue
        x1, y1 = int(k1.x * W), int(k1.y * H)
        x2, y2 = int(k2.x * W), int(k2.y * H)
        if x1 > _EPS and y1 > _EPS and x2 > _EPS and y2 > _EPS:
            r, g, b = colorsys.hsv_to_rgb(ie / float(n_edges), 1.0, 1.0)
            color = (int(r * 255), int(g * 255), int(b * 255))
            cv2.line(canvas, (x1, y1), (x2, y2), color, thickness=2)
    for keypoint in keypoints:
        x = int(keypoint.x * W)
        y = int(keypoint.y * H)
        if x > _EPS and y > _EPS:
            cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    return canvas


def _hand_detect(body, oriImg):
    ratioWristElbow = 0.33
    detect_result = []
    image_height, image_width = oriImg.shape[0:2]
    keypoints = body.keypoints
    left_shoulder = keypoints[5]
    left_elbow = keypoints[6]
    left_wrist = keypoints[7]
    right_shoulder = keypoints[2]
    right_elbow = keypoints[3]
    right_wrist = keypoints[4]

    has_left = all(kp is not None for kp in (left_shoulder, left_elbow, left_wrist))
    has_right = all(kp is not None for kp in (right_shoulder, right_elbow, right_wrist))
    if not (has_left or has_right):
        return []

    hands = []
    if has_left:
        hands.append([left_shoulder.x, left_shoulder.y, left_elbow.x, left_elbow.y,
                       left_wrist.x, left_wrist.y, True])
    if has_right:
        hands.append([right_shoulder.x, right_shoulder.y, right_elbow.x, right_elbow.y,
                       right_wrist.x, right_wrist.y, False])

    for x1, y1, x2, y2, x3, y3, is_left in hands:
        x = x3 + ratioWristElbow * (x3 - x2)
        y = y3 + ratioWristElbow * (y3 - y2)
        distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        width = 1.5 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)
        x -= width / 2
        y -= width / 2
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        width1 = width
        width2 = width
        if x + width > image_width:
            width1 = image_width - x
        if y + width > image_height:
            width2 = image_height - y
        width = min(width1, width2)
        if width >= 20:
            detect_result.append((int(x), int(y), int(width), is_left))
    return detect_result


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

def _draw_poses(poses, H, W, draw_body=True, draw_hand=True):
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    for pose in poses:
        if draw_body:
            canvas = _draw_bodypose(canvas, pose.body.keypoints)
        if draw_hand:
            canvas = _draw_handpose(canvas, pose.left_hand)
            canvas = _draw_handpose(canvas, pose.right_hand)
    return canvas


class OpenposeDetector:
    def __init__(self, body_estimation, hand_estimation=None):
        self.body_estimation = body_estimation
        self.hand_estimation = hand_estimation

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path=None):
        from huggingface_hub import hf_hub_download
        repo = pretrained_model_or_path or "lllyasviel/Annotators"
        body_model_path = hf_hub_download(repo, "body_pose_model.pth")
        hand_model_path = hf_hub_download(repo, "hand_pose_model.pth")
        body_estimation = _Body(body_model_path)
        hand_estimation = _Hand(hand_model_path)
        return cls(body_estimation, hand_estimation)

    def to(self, device):
        self.body_estimation.to(device)
        if self.hand_estimation:
            self.hand_estimation.to(device)
        return self

    def _detect_hands(self, body, oriImg):
        left_hand = None
        right_hand = None
        H, W, _ = oriImg.shape
        for x, y, w, is_left in _hand_detect(body, oriImg):
            peaks = self.hand_estimation(oriImg[y:y + w, x:x + w, :]).astype(np.float32)
            if peaks.ndim == 2 and peaks.shape[1] == 2:
                peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
                peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
                hand_result = [Keypoint(x=peak[0], y=peak[1]) for peak in peaks]
                if is_left:
                    left_hand = hand_result
                else:
                    right_hand = hand_result
        return left_hand, right_hand

    def detect_poses(self, oriImg, include_hand=False):
        oriImg = oriImg[:, :, ::-1].copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.body_estimation(oriImg)
            bodies = self.body_estimation.format_body_result(candidate, subset)
            results = []
            for body in bodies:
                left_hand, right_hand = (None, None)
                if include_hand and self.hand_estimation:
                    left_hand, right_hand = self._detect_hands(body, oriImg)
                results.append(PoseResult(
                    BodyResult(
                        keypoints=[
                            Keypoint(x=kp.x / float(W), y=kp.y / float(H))
                            if kp is not None else None
                            for kp in body.keypoints
                        ],
                        total_score=body.total_score,
                        total_parts=body.total_parts,
                    ),
                    left_hand, right_hand,
                ))
            return results

    def __call__(self, input_image, detect_resolution=512, image_resolution=512,
                 include_body=True, include_hand=False, hand_and_face=None, **kwargs):
        if hand_and_face is not None:
            include_hand = hand_and_face

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        H, W, C = input_image.shape

        poses = self.detect_poses(input_image, include_hand)
        canvas = _draw_poses(poses, H, W, draw_body=include_body, draw_hand=include_hand)

        detected_map = HWC3(canvas)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        return Image.fromarray(detected_map)
