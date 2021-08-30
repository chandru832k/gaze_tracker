import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from config_default import *
from torch.nn import functional as F
from torchvision import models

config = DefaultConfig()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

heatmap_xs = None
heatmap_ys = None
heatmap_alpha = None


def make_heatmap(centre, sigma):
    global heatmap_xs, heatmap_ys, heatmap_alpha
    w, h = config.gaze_heatmap_size
    if heatmap_xs is None:
        xs = np.arange(0, w, step=1, dtype=np.float32)
        ys = np.expand_dims(np.arange(0, h, step=1, dtype=np.float32), -1)
        heatmap_xs = torch.tensor(xs).to(device)
        heatmap_ys = torch.tensor(ys).to(device)
    heatmap_alpha = -0.5 / (sigma ** 2)
    cx = (w / config.actual_screen_size[0]) * centre[0]
    cy = (h / config.actual_screen_size[1]) * centre[1]
    heatmap = torch.exp(heatmap_alpha * ((heatmap_xs - cx)**2 + (heatmap_ys - cy)**2))
    heatmap = 1e-8 + heatmap  # Make the zeros non-zero (remove collapsing issue)
    return heatmap.unsqueeze(0)  # make it (1 x H x W) in shape


def batch_make_heatmaps(centres, sigma):
    return torch.stack([make_heatmap(centre, sigma) for centre in centres], axis=0)

def pitchyaw_to_vector(a):
    if a.shape[1] == 2:
        sin = torch.sin(a)
        cos = torch.cos(a)
        return torch.stack([cos[:, 0] * sin[:, 1], sin[:, 0], cos[:, 0] * cos[:, 1]], dim=1)
    elif a.shape[1] == 3:
        return F.normalize(a)
    else:
        raise ValueError('Do not know how to convert tensor of size %s' % a.shape)

def to_screen_coordinates(direction):
    direction = pitchyaw_to_vector(direction)

    # Negate gaze vector back (to camera perspective)
    direction = -direction

    # Convert back from mm to pixels
    ppm_w = config.pixel_per_mm[0]
    ppm_h = config.pixel_per_mm[1]
    PoG_px = torch.stack([
        torch.clamp(direction * ppm_w,
                    0.0, float(config.actual_screen_size[0])),
        torch.clamp(direction * ppm_h,
                    0.0, float(config.actual_screen_size[1]))
    ], axis=-1)

    return PoG_px

def scale_xy_to_ab(value_xy, x, y, a, b):
    value_ab = ((value_xy - x) / (y - x)) * (b - a) + a
    return value_ab

softargmax_xs = None
softargmax_ys = None

def soft_argmax(heatmaps):
    global softargmax_xs, softargmax_ys
    if softargmax_xs is None:
        # Assume normalized coordinate [0, 1] for numeric stability
        w, h = config.gaze_heatmap_size
        ref_xs, ref_ys = np.meshgrid(np.linspace(0, 1.0, num=w, endpoint=True),
                                     np.linspace(0, 1.0, num=h, endpoint=True),
                                     indexing='xy')
        ref_xs = np.reshape(ref_xs, [1, h*w])
        ref_ys = np.reshape(ref_ys, [1, h*w])
        softargmax_xs = torch.tensor(ref_xs.astype(np.float32)).to(device)
        softargmax_ys = torch.tensor(ref_ys.astype(np.float32)).to(device)
    ref_xs, ref_ys = softargmax_xs, softargmax_ys

    # Yield softmax+integrated coordinates in [0, 1]
    n, _, h, w = heatmaps.shape
    assert(w == config.gaze_heatmap_size[0])
    assert(h == config.gaze_heatmap_size[1])
    beta = 1e2
    x = heatmaps.view(-1, h*w)
    x = F.softmax(beta * x, dim=-1)
    lmrk_xs = torch.sum(ref_xs * x, axis=-1)
    lmrk_ys = torch.sum(ref_ys * x, axis=-1)

    # Return to actual coordinates ranges
    pixel_xs = torch.clamp(config.actual_screen_size[0] * lmrk_xs,
                           0.0, config.actual_screen_size[0])
    pixel_ys = torch.clamp(config.actual_screen_size[1] * lmrk_ys,
                           0.0, config.actual_screen_size[1])
    return torch.stack([pixel_xs, pixel_ys], axis=-1)
