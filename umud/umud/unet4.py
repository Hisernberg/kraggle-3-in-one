"""Public UNet4 segmentation models (second, independent model family; declared external pretrained models).

Weights: Kaggle dataset `akshaysharma2136/umud-5fold-unet4-sota-models` (CC0-1.0), 5 folds each for the aponeurosis
and fascicle tasks (`{apo,fasc}_fold{k}_best.pt`), trained on the competition masks by the author of the public
notebook "AnatomyNet: Biomechanical U-Net". Architecture and preprocessing re-implemented from that notebook:
plain 4-level U-Net (base 64, no BatchNorm), RGB/255 input squashed to 384x384, sigmoid output, hflip TTA.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

TILE = 384


class ConvBlock(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.c = nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(inplace=True),
                               nn.Conv2d(o, o, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.c(x)


class UNet4(nn.Module):
    def __init__(self, in_channels=3, base=64):
        super().__init__()
        b = base
        self.e1, self.e2, self.e3, self.e4 = ConvBlock(in_channels, b), ConvBlock(b, 2 * b), ConvBlock(2 * b, 4 * b), \
            ConvBlock(4 * b, 8 * b)
        self.bn = ConvBlock(8 * b, 16 * b)
        self.u4, self.d4 = nn.ConvTranspose2d(16 * b, 8 * b, 2, stride=2), ConvBlock(16 * b, 8 * b)
        self.u3, self.d3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, stride=2), ConvBlock(8 * b, 4 * b)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, stride=2), ConvBlock(4 * b, 2 * b)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * b, b, 2, stride=2), ConvBlock(2 * b, b)
        self.out = nn.Conv2d(b, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.bn(self.pool(e4))
        d4 = self.d4(torch.cat([self.u4(b), e4], 1))
        d3 = self.d3(torch.cat([self.u3(d4), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


def load(path: str | Path, dev: str = "cpu") -> UNet4:
    m = UNet4()
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
    return m.to(dev).eval()


@torch.no_grad()
def predict(models: list, gray: np.ndarray, dev: str = "cpu") -> np.ndarray:
    """Mean probability of the given fold models (hflip TTA) on a grayscale B-mode crop, at crop resolution."""
    h, w = gray.shape[:2]
    rgb = np.repeat(cv2.resize(gray, (TILE, TILE))[..., None], 3, axis=2).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(dev)
    t = torch.cat([t, t.flip(-1)])
    acc = None
    for m in models:
        o = torch.sigmoid(m(t))
        p = (o[0] + o[1].flip(-1))[0] / 2
        acc = p if acc is None else acc + p
    return cv2.resize((acc / len(models)).cpu().numpy(), (w, h), interpolation=cv2.INTER_LINEAR)
