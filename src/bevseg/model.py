"""Lift-Splat-Shoot camera-to-BEV segmentation model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .frustum import get_geometry, make_frustum, voxel_pooling


def conv_bn(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class CamEncoder(nn.Module):
    """Per-image encoder producing context features + depth distribution.

    Output splits into ``C`` context channels and ``D`` depth-bin logits; the
    outer product lifts context along the ray (softmax over depth).
    """

    def __init__(self, C: int, D: int, base: int = 32) -> None:
        super().__init__()
        self.C, self.D = C, D
        self.enc = nn.Sequential(conv_bn(3, base, 2), conv_bn(base, base * 2, 2),
                                 conv_bn(base * 2, base * 4, 2))
        self.head = nn.Conv2d(base * 4, C + D, 1)
        self.downsample = 8

    def forward(self, x):
        # x: (B*N, 3, H, W)
        feat = self.head(self.enc(x))               # (BN, C+D, h, w)
        depth = feat[:, :self.D].softmax(1)         # (BN, D, h, w)
        ctx = feat[:, self.D:self.D + self.C]       # (BN, C, h, w)
        # lift: (BN, C, D, h, w)
        lifted = depth.unsqueeze(1) * ctx.unsqueeze(2)
        return lifted


class BEVEncoder(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, base: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            conv_bn(in_ch, base), conv_bn(base, base),
            conv_bn(base, base), nn.Conv2d(base, num_classes, 1))

    def forward(self, x):
        return self.net(x)


class LiftSplatShoot(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        m, d = cfg["model"], cfg["data"]
        self.image_h, self.image_w = d["image_h"], d["image_w"]
        self.C = m["context_channels"]
        self.grid_conf = {"xbound": d["xbound"], "ybound": d["ybound"], "zbound": d["zbound"]}
        self.d_bounds = d["dbound"]

        self.feat_h = self.image_h // 8
        self.feat_w = self.image_w // 8
        frustum = make_frustum(self.image_h, self.image_w, self.feat_h, self.feat_w, self.d_bounds)
        self.register_buffer("frustum", frustum)
        self.D = frustum.shape[0]

        self.cam_encoder = CamEncoder(self.C, self.D, base=m.get("cam_base", 32))
        self.bev_encoder = BEVEncoder(self.C, m["num_classes"], base=m.get("bev_base", 64))

    def forward(self, images, rots, trans, intrins):
        B, N = images.shape[:2]
        x = images.view(B * N, *images.shape[2:])
        lifted = self.cam_encoder(x)                          # (BN, C, D, h, w)
        lifted = lifted.view(B, N, self.C, self.D, self.feat_h, self.feat_w)
        geom = get_geometry(self.frustum, rots, trans, intrins)   # (B,N,D,h,w,3)
        bev = voxel_pooling(geom, lifted, self.grid_conf)     # (B, C, Y, X)
        return self.bev_encoder(bev)
