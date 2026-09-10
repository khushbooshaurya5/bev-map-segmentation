"""Frustum creation, camera->ego geometry, and BEV voxel pooling.

Lift-Splat-Shoot (Philion & Fidler, ECCV 2020) turns a set of surround-view
camera images into a single bird's-eye-view feature map:

- **Lift**: each pixel predicts a categorical distribution over discrete depth
  bins; the pixel's context feature is spread along its viewing ray weighted by
  that distribution, creating a frustum-shaped point cloud of features.
- **Splat**: every frustum point is placed into a BEV grid cell (using the
  camera calibration) and features in the same cell are summed (voxel pooling).

This module holds the pure geometry; the learnable parts live in ``model.py``.
"""

from __future__ import annotations

import torch


def make_frustum(image_h: int, image_w: int, feat_h: int, feat_w: int,
                 d_bounds) -> torch.Tensor:
    """Frustum of (u, v, depth) points in *input-image* pixel coordinates.

    Returns (D, feat_h, feat_w, 3).
    """
    dmin, dmax, dstep = d_bounds
    ds = torch.arange(dmin, dmax, dstep).float()
    D = ds.shape[0]
    ds = ds.view(D, 1, 1).expand(D, feat_h, feat_w)
    xs = torch.linspace(0, image_w - 1, feat_w).view(1, 1, feat_w).expand(D, feat_h, feat_w)
    ys = torch.linspace(0, image_h - 1, feat_h).view(1, feat_h, 1).expand(D, feat_h, feat_w)
    return torch.stack([xs, ys, ds], -1)          # (D, H, W, 3)


def get_geometry(frustum: torch.Tensor, rots: torch.Tensor, trans: torch.Tensor,
                 intrins: torch.Tensor) -> torch.Tensor:
    """Map frustum image points to ego-frame 3D points.

    frustum: (D,H,W,3)  rots:(B,N,3,3)  trans:(B,N,3)  intrins:(B,N,3,3)
    returns geom (B,N,D,H,W,3) in ego coordinates.
    """
    B, N = rots.shape[:2]
    D, H, W, _ = frustum.shape
    points = frustum.view(1, 1, D, H, W, 3).expand(B, N, D, H, W, 3).clone()
    # (u,v,d) -> (u*d, v*d, d)
    points = torch.cat([points[..., :2] * points[..., 2:3], points[..., 2:3]], -1)
    combine = rots @ torch.inverse(intrins)                      # (B,N,3,3)
    points = combine.view(B, N, 1, 1, 1, 3, 3) @ points.unsqueeze(-1)
    points = points.squeeze(-1) + trans.view(B, N, 1, 1, 1, 3)
    return points


def voxel_pooling(geom: torch.Tensor, feats: torch.Tensor, grid_conf) -> torch.Tensor:
    """Sum-pool frustum features into a BEV grid via scatter_add.

    geom:  (B,N,D,H,W,3) ego points
    feats: (B,N,C,D,H,W) context features
    grid_conf: dict with 'xbound','ybound','zbound' = [min,max,step]
    returns BEV features (B, C, Y, X).
    """
    device = feats.device
    xb, yb, zb = grid_conf["xbound"], grid_conf["ybound"], grid_conf["zbound"]
    bx = torch.tensor([xb[0], yb[0], zb[0]], device=device)
    dx = torch.tensor([xb[2], yb[2], zb[2]], device=device)
    nx = torch.tensor([int(round((xb[1] - xb[0]) / xb[2])),
                       int(round((yb[1] - yb[0]) / yb[2])),
                       int(round((zb[1] - zb[0]) / zb[2]))], device=device)

    B, N, C, D, H, W = feats.shape
    Nprime = N * D * H * W
    # (B, Nprime, C)
    x = feats.permute(0, 1, 3, 4, 5, 2).reshape(B, Nprime, C)
    # voxel index per point
    idx = ((geom - (bx - dx / 2.0)) / dx).long().reshape(B, Nprime, 3)

    Xn, Yn, Zn = int(nx[0]), int(nx[1]), int(nx[2])
    out = feats.new_zeros(B, C, Yn, Xn)
    for b in range(B):
        ix, iy, iz = idx[b, :, 0], idx[b, :, 1], idx[b, :, 2]
        valid = (ix >= 0) & (ix < Xn) & (iy >= 0) & (iy < Yn) & (iz >= 0) & (iz < Zn)
        ix, iy = ix[valid], iy[valid]
        feat = x[b][valid]                                       # (M, C)
        flat = (iy * Xn + ix)                                   # (M,)
        canvas = feats.new_zeros(Yn * Xn, C)
        canvas.index_add_(0, flat, feat)
        out[b] = canvas.view(Yn, Xn, C).permute(2, 0, 1)
    return out
