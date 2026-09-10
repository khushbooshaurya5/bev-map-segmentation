"""nuScenes BEV dataset (loader stub) and a synthetic drop-in.

Full nuScenes loading needs the `nuscenes-devkit`; `NuScenesBEV` documents the
expected per-sample tensors and where they come from. `SyntheticBEV` produces
the same tensor signature so the model trains end-to-end with no download.

Per sample:
  images  (N, 3, H, W)   surround-view camera images, normalised
  rots    (N, 3, 3)      camera-to-ego rotation
  trans   (N, 3)         camera-to-ego translation
  intrins (N, 3, 3)      camera intrinsics (for the input-image resolution)
  bev     (Y, X)         BEV semantic label (0=background, >=1 classes)
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

NUM_CLASSES = 2  # background + vehicle (extend as needed)
CLASS_NAMES = ["background", "vehicle"]


class NuScenesBEV(Dataset):
    """Loader stub. Implement with nuscenes-devkit; see scripts/DATASET.md."""

    def __init__(self, root: str, cfg_data: dict, split: str = "train") -> None:
        raise NotImplementedError(
            "Wire this to nuscenes-devkit (NuScenes(...).sample). See "
            "scripts/DATASET.md for the fields to populate. Use SyntheticBEV "
            "for a smoke test in the meantime."
        )


class SyntheticBEV(Dataset):
    def __init__(self, cfg_data: dict, length: int = 8) -> None:
        self.length = length
        self.n_cams = cfg_data.get("n_cams", 4)
        self.H, self.W = cfg_data["image_h"], cfg_data["image_w"]
        xb, yb = cfg_data["xbound"], cfg_data["ybound"]
        self.Y = int(round((yb[1] - yb[0]) / yb[2]))
        self.X = int(round((xb[1] - xb[0]) / xb[2]))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int):
        rng = np.random.default_rng(i)
        images = rng.uniform(0, 1, size=(self.n_cams, 3, self.H, self.W)).astype(np.float32)

        rots = np.zeros((self.n_cams, 3, 3), np.float32)
        trans = np.zeros((self.n_cams, 3), np.float32)
        intrins = np.zeros((self.n_cams, 3, 3), np.float32)
        f = 0.9 * self.W
        for c in range(self.n_cams):
            yaw = 2 * np.pi * c / self.n_cams
            cz, sz = np.cos(yaw), np.sin(yaw)
            # camera looks outward along its yaw direction
            rots[c] = np.array([[cz, 0, sz], [0, 1, 0], [-sz, 0, cz]], np.float32)
            trans[c] = np.array([0, 0, 1.5], np.float32)
            intrins[c] = np.array([[f, 0, self.W / 2], [0, f, self.H / 2], [0, 0, 1]], np.float32)

        # BEV label: a few vehicle blobs
        bev = np.zeros((self.Y, self.X), np.int64)
        for _ in range(rng.integers(2, 5)):
            cy, cx = rng.integers(0, self.Y), rng.integers(0, self.X)
            bev[max(0, cy - 3):cy + 3, max(0, cx - 2):cx + 2] = 1
        return {"images": images, "rots": rots, "trans": trans,
                "intrins": intrins, "bev": bev}


def collate(batch):
    return {k: torch.from_numpy(np.stack([b[k] for b in batch])) for k in batch[0]}
