"""Save GT vs predicted BEV maps side by side."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from .dataset import NuScenesBEV, SyntheticBEV, collate
from .model import LiftSplatShoot
from .train import load_config

PALETTE = np.array([[30, 30, 30], [0, 200, 0], [0, 120, 255], [255, 200, 0]], dtype=np.uint8)


def colorize(bev: np.ndarray) -> np.ndarray:
    return PALETTE[np.clip(bev, 0, len(PALETTE) - 1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", default="bev.png")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = cfg["data"]
    ds = SyntheticBEV(d, length=4) if args.synthetic else NuScenesBEV(d["root"], d, "val")
    batch = collate([ds[args.index]])

    model = LiftSplatShoot(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])
    model.eval()
    with torch.no_grad():
        out = model(batch["images"].float(), batch["rots"].float(),
                    batch["trans"].float(), batch["intrins"].float())
        pred = out.argmax(1)[0].numpy()

    gt = colorize(batch["bev"][0].numpy())
    pr = colorize(pred)
    sep = np.full((gt.shape[0], 4, 3), 255, np.uint8)
    panel = np.concatenate([gt, sep, pr], axis=1)
    try:
        from PIL import Image
        Image.fromarray(panel).resize((panel.shape[1] * 3, panel.shape[0] * 3), Image.NEAREST).save(args.out)
        print(f"saved {args.out}  (left=GT, right=prediction)")
    except ImportError:
        np.save(args.out + ".npy", panel)


if __name__ == "__main__":
    main()
