"""Evaluate a BEV checkpoint; print per-class IoU."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from .dataset import CLASS_NAMES, NuScenesBEV, SyntheticBEV, collate
from .metrics import IoUMeter
from .model import LiftSplatShoot
from .train import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = cfg["data"]
    ds = SyntheticBEV(d, length=4) if args.synthetic else NuScenesBEV(d["root"], d, "val")
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], collate_fn=collate)

    model = LiftSplatShoot(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])
    model.eval()

    meter = IoUMeter(cfg["model"]["num_classes"])
    with torch.no_grad():
        for b in loader:
            out = model(b["images"].to(device).float(), b["rots"].to(device).float(),
                        b["trans"].to(device).float(), b["intrins"].to(device).float())
            meter.update(out.argmax(1), b["bev"].to(device))

    iou = meter.per_class_iou()
    print(f"mIoU: {meter.miou():.4f}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:12s} {iou[i]:.4f}")


if __name__ == "__main__":
    main()
