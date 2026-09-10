"""Train the Lift-Splat-Shoot BEV segmentation model.

Smoke test (CPU, synthetic)::

    python -m bevseg.train --config configs/smoke.yaml --synthetic

Full nuScenes training (GPU)::

    python -m bevseg.train --config configs/nuscenes.yaml
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from .dataset import NuScenesBEV, SyntheticBEV, collate
from .metrics import IoUMeter
from .model import LiftSplatShoot


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_loader(cfg, synthetic, split):
    d = cfg["data"]
    ds = SyntheticBEV(d, length=d.get("synth_len", 8)) if synthetic else NuScenesBEV(d["root"], d, split)
    return DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=(split == "train"),
                      collate_fn=collate, num_workers=cfg["train"].get("num_workers", 0),
                      drop_last=(split == "train"))


@torch.no_grad()
def evaluate(model, loader, meter, device):
    model.eval(); meter.reset()
    for b in loader:
        out = model(b["images"].to(device).float(), b["rots"].to(device).float(),
                    b["trans"].to(device).float(), b["intrins"].to(device).float())
        meter.update(out.argmax(1), b["bev"].to(device))
    return meter.miou()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    train_loader = make_loader(cfg, args.synthetic, "train")
    val_loader = make_loader(cfg, args.synthetic, "val")
    model = LiftSplatShoot(cfg).to(device)
    print(f"[bevseg] device={device} D={model.D} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    num_classes = cfg["model"]["num_classes"]
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"].get("weight_decay", 1e-4))
    meter = IoUMeter(num_classes)

    epochs = args.epochs or cfg["train"]["epochs"]
    best = 0.0
    for epoch in range(epochs):
        model.train(); running = 0.0
        for b in train_loader:
            out = model(b["images"].to(device).float(), b["rots"].to(device).float(),
                        b["trans"].to(device).float(), b["intrins"].to(device).float())
            loss = criterion(out, b["bev"].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        miou = evaluate(model, val_loader, meter, device)
        print(f"epoch {epoch+1}/{epochs}  loss={running/max(1,len(train_loader)):.4f}  val_mIoU={miou:.4f}")
        if miou >= best:
            best = miou
            torch.save({"model": model.state_dict(), "config": cfg},
                       os.path.join(args.out, "best.pth"))
    print(f"[bevseg] done. best val mIoU={best:.4f}")


if __name__ == "__main__":
    main()
