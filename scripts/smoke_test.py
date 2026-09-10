"""End-to-end smoke test for Lift-Splat-Shoot BEV segmentation (CPU)."""

from __future__ import annotations

import sys

import torch

from bevseg.dataset import SyntheticBEV, collate
from bevseg.metrics import IoUMeter
from bevseg.model import LiftSplatShoot

CFG = {
    "data": {"n_cams": 4, "image_h": 64, "image_w": 96,
             "xbound": [-20.0, 20.0, 1.0], "ybound": [-20.0, 20.0, 1.0],
             "zbound": [-5.0, 5.0, 10.0], "dbound": [4.0, 25.0, 2.0]},
    "model": {"context_channels": 16, "num_classes": 2, "cam_base": 16, "bev_base": 32},
}


def test_geometry() -> None:
    from bevseg.frustum import make_frustum, get_geometry, voxel_pooling
    fr = make_frustum(64, 96, 8, 12, [4.0, 25.0, 2.0])
    D = fr.shape[0]
    rots = torch.eye(3)[None, None].repeat(2, 4, 1, 1)
    trans = torch.zeros(2, 4, 3)
    intr = torch.eye(3)[None, None].repeat(2, 4, 1, 1)
    geom = get_geometry(fr, rots, trans, intr)
    assert geom.shape == (2, 4, D, 8, 12, 3)
    feats = torch.randn(2, 4, 16, D, 8, 12)
    bev = voxel_pooling(geom, feats, {"xbound": [-20, 20, 1], "ybound": [-20, 20, 1], "zbound": [-5, 5, 10]})
    assert bev.shape == (2, 16, 40, 40)
    print(f"  geometry OK: frustum D={D}, bev {tuple(bev.shape)}")


def test_train_step() -> None:
    ds = SyntheticBEV(CFG["data"], length=4)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=collate)
    model = LiftSplatShoot(CFG)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = torch.nn.CrossEntropyLoss()
    meter = IoUMeter(2)

    first, last = None, None
    for _ in range(2):
        for b in loader:
            out = model(b["images"].float(), b["rots"].float(), b["trans"].float(), b["intrins"].float())
            assert out.shape[1] == 2 and out.shape[-2:] == b["bev"].shape[-2:]
            loss = crit(out, b["bev"])
            opt.zero_grad(); loss.backward(); opt.step()
            last = loss.item()
            if first is None:
                first = last
    model.eval()
    with torch.no_grad():
        for b in loader:
            out = model(b["images"].float(), b["rots"].float(), b["trans"].float(), b["intrins"].float())
            meter.update(out.argmax(1), b["bev"])
    print(f"  train loss {first:.4f} -> {last:.4f}  val_mIoU={meter.miou():.4f}")
    assert torch.isfinite(torch.tensor(last)), "loss diverged"


if __name__ == "__main__":
    print("[smoke] geometry...")
    test_geometry()
    print("[smoke] train step...")
    test_train_step()
    print("[smoke] PASSED")
    sys.exit(0)
