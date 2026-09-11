# BEV Map Segmentation — Lift, Splat, Shoot

### ▶ Live demo: **https://khushbooshaurya5.github.io/khushboo-portfolio-projects/demos/bev-map-segmentation/**

Turn surround-view **camera** images into a top-down **bird's-eye-view (BEV)
semantic map** — no LiDAR at inference. A from-scratch implementation of
Lift-Splat-Shoot (Philion & Fidler, ECCV 2020), the method behind most modern
camera-only BEV perception stacks (BEVFusion, BEVFormer lineage).

BEV representations are how autonomous-driving stacks fuse multiple cameras into
one ego-centric map for planning — a direct "3D scene understanding" capability.

## How it works

```
6 camera images
   │  per-image CNN → context features  C   +   depth distribution over D bins
   ▼
LIFT   each pixel's feature is spread along its ray, weighted by depth prob
       → a frustum-shaped cloud of 3D features per camera
   │   (camera intrinsics + extrinsics place each frustum point in the ego frame)
   ▼
SPLAT  frustum points dropped into a BEV grid; features in a cell are summed
       (differentiable voxel pooling)
   ▼
BEV CNN → per-cell semantic class  (background / vehicle / …)
```

| Piece | File |
|-------|------|
| frustum, camera→ego geometry, voxel pooling | `frustum.py` |
| `CamEncoder` (context+depth lift), `BEVEncoder`, `LiftSplatShoot` | `model.py` |
| nuScenes loader (stub) + `SyntheticBEV` | `dataset.py` |

The geometry is fully differentiable end-to-end — gradients flow from the BEV
loss back through the splat into each camera's depth distribution, which is what
lets the network *learn* depth without depth labels.

## Quickstart

```bash
pip install -e .

# End-to-end smoke test (CPU, synthetic 4-camera rig, seconds)
python scripts/smoke_test.py

# Synthetic training + eval + a GT-vs-pred BEV panel
python -m bevseg.train --config configs/smoke.yaml --synthetic
python -m bevseg.evaluate --config configs/smoke.yaml --synthetic --checkpoint checkpoints/best.pth
python -m bevseg.visualize --config configs/smoke.yaml --synthetic --checkpoint checkpoints/best.pth --out bev.png

# Full nuScenes training (GPU) — implement the loader per scripts/DATASET.md
python -m bevseg.train --config configs/nuscenes.yaml
```

## Results (nuScenes val — fill after training)

| Task | IoU | Notes |
|------|-----|-------|
| Vehicle BEV segmentation | _TBD_ | this repo |
| Drivable-area BEV segmentation | _TBD_ | add the map layer |

> LSS reports ~32 IoU vehicle / ~73 IoU drivable on nuScenes. This compact
> version is a clear, hackable baseline; a larger CamEncoder (EfficientNet) and
> the frustum-pooling cumsum trick bring speed and accuracy up.

## Roadmap
- [ ] Real nuScenes loader (map layers + box rasterisation)
- [ ] EfficientNet camera backbone + BEV ResNet
- [ ] Temporal fusion across sweeps; multi-task (vehicle + drivable)

## References
- Philion & Fidler, *Lift, Splat, Shoot* (ECCV 2020)
- Caesar et al., *nuScenes* (CVPR 2020)

## License
MIT © Khushboo Kumari
