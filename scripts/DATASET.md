# nuScenes for BEV map segmentation

1. Register and download from <https://www.nuscenes.org/nuscenes> — start with
   the **v1.0-mini** split (~4 GB) to prototype, then the full trainval.

2. Install the devkit: `pip install nuscenes-devkit`.

3. Implement `NuScenesBEV` in `src/bevseg/dataset.py` (currently a documented
   stub). For each `nusc.sample`, gather the 6 camera images and, per camera:
   - `intrins` from `calibrated_sensor['camera_intrinsic']` (scaled to your
     input resolution),
   - `rots`/`trans` from `calibrated_sensor` rotation/translation
     (camera→ego).
   Rasterise the BEV label from the map layers (e.g. `drivable_area`) and/or the
   annotated 3D boxes projected to BEV (vehicle occupancy). Return the tensors
   in the signature documented at the top of `dataset.py`.

The reference LSS repo (`nv-tlabs/lift-splat-shoot`) has a full nuScenes loader
you can adapt.

## Try it with no download

```bash
python scripts/smoke_test.py
python -m bevseg.train --config configs/smoke.yaml --synthetic
python -m bevseg.visualize --config configs/smoke.yaml --synthetic --checkpoint checkpoints/best.pth --out bev.png
```
