"""Streaming confusion-matrix IoU meter for BEV segmentation."""

from __future__ import annotations

import numpy as np
import torch


class IoUMeter:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self.conf = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.detach().cpu().numpy().reshape(-1)
        target = target.detach().cpu().numpy().reshape(-1)
        idx = target * self.num_classes + pred
        self.conf += np.bincount(idx, minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes)

    def per_class_iou(self) -> np.ndarray:
        tp = np.diag(self.conf).astype(np.float64)
        fp = self.conf.sum(0) - tp
        fn = self.conf.sum(1) - tp
        denom = tp + fp + fn
        return np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)

    def miou(self) -> float:
        iou = self.per_class_iou()
        return float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else 0.0
