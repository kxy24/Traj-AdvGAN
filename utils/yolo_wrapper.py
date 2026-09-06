"""Victim-detector wrapper.

The paper evaluates several YOLO versions. Their raw head layouts differ, so this wrapper
keeps post-NMS detection and differentiable raw-head forwarding separate.
"""

from __future__ import annotations

import os
import sys
from typing import Any, List

import torch


class YOLOTargetModel:
    def __init__(self, model_path: str, device: torch.device, yolo_repo: str | None = None):
        self.device = device
        if yolo_repo:
            yolo_repo = os.path.abspath(os.path.expanduser(yolo_repo))
            if yolo_repo not in sys.path:
                sys.path.insert(0, yolo_repo)

        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(
                "Could not import a compatible ultralytics.YOLO implementation. "
                "Install the required detector package or pass --yolo-repo."
            ) from exc

        self.model = YOLO(model_path)
        self.nn = self.model.model.to(device)
        self.nn.eval()
        for p in self.nn.parameters():
            p.requires_grad_(False)

    def detect(self, images: torch.Tensor, conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> List[torch.Tensor]:
        """Return one post-NMS tensor per image, normally shaped [N, 6]."""
        with torch.no_grad():
            results = self.model.predict(
                source=images,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
            )
        boxes = []
        for result in results:
            if getattr(result, "boxes", None) is None or result.boxes.data is None:
                boxes.append(torch.zeros((0, 6), device=self.device))
            else:
                boxes.append(result.boxes.data.to(self.device))
        return boxes

    def forward_head_raw(self, images: torch.Tensor) -> Any:
        """Differentiable detector forward used by L_det and the inner PGD loop.

        Some detection heads expose their training-form raw tensors only in train mode.
        BatchNorm layers are forced to eval mode so their running statistics remain fixed.
        """
        model = self.nn
        was_training = model.training
        model.train()
        for module in model.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                module.eval()
        try:
            return model(images)
        finally:
            model.train(was_training)
