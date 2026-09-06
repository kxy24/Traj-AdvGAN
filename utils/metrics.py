"""Metrics used by the Traj-AdvGAN paper release."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torchvision.ops import box_iou


def calculate_vanishing_asr(
    clean_boxes_list: Iterable[torch.Tensor],
    adv_boxes_list: Iterable[torch.Tensor],
    iou_thresh: float = 0.5,
) -> Tuple[int, int]:
    """Count class-agnostic clean-box disappearances.

    A clean baseline box is vanished when no adversarial box overlaps it with IoU >= iou_thresh.
    """
    missed_objects = 0
    total_clean_objects = 0

    for clean_b, adv_b in zip(clean_boxes_list, adv_boxes_list):
        if clean_b is None or len(clean_b) == 0:
            continue
        clean_xyxy = clean_b[:, :4]
        total_clean_objects += len(clean_xyxy)

        if adv_b is None or len(adv_b) == 0:
            missed_objects += len(clean_xyxy)
            continue

        adv_xyxy = adv_b[:, :4]
        ious = box_iou(clean_xyxy, adv_xyxy)
        max_ious = ious.max(dim=1).values
        missed_objects += (max_ious < iou_thresh).sum().item()

    return missed_objects, total_clean_objects


def perturbation_norms(delta: torch.Tensor) -> tuple[float, float]:
    linf = delta.abs().flatten(1).max(dim=1).values.mean().item()
    l2 = delta.flatten(1).norm(p=2, dim=1).mean().item()
    return linf, l2
