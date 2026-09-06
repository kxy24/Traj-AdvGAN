"""Evaluation helpers for clean-box disappearance ASR and perturbation norms."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .metrics import calculate_vanishing_asr, perturbation_norms


@dataclass
class AttackEvaluation:
    asr: float
    vanished: int
    clean_boxes: int
    avg_linf: float
    avg_l2: float
    images: int


def evaluate_generator(
    generator,
    target_model,
    dataloader,
    device: torch.device,
    eps: float,
    conf_threshold: float = 0.25,
    detector_iou_threshold: float = 0.45,
    vanish_iou_threshold: float = 0.5,
    max_batches: int | None = None,
) -> AttackEvaluation:
    was_training = generator.training
    generator.eval()
    total_vanished = 0
    total_clean = 0
    linf_sum = 0.0
    l2_sum = 0.0
    image_count = 0

    try:
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = batch[0].to(device)
            with torch.no_grad():
                raw_delta = generator(images)
                delta = raw_delta * eps
                adv = torch.clamp(images + delta, 0.0, 1.0)
                clean_boxes = target_model.detect(images, conf_threshold, detector_iou_threshold)
                adv_boxes = target_model.detect(adv, conf_threshold, detector_iou_threshold)

            vanished, clean = calculate_vanishing_asr(clean_boxes, adv_boxes, vanish_iou_threshold)
            linf, l2 = perturbation_norms(delta)
            batch_size = images.shape[0]
            total_vanished += vanished
            total_clean += clean
            linf_sum += linf * batch_size
            l2_sum += l2 * batch_size
            image_count += batch_size
    finally:
        generator.train(was_training)

    asr = 100.0 * total_vanished / total_clean if total_clean else 0.0
    return AttackEvaluation(
        asr=asr,
        vanished=total_vanished,
        clean_boxes=total_clean,
        avg_linf=linf_sum / max(image_count, 1),
        avg_l2=l2_sum / max(image_count, 1),
        images=image_count,
    )
