#!/usr/bin/env python3
"""Evaluate clean-box disappearance ASR for a trained generator."""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from datasets import build_dataset
from models.generator import TrafficSignGenerator
from utils.config import load_config, resolve_dataset_split
from utils.evaluation import evaluate_generator
from utils.yolo_wrapper import YOLOTargetModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-config", required=True)
    p.add_argument("--attack-config", default="configs/attack.yaml")
    p.add_argument("--data-root", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--victim-weights", required=True)
    p.add_argument("--yolo-repo", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--disable-attention", action="store_true", help="Use only for a matching ablation checkpoint")
    return p.parse_args()


def main():
    args = parse_args()
    data_cfg = load_config(args.data_config)["dataset"]
    method_cfg = load_config(args.attack_config)
    attack_cfg = method_cfg["attack"]
    eval_cfg = method_cfg["evaluation"]
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    val_images = resolve_dataset_split(data_cfg, "val", args.data_root)
    dataset = build_dataset(data_cfg["name"], val_images, int(data_cfg.get("image_size", 640)))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    target = YOLOTargetModel(args.victim_weights, device, args.yolo_repo)
    generator = TrafficSignGenerator(
        attention_reduction=int(attack_cfg.get("attention_reduction", 4)),
        use_attention=bool(attack_cfg.get("use_attention", True)) and not args.disable_attention,
    ).to(device)
    generator.load_state_dict(torch.load(args.checkpoint, map_location=device))

    result = evaluate_generator(
        generator,
        target,
        loader,
        device,
        eps=float(attack_cfg.get("eps", 8.0 / 255.0)),
        conf_threshold=float(eval_cfg.get("conf_threshold", 0.25)),
        detector_iou_threshold=float(eval_cfg.get("detector_iou_threshold", 0.45)),
        vanish_iou_threshold=float(eval_cfg.get("vanish_iou_threshold", 0.5)),
        max_batches=args.max_batches,
    )
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
