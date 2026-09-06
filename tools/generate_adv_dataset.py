#!/usr/bin/env python3
"""Generate adversarial images for standard detector validation (mAP/Recall/F1)."""

from __future__ import annotations

import sys
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from datasets import build_dataset
from models.generator import TrafficSignGenerator
from utils.config import load_config, resolve_dataset_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-config", required=True)
    p.add_argument("--attack-config", default="configs/attack.yaml")
    p.add_argument("--data-root", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--disable-attention", action="store_true")
    args = p.parse_args()

    data_cfg = load_config(args.data_config)["dataset"]
    attack_cfg = load_config(args.attack_config)["attack"]
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    source_root = Path(resolve_dataset_split(data_cfg, args.split, args.data_root)).expanduser().resolve()
    out_root = Path(args.output).expanduser().resolve()

    dataset = build_dataset(data_cfg["name"], str(source_root), int(data_cfg.get("image_size", 640)))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    generator = TrafficSignGenerator(
        attention_reduction=int(attack_cfg.get("attention_reduction", 4)),
        use_attention=bool(attack_cfg.get("use_attention", True)) and not args.disable_attention,
    ).to(device)
    generator.load_state_dict(torch.load(args.checkpoint, map_location=device))
    generator.eval()
    eps = float(attack_cfg.get("eps", 8.0 / 255.0))

    with torch.no_grad():
        for images, paths in loader:
            images = images.to(device)
            delta = generator(images) * eps
            adv = torch.clamp(images + delta, 0.0, 1.0)[0].cpu()
            source_path = Path(paths[0]).resolve()
            try:
                rel = source_path.relative_to(source_root)
            except ValueError:
                rel = Path(source_path.name)
            target_path = out_root / rel
            target_path.parent.mkdir(parents=True, exist_ok=True)
            to_pil_image(adv).save(target_path)

    print(f"Adversarial images saved to: {out_root}")
    print("Run the victim detector's normal validation pipeline on this directory with the original labels to obtain mAP/Recall/F1.")


if __name__ == "__main__":
    main()
