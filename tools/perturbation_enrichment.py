#!/usr/bin/env python3
"""Compute Figure 4(a)-style perturbation enrichment using YOLO-format labels.

Expected labels: class x_center y_center width height, all coordinates normalized to [0,1].
"""

from __future__ import annotations

import sys
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.generator import TrafficSignGenerator


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def label_to_mask(label_path: Path, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if not label_path.exists():
        return mask
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        _, xc, yc, bw, bh = map(float, parts[:5])
        x1 = max(0, int((xc - bw / 2) * w))
        y1 = max(0, int((yc - bh / 2) * h))
        x2 = min(w, int(np.ceil((xc + bw / 2) * w)))
        y2 = min(h, int(np.ceil((yc + bh / 2) * h)))
        mask[y1:y2, x1:x2] = True
    return mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--top-percent", type=float, default=5.0)
    p.add_argument("--eps", type=float, default=8.0 / 255.0)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--disable-attention", action="store_true")
    args = p.parse_args()

    image_root = Path(args.images).expanduser().resolve()
    label_root = Path(args.labels).expanduser().resolve()
    image_paths = sorted(p for p in image_root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise RuntimeError(f"No images under {image_root}")

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    tfm = transforms.Compose([transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()])
    generator = TrafficSignGenerator(use_attention=not args.disable_attention).to(device)
    generator.load_state_dict(torch.load(args.checkpoint, map_location=device))
    generator.eval()

    total_pixels = 0
    total_roi_pixels = 0
    total_top_pixels = 0
    total_top_in_roi = 0

    with torch.no_grad():
        for image_path in image_paths:
            image = tfm(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
            delta = generator(image) * args.eps
            magnitude = delta.abs().mean(dim=1)[0].cpu().numpy()
            rel = image_path.relative_to(image_root).with_suffix(".txt")
            mask = label_to_mask(label_root / rel, args.image_size, args.image_size)

            k = max(1, int(magnitude.size * args.top_percent / 100.0))
            threshold = np.partition(magnitude.ravel(), -k)[-k]
            top_mask = magnitude >= threshold

            total_pixels += magnitude.size
            total_roi_pixels += int(mask.sum())
            total_top_pixels += int(top_mask.sum())
            total_top_in_roi += int((top_mask & mask).sum())

    area_ratio = total_roi_pixels / max(total_pixels, 1)
    top_inside_ratio = total_top_in_roi / max(total_top_pixels, 1)
    enrichment = top_inside_ratio / area_ratio if area_ratio > 0 else float("nan")
    print(f"Traffic-sign area ratio: {100 * area_ratio:.4f}%")
    print(f"Top-{args.top_percent:g}% perturbation pixels inside signs: {100 * top_inside_ratio:.4f}%")
    print(f"Enrichment ratio: {enrichment:.4f}x")


if __name__ == "__main__":
    main()
