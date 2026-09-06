#!/usr/bin/env python3
"""Create a perturbation-intensity heatmap for one image (paper Figure 4(b)-style visualization)."""

from __future__ import annotations

import sys
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from models.generator import TrafficSignGenerator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--eps", type=float, default=8.0 / 255.0)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--disable-attention", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    tfm = transforms.Compose([transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()])
    image = tfm(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    generator = TrafficSignGenerator(use_attention=not args.disable_attention).to(device)
    generator.load_state_dict(torch.load(args.checkpoint, map_location=device))
    generator.eval()

    with torch.no_grad():
        delta = generator(image) * args.eps
        magnitude = delta.abs().mean(dim=1)[0].cpu().numpy()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 7))
    plt.imshow(magnitude, cmap="inferno")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Saved heatmap: {output}")


if __name__ == "__main__":
    main()
