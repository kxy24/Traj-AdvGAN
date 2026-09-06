#!/usr/bin/env python3
"""Measure generator-only inference latency."""

import sys
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[1]))

import argparse
import time

import torch

from models.generator import TrafficSignGenerator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--disable-attention", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    model = TrafficSignGenerator(use_attention=not args.disable_attention).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    x = torch.rand(1, 3, args.image_size, args.image_size, device=device)

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.runs):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    print(f"Generator-only latency: {1000 * elapsed / args.runs:.3f} ms/image")


if __name__ == "__main__":
    main()
