#!/usr/bin/env python3
"""Train Traj-AdvGAN on TT100K or CCTSDB images."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from attacks.traj_advgan import TrajAdvGAN
from datasets import build_dataset
from utils.config import load_config, resolve_dataset_split
from utils.evaluation import evaluate_generator
from utils.yolo_wrapper import YOLOTargetModel


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-config", required=True, help="configs/datasets/tt100k.yaml or cctsdb.yaml")
    p.add_argument("--attack-config", default="configs/attack.yaml")
    p.add_argument("--data-root", default=None, help="Optional local dataset root override")
    p.add_argument("--victim-weights", required=True, help="Local path to the victim detector weights")
    p.add_argument("--yolo-repo", default=None, help="Optional local YOLO repository path")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default=None, help="e.g. cuda:0 or cpu")
    p.add_argument("--output", default=None)
    p.add_argument("--disable-attention", action="store_true")
    p.add_argument("--disable-trajectory", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    data_file = load_config(args.data_config)
    method_cfg = load_config(args.attack_config)
    seed_everything(args.seed)

    data_cfg = data_file["dataset"]
    attack_cfg = method_cfg["attack"]
    train_cfg = method_cfg["training"]
    eval_cfg = method_cfg["evaluation"]
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    train_images = resolve_dataset_split(data_cfg, "train", args.data_root)
    dataset = build_dataset(data_cfg["name"], train_images, int(data_cfg.get("image_size", 640)))
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 4))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(train_cfg.get("workers", 4)),
        pin_memory=device.type == "cuda",
    )

    target = YOLOTargetModel(args.victim_weights, device=device, yolo_repo=args.yolo_repo)
    use_attention = bool(attack_cfg.get("use_attention", True)) and not args.disable_attention
    use_trajectory = bool(attack_cfg.get("use_trajectory", True)) and not args.disable_trajectory
    attack = TrajAdvGAN(
        device=device,
        target_model=target,
        eps=float(attack_cfg.get("eps", 8.0 / 255.0)),
        steps=int(attack_cfg.get("steps", 10)),
        step_size=float(attack_cfg.get("step_size", 2.0 / 255.0)),
        lambda_adv=float(attack_cfg.get("lambda_adv", 1.0)),
        lambda_det=float(attack_cfg.get("lambda_det", 10.0)),
        lambda_traj=float(attack_cfg.get("lambda_traj", 10.0)),
        lambda_smooth=float(attack_cfg.get("lambda_smooth", 0.05)),
        lr_g=float(train_cfg.get("lr_g", 2e-4)),
        lr_d=float(train_cfg.get("lr_d", 5e-6)),
        attention_reduction=int(attack_cfg.get("attention_reduction", 4)),
        use_attention=use_attention,
        use_trajectory=use_trajectory,
        use_amp=bool(train_cfg.get("amp", True)),
    )

    epochs = args.epochs or int(train_cfg.get("epochs", 50))
    default_output = f"outputs/{data_cfg['name']}_{Path(args.victim_weights).stem}"
    output = Path(args.output or default_output)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "training_log.csv"
    fields = ["epoch", "loss_d", "loss_adv", "loss_det", "loss_traj", "loss_smooth", "loss_total", "linf", "l2", "quick_asr"]
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    lr_lambda = lambda ep: max(0.0, 1.0 - float(ep) / max(epochs, 1))
    scheduler_g = LambdaLR(attack.optimizer_G, lr_lambda=lr_lambda)
    scheduler_d = LambdaLR(attack.optimizer_D, lr_lambda=lr_lambda)

    print(
        f"dataset={data_cfg['name']} eps={attack.eps:.8f} K={attack.steps} eta={attack.step_size:.8f} "
        f"attention={use_attention} trajectory={use_trajectory}"
    )
    best_asr = -1.0

    for epoch in range(1, epochs + 1):
        attack.netG.train()
        attack.netD.train()
        sums = {k: 0.0 for k in fields[1:-1]}
        batches = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for images, _paths in pbar:
            metrics = attack.train_step(images)
            batches += 1
            for key in sums:
                sums[key] += metrics[key]
            pbar.set_postfix(Ldet=f"{metrics['loss_det']:.3f}", Ltraj=f"{metrics['loss_traj']:.4f}", Linf=f"{metrics['linf']:.4f}")

        scheduler_g.step()
        scheduler_d.step()
        averaged = {k: v / max(batches, 1) for k, v in sums.items()}

        quick = evaluate_generator(
            attack.netG,
            target,
            loader,
            device,
            eps=attack.eps,
            conf_threshold=float(eval_cfg.get("conf_threshold", 0.25)),
            detector_iou_threshold=float(eval_cfg.get("detector_iou_threshold", 0.45)),
            vanish_iou_threshold=float(eval_cfg.get("vanish_iou_threshold", 0.5)),
            max_batches=int(eval_cfg.get("quick_eval_batches", 8)),
        )
        row = {"epoch": epoch, **averaged, "quick_asr": quick.asr}
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

        attack.save_generator(str(output / "generator_latest.pth"))
        if quick.asr > best_asr:
            best_asr = quick.asr
            attack.save_generator(str(output / "generator_best.pth"))
        if epoch % int(train_cfg.get("save_every", 10)) == 0 or epoch == epochs:
            attack.save_generator(str(output / f"generator_epoch_{epoch}.pth"))

        print(
            f"Epoch {epoch}: ASR={quick.asr:.2f}% ({quick.vanished}/{quick.clean_boxes}), "
            f"AvgLinf={quick.avg_linf:.6f}, AvgL2={quick.avg_l2:.6f}"
        )


if __name__ == "__main__":
    main()
