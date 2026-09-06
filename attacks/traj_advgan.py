"""Paper-aligned Traj-AdvGAN training core."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict

import torch
import torch.nn.functional as F

from models.generator import TrafficSignGenerator
from models.discriminator import TrafficSignDiscriminator


def weights_init(module):
    name = module.__class__.__name__
    if "Conv" in name and hasattr(module, "weight") and module.weight is not None:
        torch.nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in name and hasattr(module, "weight") and module.weight is not None:
        torch.nn.init.normal_(module.weight.data, 1.0, 0.02)
        if getattr(module, "bias", None) is not None:
            torch.nn.init.constant_(module.bias.data, 0.0)


class TrajAdvGAN:
    def __init__(
        self,
        device: torch.device,
        target_model,
        eps: float = 8.0 / 255.0,
        steps: int = 10,
        step_size: float = 2.0 / 255.0,
        lambda_adv: float = 1.0,
        lambda_det: float = 10.0,
        lambda_traj: float = 10.0,
        lambda_smooth: float = 0.05,
        lr_g: float = 2e-4,
        lr_d: float = 5e-6,
        attention_reduction: int = 4,
        use_attention: bool = True,
        use_trajectory: bool = True,
        use_amp: bool = True,
    ):
        self.device = device
        self.target_model = target_model
        self.eps = float(eps)
        self.steps = int(steps)
        self.step_size = float(step_size)
        self.lambda_adv = float(lambda_adv)
        self.lambda_det = float(lambda_det)
        self.lambda_traj = float(lambda_traj)
        self.lambda_smooth = float(lambda_smooth)
        self.use_trajectory = bool(use_trajectory)
        self.amp_enabled = bool(use_amp and device.type == "cuda")

        self.netG = TrafficSignGenerator(
            attention_reduction=attention_reduction,
            use_attention=use_attention,
        ).to(device)
        self.netD = TrafficSignDiscriminator().to(device)
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=lr_d, betas=(0.5, 0.999))
        self.scaler_G = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.scaler_D = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

    def _autocast(self):
        if self.amp_enabled:
            return torch.cuda.amp.autocast(enabled=True)
        return nullcontext()

    @staticmethod
    def smoothness_loss(delta: torch.Tensor) -> torch.Tensor:
        vertical = torch.abs(delta[:, :, 1:, :] - delta[:, :, :-1, :]).mean()
        horizontal = torch.abs(delta[:, :, :, 1:] - delta[:, :, :, :-1]).mean()
        return vertical + horizontal

    def detection_suppression_loss(self, predictions: Any) -> torch.Tensor:
        """Architecture-tolerant approximation of objectness/class suppression.

        When a recognizable YOLO raw-head layout is available, class/objectness logits are
        suppressed. Otherwise, the differentiable raw-head responses are suppressed as a fallback.
        """
        tensors = []

        def collect(obj):
            if isinstance(obj, torch.Tensor):
                tensors.append(obj)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    collect(item)
            elif isinstance(obj, dict):
                for item in obj.values():
                    collect(item)

        collect(predictions)
        if not tensors:
            return torch.zeros((), device=self.device)

        nc = None
        for obj in (getattr(self.target_model, "nn", None), getattr(getattr(self.target_model, "model", None), "model", None)):
            if obj is not None and hasattr(obj, "nc"):
                try:
                    nc = int(obj.nc)
                    break
                except Exception:
                    pass

        losses = []
        for tensor in tensors:
            if not tensor.is_floating_point() or tensor.numel() == 0:
                continue
            used = False

            # Anchor-style [B, A, H, W, 5 + nc]
            if tensor.ndim == 5 and tensor.shape[-1] >= 6:
                obj_logits = tensor[..., 4]
                cls_logits = tensor[..., 5:] if nc is None else tensor[..., 5:5 + nc]
                losses.append(torch.sigmoid(obj_logits).mean())
                if cls_logits.numel() > 0:
                    losses.append(torch.sigmoid(cls_logits).mean())
                used = True

            # Modern [B, C, H, W]
            if not used and tensor.ndim == 4 and nc is not None and tensor.shape[1] >= nc:
                losses.append(torch.sigmoid(tensor[:, -nc:, :, :]).mean())
                used = True

            # Flattened [B, C, N] or [B, N, C]
            if not used and tensor.ndim == 3 and nc is not None:
                if tensor.shape[1] >= nc and tensor.shape[1] < tensor.shape[2]:
                    losses.append(torch.sigmoid(tensor[:, -nc:, :]).mean())
                    used = True
                elif tensor.shape[2] >= nc:
                    losses.append(torch.sigmoid(tensor[:, :, -nc:]).mean())
                    used = True

            if not used:
                losses.append(torch.sigmoid(tensor).mean())

        if not losses:
            return torch.zeros((), device=self.device)
        return torch.stack(losses).mean()

    def inner_search(self, x_adv_init: torch.Tensor, x_clean: torch.Tensor) -> torch.Tensor:
        """AdaAD-inspired PGD trajectory search: K steps, sign gradient, L_inf projection."""
        x_temp = x_adv_init.detach().clone()
        for _ in range(self.steps):
            x_temp = x_temp.detach().requires_grad_(True)
            raw = self.target_model.forward_head_raw(x_temp)
            loss = self.detection_suppression_loss(raw)
            grad = torch.autograd.grad(loss, x_temp, only_inputs=True)[0]
            with torch.no_grad():
                x_temp = x_temp - self.step_size * grad.sign()
                delta = torch.clamp(x_temp - x_clean, -self.eps, self.eps)
                x_temp = torch.clamp(x_clean + delta, 0.0, 1.0)
        return x_temp.detach()

    def generate(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_delta = self.netG(x)
        delta = raw_delta * self.eps
        adv = torch.clamp(x + delta, 0.0, 1.0)
        return adv, delta

    def train_step(self, x: torch.Tensor, discriminator_updates: int = 1) -> Dict[str, float]:
        x = torch.clamp(x.to(self.device), 0.0, 1.0)

        with self._autocast():
            adv, delta = self.generate(x)

        loss_d_value = 0.0
        for _ in range(max(int(discriminator_updates), 0)):
            self.optimizer_D.zero_grad(set_to_none=True)
            with self._autocast():
                pred_real = self.netD(x)
                pred_fake = self.netD(adv.detach())
                loss_d = (
                    F.mse_loss(pred_real, torch.full_like(pred_real, 0.9))
                    + F.mse_loss(pred_fake, torch.full_like(pred_fake, 0.1))
                )
            self.scaler_D.scale(loss_d).backward()
            self.scaler_D.step(self.optimizer_D)
            self.scaler_D.update()
            loss_d_value = float(loss_d.detach())

        if self.use_trajectory:
            x_ref = self.inner_search(adv, x)
            delta_ideal = x_ref - x
        else:
            delta_ideal = delta.detach()

        self.optimizer_G.zero_grad(set_to_none=True)
        with self._autocast():
            pred_fake = self.netD(adv)
            loss_adv = F.mse_loss(pred_fake, torch.full_like(pred_fake, 0.9))
            raw = self.target_model.forward_head_raw(adv)
            loss_det = self.detection_suppression_loss(raw)
            loss_traj = F.l1_loss(delta, delta_ideal.detach()) if self.use_trajectory else torch.zeros((), device=self.device)
            loss_smooth = self.smoothness_loss(delta)
            loss_total = (
                self.lambda_adv * loss_adv
                + self.lambda_det * loss_det
                + self.lambda_traj * loss_traj
                + self.lambda_smooth * loss_smooth
            )

        self.scaler_G.scale(loss_total).backward()
        self.scaler_G.unscale_(self.optimizer_G)
        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), 10.0)
        self.scaler_G.step(self.optimizer_G)
        self.scaler_G.update()

        linf = delta.detach().abs().flatten(1).max(dim=1).values.mean().item()
        l2 = delta.detach().flatten(1).norm(p=2, dim=1).mean().item()
        return {
            "loss_d": loss_d_value,
            "loss_adv": float(loss_adv.detach()),
            "loss_det": float(loss_det.detach()),
            "loss_traj": float(loss_traj.detach()),
            "loss_smooth": float(loss_smooth.detach()),
            "loss_total": float(loss_total.detach()),
            "linf": linf,
            "l2": l2,
        }

    def save_generator(self, path: str) -> None:
        torch.save(self.netG.state_dict(), path)

    def load_generator(self, path: str, strict: bool = True) -> None:
        state = torch.load(path, map_location=self.device)
        self.netG.load_state_dict(state, strict=strict)
