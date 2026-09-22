"""Precision Hyper-Model - adaptive precision via variational free energy."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class HyperModelConfig:
    state_dim: int = 64
    hidden_dim: int = 128
    n_layers: int = 3
    n_channels: int = 4
    lr: float = 3e-4
    min_precision: float = 1e-3
    max_precision: float = 1e3
    ema_decay: float = 0.995


class PrecisionHyperModel(nn.Module):
    """Predicts per-channel precision by minimizing variational free energy."""

    def __init__(self, cfg: HyperModelConfig):
        super().__init__()
        self.cfg = cfg
        layers: list[nn.Module] = []
        dim = cfg.state_dim
        for _ in range(cfg.n_layers - 1):
            layers += [
                nn.Linear(dim, cfg.hidden_dim),
                nn.LayerNorm(cfg.hidden_dim),
                nn.GELU(),
            ]
            dim = cfg.hidden_dim
        layers.append(nn.Linear(dim, cfg.n_channels))
        self.net = nn.Sequential(*layers)

        self.register_buffer("_mean", torch.zeros(cfg.n_channels))
        self.register_buffer("_var", torch.ones(cfg.n_channels))
        self.register_buffer("_n", torch.tensor(0, dtype=torch.long))
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=cfg.lr)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        raw = self.net(state)
        prec = F.softplus(raw) + self.cfg.min_precision
        return torch.clamp(prec, self.cfg.min_precision, self.cfg.max_precision)

    @torch.no_grad()
    def update_stats(self, prec: torch.Tensor) -> None:
        d = self.cfg.ema_decay
        self._mean.mul_(d).add_(prec.mean(dim=0), alpha=1.0 - d)
        self._var.mul_(d).add_(prec.var(dim=0, unbiased=False), alpha=1.0 - d)
        self._n += 1

    def compute_free_energy(self, errors: torch.Tensor,
                            prec: torch.Tensor) -> torch.Tensor:
        weighted = 0.5 * torch.sum(prec * errors**2, dim=-1)
        complexity = -0.5 * torch.sum(torch.log(prec + 1e-12), dim=-1)
        return (weighted + complexity).mean()

    def step(self, state: torch.Tensor, errors: torch.Tensor) -> dict[str, float]:
        prec = self.forward(state)
        loss = self.compute_free_energy(errors, prec)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        self.update_stats(prec.detach())
        return {
            "free_energy": float(loss.detach()),
            "mean_precision": float(prec.detach().mean()),
            "min_precision": float(prec.detach().min()),
            "max_precision": float(prec.detach().max()),
        }

    @torch.no_grad()
    def infer(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state)

    def save(self, path: str) -> None:
        torch.save({
            "model": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "mean": self._mean, "var": self._var, "n": self._n,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._mean.copy_(ckpt["mean"])
        self._var.copy_(ckpt["var"])
        self._n.copy_(ckpt["n"])
