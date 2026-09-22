"""Visualization for active inference: beliefs, FE, EFE breakdown, errors."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional

from active_inference import EFEBreakdown


@dataclass
class VisualizationConfig:
    figsize: tuple = (10.0, 6.0)
    dpi: int = 120
    cmap: str = "viridis"
    grid: bool = True
    title_size: int = 13
    label_size: int = 11


def plot_belief_dynamics(beliefs: np.ndarray,
                          cfg: Optional[VisualizationConfig] = None,
                          state_labels: Optional[list[str]] = None,
                          title: str = "Belief dynamics") -> plt.Figure:
    cfg = cfg or VisualizationConfig()
    T, S = beliefs.shape
    fig, ax = plt.subplots(figsize=cfg.figsize, dpi=cfg.dpi)
    im = ax.imshow(beliefs.T, aspect="auto", cmap=cfg.cmap,
                    interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_xlabel("timestep", fontsize=cfg.label_size)
    ax.set_ylabel("state", fontsize=cfg.label_size)
    ax.set_title(title, fontsize=cfg.title_size)
    if state_labels is not None:
        ax.set_yticks(range(S))
        ax.set_yticklabels(state_labels)
    fig.colorbar(im, ax=ax, label="P(s)")
    fig.tight_layout()
    return fig


def plot_free_energy(fe_trace: list[float],
                      cfg: Optional[VisualizationConfig] = None,
                      title: str = "Free energy convergence") -> plt.Figure:
    cfg = cfg or VisualizationConfig()
    fig, ax = plt.subplots(figsize=cfg.figsize, dpi=cfg.dpi)
    ax.plot(range(len(fe_trace)), fe_trace, marker="o", linewidth=1.5)
    ax.set_xlabel("iteration", fontsize=cfg.label_size)
    ax.set_ylabel("variational free energy", fontsize=cfg.label_size)
    ax.set_title(title, fontsize=cfg.title_size)
    if cfg.grid:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_efe_breakdown(breakdowns: list[EFEBreakdown],
                        cfg: Optional[VisualizationConfig] = None,
                        title: str = "EFE decomposition") -> plt.Figure:
    cfg = cfg or VisualizationConfig()
    labels = [str(b.policy) for b in breakdowns]
    risk = [b.risk for b in breakdowns]
    amb = [b.ambiguity for b in breakdowns]
    epist = [b.epistemic for b in breakdowns]
    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=cfg.figsize, dpi=cfg.dpi)
    ax.bar(x - width, risk, width, label="risk")
    ax.bar(x, amb, width, label="ambiguity")
    ax.bar(x + width, epist, width, label="epistemic")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("value", fontsize=cfg.label_size)
    ax.set_title(title, fontsize=cfg.title_size)
    ax.legend()
    if cfg.grid:
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_prediction_errors(errors: np.ndarray,
                            cfg: Optional[VisualizationConfig] = None,
                            channel_labels: Optional[list[str]] = None,
                            title: str = "Prediction errors") -> plt.Figure:
    cfg = cfg or VisualizationConfig()
    T, C = errors.shape
    fig, ax = plt.subplots(figsize=cfg.figsize, dpi=cfg.dpi)
    for c in range(C):
        label = channel_labels[c] if channel_labels else f"ch{c}"
        ax.plot(range(T), errors[:, c], label=label, linewidth=1.3)
    ax.set_xlabel("timestep", fontsize=cfg.label_size)
    ax.set_ylabel("prediction error", fontsize=cfg.label_size)
    ax.set_title(title, fontsize=cfg.title_size)
    ax.legend()
    if cfg.grid:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_dashboard(beliefs: np.ndarray, fe_trace: list[float],
                    errors: np.ndarray,
                    breakdowns: Optional[list[EFEBreakdown]] = None,
                    cfg: Optional[VisualizationConfig] = None) -> plt.Figure:
    cfg = cfg or VisualizationConfig()
    fig, axes = plt.subplots(2, 2,
                             figsize=(cfg.figsize[0] * 1.4, cfg.figsize[1]),
                             dpi=cfg.dpi)
    axes[0, 0].imshow(beliefs.T, aspect="auto", cmap=cfg.cmap,
                       interpolation="nearest", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("beliefs", fontsize=cfg.title_size)
    axes[0, 1].plot(range(len(fe_trace)), fe_trace, marker="o", linewidth=1.3)
    axes[0, 1].set_title("free energy", fontsize=cfg.title_size)
    axes[0, 1].grid(True, alpha=0.3)
    for c in range(errors.shape[1]):
        axes[1, 0].plot(range(errors.shape[0]), errors[:, c], linewidth=1.2)
    axes[1, 0].set_title("prediction errors", fontsize=cfg.title_size)
    axes[1, 0].grid(True, alpha=0.3)
    if breakdowns:
        labels = [str(b.policy) for b in breakdowns]
        axes[1, 1].bar(labels, [b.total for b in breakdowns])
        axes[1, 1].set_title("EFE total", fontsize=cfg.title_size)
        axes[1, 1].grid(True, axis="y", alpha=0.3)
    else:
        axes[1, 1].axis("off")
    fig.tight_layout()
    return fig
