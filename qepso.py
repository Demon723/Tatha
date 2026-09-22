"""Quantum-Entangled PSO with wavefunction density."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class QEPSOConfig:
    n_particles: int = 64
    dim: int = 2
    max_iter: int = 500
    beta_start: float = 1.0
    beta_end: float = 0.5
    entanglement: float = 0.3
    contraction: float = 0.7
    seed: Optional[int] = 42
    tol: float = 1e-6
    patience: int = 50


class QEPSOSwarm:
    """QPSO with phase entanglement across the swarm."""

    def __init__(self, config: QEPSOConfig, bounds: np.ndarray,
                 objective: Callable[[np.ndarray], float]):
        self.cfg = config
        self.bounds = bounds
        self.objective = objective
        self.rng = np.random.default_rng(config.seed)

        lo, hi = bounds[:, 0], bounds[:, 1]
        self.x = self.rng.uniform(lo, hi, size=(config.n_particles, config.dim))
        self.p = self.x.copy()
        self.phi = self.rng.uniform(0.0, 2.0 * np.pi, size=config.n_particles)

        self.p_fit = np.array([objective(xi) for xi in self.x])
        self.g = self.p[int(np.argmin(self.p_fit))].copy()
        self.g_fit = float(np.min(self.p_fit))

    def _mean_best(self) -> np.ndarray:
        f = self.p_fit - self.p_fit.min()
        w = np.exp(-f)
        w /= w.sum() + 1e-12
        return np.sum(w[:, None] * self.p, axis=0)

    def _entangled_phase(self) -> np.ndarray:
        z = np.exp(1j * self.phi)
        mean_phase = np.angle(np.mean(z))
        delta = np.angle(np.exp(1j * (mean_phase - self.phi)))
        return self.phi + self.cfg.entanglement * delta

    def _delta(self, beta: float) -> np.ndarray:
        mb = self._mean_best()
        base = beta * np.abs(self.x - mb)
        coherence = abs(np.mean(np.exp(1j * self.phi)))
        return base + self.cfg.entanglement * (1.0 - coherence)

    def step(self, iteration: int) -> float:
        t = iteration / max(self.cfg.max_iter - 1, 1)
        beta = self.cfg.beta_start + (self.cfg.beta_end - self.cfg.beta_start) * t
        self.phi = self._entangled_phase()
        delta = self._delta(beta)

        u = self.rng.uniform(0.0, 1.0, size=self.x.shape)
        sign = np.where(self.rng.uniform(size=self.x.shape) < 0.5, -1.0, 1.0)
        attractor = self.cfg.contraction * self.p + (1.0 - self.cfg.contraction) * self.g
        self.x = attractor + sign * delta * np.log(1.0 / (u + 1e-12))

        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        self.x = np.clip(self.x, lo, hi)

        for i in range(self.cfg.n_particles):
            f = self.objective(self.x[i])
            if f < self.p_fit[i]:
                self.p_fit[i] = f
                self.p[i] = self.x[i].copy()

        idx = int(np.argmin(self.p_fit))
        if self.p_fit[idx] < self.g_fit:
            self.g_fit = float(self.p_fit[idx])
            self.g = self.p[idx].copy()

        return self.g_fit

    def optimize(self) -> tuple[np.ndarray, float]:
        best = self.g_fit
        stall = 0
        for it in range(self.cfg.max_iter):
            self.step(it)
            if abs(best - self.g_fit) < self.cfg.tol:
                stall += 1
                if stall >= self.cfg.patience:
                    break
            else:
                stall = 0
                best = self.g_fit
        return self.g, self.g_fit

    def wavefunction_density(self, grid_size: int) -> np.ndarray:
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        xs = np.linspace(lo[0], hi[0], grid_size)
        ys = np.linspace(lo[1], hi[1], grid_size)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        density = np.zeros((grid_size, grid_size))
        span = max(hi[0] - lo[0], hi[1] - lo[1]) / grid_size
        for i in range(self.cfg.n_particles):
            sx = max(abs(self.x[i, 0] - self.g[0]) / 3.0, span)
            sy = max(abs(self.x[i, 1] - self.g[1]) / 3.0, span)
            density += np.exp(
                -((X - self.x[i, 0]) ** 2 / (2 * sx**2)
                  + (Y - self.x[i, 1]) ** 2 / (2 * sy**2))
            )
        return density / (density.sum() + 1e-12)
