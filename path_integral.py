"""Path Integral Planner - Trotter-Suzuki propagator, EFE."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class PathIntegralConfig:
    grid_size: int = 32
    n_slices: int = 16
    dt: float = 0.05
    hbar: float = 1.0
    mass: float = 1.0
    cache_propagator: bool = True


class PathIntegralPlanner:
    """Euclidean path integral via split-operator Trotter decomposition."""

    def __init__(self, config: PathIntegralConfig):
        self.cfg = config
        n = config.grid_size
        self.n = n
        k = 2.0 * np.pi * np.fft.fftfreq(n)
        kx, ky = np.meshgrid(k, k, indexing="ij")
        self._kinetic = np.exp(
            -config.dt * config.hbar * (kx**2 + ky**2) / (2.0 * config.mass)
        )
        self._propagator: Optional[np.ndarray] = None
        self._cached_hash: Optional[int] = None

    def _potential_factor(self, V: np.ndarray) -> np.ndarray:
        return np.exp(-self.cfg.dt * V / self.cfg.hbar)

    def _trotter_step(self, psi: np.ndarray, V: np.ndarray) -> np.ndarray:
        half = self._potential_factor(V) ** 0.5
        psi = psi * half
        psi = np.fft.ifft2(np.fft.fft2(psi) * self._kinetic)
        psi = psi * half
        return psi

    def propagate(self, psi0: np.ndarray, V: np.ndarray,
                  n_slices: Optional[int] = None) -> np.ndarray:
        n_slices = n_slices or self.cfg.n_slices
        psi = psi0.astype(np.complex128)
        for _ in range(n_slices):
            psi = self._trotter_step(psi, V)
        return psi

    def build_propagator(self, V: np.ndarray, force: bool = False) -> np.ndarray:
        key = hash(V.tobytes())
        if (not force and self.cfg.cache_propagator
                and self._propagator is not None
                and self._cached_hash == key):
            return self._propagator
        n = self.n
        K = np.zeros((n * n, n * n), dtype=np.complex128)
        for i in range(n):
            for j in range(n):
                psi0 = np.zeros((n, n), dtype=np.complex128)
                psi0[i, j] = 1.0
                K[:, i * n + j] = self.propagate(psi0, V).ravel()
        self._propagator = K
        self._cached_hash = key
        return K

    def transition_amplitudes(self, start: tuple[int, int],
                              V: np.ndarray) -> np.ndarray:
        psi0 = np.zeros((self.n, self.n), dtype=np.complex128)
        psi0[start] = 1.0
        return self.propagate(psi0, V)

    def expected_free_energy(self, beliefs: np.ndarray, preferred: np.ndarray,
                             V: np.ndarray) -> float:
        eps = 1e-12
        q = np.abs(beliefs) ** 2
        q = q / (q.sum() + eps)
        pref = preferred / (preferred.sum() + eps)
        kl = float(np.sum(q * np.log((q + eps) / (pref + eps))))
        pragmatic = float(-np.sum(q * np.log(np.abs(V) + eps)))
        return kl + pragmatic

    @staticmethod
    def normalize(psi: np.ndarray) -> np.ndarray:
        nrm = np.sqrt(np.sum(np.abs(psi) ** 2))
        return psi if nrm < 1e-15 else psi / nrm
