"""Density Matrix Swarm - Lindblad evolution of multi-agent density operator."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class DensitySwarmConfig:
    hilbert_dim: int = 16
    gamma_measure: float = 0.05
    gamma_decohere: float = 0.02
    dt: float = 0.01
    n_substeps: int = 10
    seed: Optional[int] = 42


class DensityMatrixSwarm:
    """Lindblad evolution of a single density operator for the whole swarm."""

    def __init__(self, config: DensitySwarmConfig):
        self.cfg = config
        d = config.hilbert_dim
        self.d = d
        self.rng = np.random.default_rng(config.seed)

        psi = self.rng.normal(size=d) + 1j * self.rng.normal(size=d)
        psi /= np.linalg.norm(psi)
        self.rho = self._make_valid(np.outer(psi, psi.conj()))

        self.H = self._build_H()
        self.L_measure = self._build_measure_ops()
        self.L_decohere = self._build_decohere_ops()

    def _make_valid(self, rho: np.ndarray) -> np.ndarray:
        rho = 0.5 * (rho + rho.conj().T)
        tr = np.trace(rho).real
        if abs(tr) > 1e-15:
            rho = rho / tr
        w, v = np.linalg.eigh(rho)
        w = np.clip(w.real, 0.0, None)
        return (v * w) @ v.conj().T

    def _build_H(self) -> np.ndarray:
        d = self.d
        H = np.zeros((d, d), dtype=np.complex128)
        for i in range(d - 1):
            H[i, i + 1] = -1.0
            H[i + 1, i] = -1.0
        H += np.diag(0.1 * np.arange(d).astype(np.complex128))
        return H

    def _build_measure_ops(self) -> list[np.ndarray]:
        ops = []
        for i in range(self.d):
            L = np.zeros((self.d, self.d), dtype=np.complex128)
            L[i, i] = 1.0
            ops.append(L)
        return ops

    def _build_decohere_ops(self) -> list[np.ndarray]:
        ops = []
        for i in range(self.d):
            for j in range(i + 1, self.d):
                L = np.zeros((self.d, self.d), dtype=np.complex128)
                L[i, j] = 1.0
                L[j, i] = 1.0
                ops.append(L)
        return ops

    def _lindblad(self, L: np.ndarray, gamma: float) -> np.ndarray:
        Ld = L.conj().T
        return gamma * (
            L @ self.rho @ Ld
            - 0.5 * (Ld @ L @ self.rho + self.rho @ Ld @ L)
        )

    def _drho(self) -> np.ndarray:
        drho = -1j * (self.H @ self.rho - self.rho @ self.H)
        for L in self.L_measure:
            drho += self._lindblad(L, self.cfg.gamma_measure)
        for L in self.L_decohere:
            drho += self._lindblad(L, self.cfg.gamma_decohere)
        return drho

    def step(self) -> None:
        h = self.cfg.dt / self.cfg.n_substeps
        for _ in range(self.cfg.n_substeps):
            r0 = self.rho
            k1 = self._drho()
            self.rho = r0 + 0.5 * h * k1
            k2 = self._drho()
            self.rho = r0 + 0.5 * h * k2
            k3 = self._drho()
            self.rho = r0 + h * k3
            k4 = self._drho()
            self.rho = r0 + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            self.rho = self._make_valid(self.rho)

    def evolve(self, n_steps: int) -> None:
        for _ in range(n_steps):
            self.step()

    def populations(self) -> np.ndarray:
        return np.clip(np.real(np.diag(self.rho)), 0.0, None)

    def coherence(self) -> float:
        off = self.rho - np.diag(np.diag(self.rho))
        return float(np.sum(np.abs(off)))

    def entanglement_entropy(self, subsystem_dim: int) -> float:
        d = self.d
        if d % subsystem_dim != 0:
            raise ValueError(f"hilbert_dim={d} not divisible by {subsystem_dim}")
        rest = d // subsystem_dim
        t = self.rho.reshape(subsystem_dim, rest, subsystem_dim, rest)
        reduced = np.einsum("irjr->ij", t)
        w = np.linalg.eigvalsh(reduced)
        w = np.clip(w, 1e-12, None)
        return float(-np.sum(w * np.log(w)))

    def measure(self, observable: np.ndarray) -> float:
        w, v = np.linalg.eigh(observable)
        probs = np.array([
            np.real(np.trace(v[:, i:i + 1] @ v[:, i:i + 1].conj().T @ self.rho))
            for i in range(len(w))
        ])
        probs = np.clip(probs, 0.0, None)
        probs /= probs.sum() + 1e-15
        k = int(self.rng.choice(len(w), p=probs))
        vec = v[:, k:k + 1]
        self.rho = self._make_valid(vec @ vec.conj().T)
        return float(w[k])
