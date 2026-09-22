"""
tatha_master.py
Tatha: complete single-file stack.

Sections:
  1.  Quantum          - PathIntegralPlanner, PrecisionHyperModel,
                         QEPSOSwarm, DensityMatrixSwarm
  2.  Active inference - GenerativeModel, FPI, EFE, Dirichlet learning,
                         HierarchicalAgent
  3.  LLM layer        - PromptPolicy, hash_embed, discretize,
                         TathaLLMAgent, make_anthropic_llm
  4.  Gym bridge       - Discretizer, GymWrapper
  5.  Plotting         - belief/FE/error plots and dashboard
  6.  Helpers + CLI    - demo / agent / hier / gym / llm

Dependencies:
  required : numpy, matplotlib
  optional : torch (PrecisionHyperModel), gymnasium (gym mode),
             anthropic (real LLM via TATHA_REAL_LLM=1)
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import gymnasium as gym
    _GYM_AVAILABLE = True
except ImportError:
    gym = None
    _GYM_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    torch = nn = F = None
    _TORCH_AVAILABLE = False


# ============================================================================
# 1. QUANTUM MODULES
# ============================================================================

@dataclass
class PathIntegralConfig:
    grid_size: int = 32
    n_slices: int = 16
    dt: float = 0.05
    hbar: float = 1.0
    mass: float = 1.0


class PathIntegralPlanner:
    """Euclidean path integral via split-operator Trotter decomposition."""

    def __init__(self, config: PathIntegralConfig):
        self.cfg = config
        n = config.grid_size
        self.n = n
        k = 2.0 * np.pi * np.fft.fftfreq(n)
        kx, ky = np.meshgrid(k, k, indexing="ij")
        self._kinetic = np.exp(
            -config.dt * config.hbar * (kx ** 2 + ky ** 2)
            / (2.0 * config.mass)
        )

    def _trotter_step(self, psi: np.ndarray, V: np.ndarray) -> np.ndarray:
        half = np.exp(-self.cfg.dt * V / (2.0 * self.cfg.hbar))
        psi = psi * half
        psi = np.fft.ifft2(np.fft.fft2(psi) * self._kinetic)
        return psi * half

    def propagate(self, psi0: np.ndarray, V: np.ndarray,
                  n_slices: Optional[int] = None) -> np.ndarray:
        psi = psi0.astype(np.complex128)
        for _ in range(n_slices or self.cfg.n_slices):
            psi = self._trotter_step(psi, V)
        return psi

    def expected_free_energy(self, beliefs: np.ndarray,
                               preferred: np.ndarray, V: np.ndarray) -> float:
        eps = 1e-12
        q = np.abs(beliefs) ** 2
        q = q / (q.sum() + eps)
        pref = preferred / (preferred.sum() + eps)
        kl = float(np.sum(q * np.log((q + eps) / (pref + eps))))
        pragmatic = float(-np.sum(q * np.log(np.abs(V) + eps)))
        return kl + pragmatic

    @staticmethod
    def normalize(psi: np.ndarray) -> np.ndarray:
        n = np.sqrt(np.sum(np.abs(psi) ** 2))
        return psi if n < 1e-15 else psi / n


@dataclass
class HyperModelConfig:
    state_dim: int = 64
    hidden_dim: int = 128
    n_layers: int = 3
    n_channels: int = 4
    lr: float = 3e-4
    min_precision: float = 1e-3
    max_precision: float = 1e3


if _TORCH_AVAILABLE:

    class PrecisionHyperModel(nn.Module):
        """Predicts per-channel precision by minimizing variational FE."""

        def __init__(self, cfg: HyperModelConfig):
            super().__init__()
            self.cfg = cfg
            layers: list = []
            dim = cfg.state_dim
            for _ in range(cfg.n_layers - 1):
                layers += [nn.Linear(dim, cfg.hidden_dim),
                           nn.LayerNorm(cfg.hidden_dim),
                           nn.GELU()]
                dim = cfg.hidden_dim
            layers.append(nn.Linear(dim, cfg.n_channels))
            self.net = nn.Sequential(*layers)
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=cfg.lr)

        def forward(self, state):
            raw = self.net(state)
            prec = F.softplus(raw) + self.cfg.min_precision
            return torch.clamp(prec, self.cfg.min_precision, self.cfg.max_precision)

        def step(self, state, errors):
            prec = self.forward(state)
            weighted = 0.5 * torch.sum(prec * errors ** 2, dim=-1)
            complexity = -0.5 * torch.sum(torch.log(prec + 1e-12), dim=-1)
            loss = (weighted + complexity).mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
            return {"free_energy": float(loss.detach()),
                    "mean_precision": float(prec.detach().mean())}

else:
    PrecisionHyperModel = None  # type: ignore


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

    def _mean_best(self):
        f = self.p_fit - self.p_fit.min()
        w = np.exp(-f)
        w /= w.sum() + 1e-12
        return np.sum(w[:, None] * self.p, axis=0)

    def step(self, iteration: int) -> float:
        t = iteration / max(self.cfg.max_iter - 1, 1)
        beta = self.cfg.beta_start + (self.cfg.beta_end - self.cfg.beta_start) * t
        z = np.exp(1j * self.phi)
        mean_phase = np.angle(np.mean(z))
        delta_phi = np.angle(np.exp(1j * (mean_phase - self.phi)))
        self.phi = self.phi + self.cfg.entanglement * delta_phi
        mb = self._mean_best()
        coherence = abs(np.mean(np.exp(1j * self.phi)))
        delta = beta * np.abs(self.x - mb) + self.cfg.entanglement * (1.0 - coherence)
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

    def optimize(self):
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
        d = np.zeros((grid_size, grid_size))
        span = max(hi[0] - lo[0], hi[1] - lo[1]) / grid_size
        for i in range(self.cfg.n_particles):
            sx = max(abs(self.x[i, 0] - self.g[0]) / 3.0, span)
            sy = max(abs(self.x[i, 1] - self.g[1]) / 3.0, span)
            d += np.exp(-((X - self.x[i, 0]) ** 2 / (2 * sx ** 2)
                          + (Y - self.x[i, 1]) ** 2 / (2 * sy ** 2)))
        return d / (d.sum() + 1e-12)


@dataclass
class DensitySwarmConfig:
    hilbert_dim: int = 16
    gamma_measure: float = 0.05
    gamma_decohere: float = 0.02
    dt: float = 0.01
    n_substeps: int = 10
    seed: Optional[int] = 42


class DensityMatrixSwarm:
    """Lindblad evolution of one density operator for the whole swarm."""

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

    def _make_valid(self, rho):
        rho = 0.5 * (rho + rho.conj().T)
        tr = np.trace(rho).real
        if abs(tr) > 1e-15:
            rho = rho / tr
        w, v = np.linalg.eigh(rho)
        w = np.clip(w.real, 0.0, None)
        return (v * w) @ v.conj().T

    def _build_H(self):
        d = self.d
        H = np.zeros((d, d), dtype=np.complex128)
        for i in range(d - 1):
            H[i, i + 1] = -1.0
            H[i + 1, i] = -1.0
        return H + np.diag(0.1 * np.arange(d).astype(np.complex128))

    def _build_measure_ops(self):
        ops = []
        for i in range(self.d):
            L = np.zeros((self.d, self.d), dtype=np.complex128)
            L[i, i] = 1.0
            ops.append(L)
        return ops

    def _build_decohere_ops(self):
        ops = []
        for i in range(self.d):
            for j in range(i + 1, self.d):
                L = np.zeros((self.d, self.d), dtype=np.complex128)
                L[i, j] = 1.0
                L[j, i] = 1.0
                ops.append(L)
        return ops

    def _lindblad(self, L, gamma):
        Ld = L.conj().T
        return gamma * (L @ self.rho @ Ld
                        - 0.5 * (Ld @ L @ self.rho + self.rho @ Ld @ L))

    def _drho(self):
        drho = -1j * (self.H @ self.rho - self.rho @ self.H)
        for L in self.L_measure:
            drho += self._lindblad(L, self.cfg.gamma_measure)
        for L in self.L_decohere:
            drho += self._lindblad(L, self.cfg.gamma_decohere)
        return drho

    def step(self):
        h = self.cfg.dt / self.cfg.n_substeps
        for _ in range(self.cfg.n_substeps):
            r0 = self.rho
            k1 = self._drho(); self.rho = r0 + 0.5 * h * k1
            k2 = self._drho(); self.rho = r0 + 0.5 * h * k2
            k3 = self._drho(); self.rho = r0 + h * k3
            k4 = self._drho()
            self.rho = r0 + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            self.rho = self._make_valid(self.rho)

    def evolve(self, n_steps: int):
        for _ in range(n_steps):
            self.step()

    def populations(self):
        return np.clip(np.real(np.diag(self.rho)), 0.0, None)

    def coherence(self):
        off = self.rho - np.diag(np.diag(self.rho))
        return float(np.sum(np.abs(off)))

    def entanglement_entropy(self, subsystem_dim: int) -> float:
        d = self.d
        if d % subsystem_dim != 0:
            raise ValueError("hilbert dim not divisible by subsystem dim")
        rest = d // subsystem_dim
        t = self.rho.reshape(subsystem_dim, rest, subsystem_dim, rest)
        reduced = np.einsum("irjr->ij", t)
        w = np.clip(np.linalg.eigvalsh(reduced), 1e-12, None)
        return float(-np.sum(w * np.log(w)))


# ============================================================================
# 2. ACTIVE INFERENCE MODULES
# ============================================================================

@dataclass
class GenerativeModel:
    A: list
    B: list
    C: Optional[list] = None
    D: Optional[list] = None
    E: Optional[np.ndarray] = None
    num_controls: Optional[list] = None
    normalize: bool = True
    eps: float = 1e-16

    def __post_init__(self):
        self.num_modalities = len(self.A)
        self.num_factors = len(self.B)
        self.num_obs = [a.shape[0] for a in self.A]
        self.num_states = [b.shape[0] for b in self.B]
        if self.num_controls is None:
            self.num_controls = [b.shape[2] for b in self.B]
        if self.C is None:
            self.C = [np.ones(n) / n for n in self.num_obs]
        if self.D is None:
            self.D = [np.ones(n) / n for n in self.num_states]
        if self.E is None:
            self.E = np.ones(self.num_controls[0]) / self.num_controls[0]
        if self.normalize:
            self._normalize()
        self.validate()

    def _normalize(self):
        self.A = [a / (a.sum(axis=0, keepdims=True) + self.eps) for a in self.A]
        B_new = []
        for b in self.B:
            s, sp, a = b.shape
            flat = b.reshape(s, sp * a)
            flat = flat / (flat.sum(axis=0, keepdims=True) + self.eps)
            B_new.append(flat.reshape(s, sp, a))
        self.B = B_new
        self.C = [c / (c.sum() + self.eps) for c in self.C]
        self.D = [d / (d.sum() + self.eps) for d in self.D]
        self.E = self.E / (self.E.sum() + self.eps)

    def validate(self):
        for m, a in enumerate(self.A):
            if a.shape[1] != self.num_states[0]:
                raise ValueError(f"A[{m}] second dim != num_states[0]")
            if not np.allclose(a.sum(axis=0), 1.0, atol=1e-6):
                raise ValueError(f"A[{m}] columns not stochastic")
        for f, b in enumerate(self.B):
            if b.shape[0] != self.num_states[f] or b.shape[1] != self.num_states[f]:
                raise ValueError(f"B[{f}] state dims mismatch")
            if not np.allclose(b.sum(axis=0), 1.0, atol=1e-6):
                raise ValueError(f"B[{f}] columns not stochastic")

    def log_A(self): return [np.log(a + self.eps) for a in self.A]
    def log_B(self): return [np.log(b + self.eps) for b in self.B]
    def log_D(self): return [np.log(d + self.eps) for d in self.D]

    def summary(self):
        return {"modalities": self.num_modalities, "factors": self.num_factors,
                "obs": self.num_obs, "states": self.num_states,
                "controls": self.num_controls}


def _log(x, eps=1e-16): return np.log(x + eps)


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-16)


@dataclass
class FPIResult:
    qs: list
    free_energy: float
    iterations: int
    converged: bool
    fe_trace: list


def free_energy(model, obs, qs, qs_prev=None, action=None, eps=1e-16):
    if qs_prev is None:
        qs_prev = [d.copy() for d in model.D]
    F_val = 0.0
    for q in qs:
        F_val += float(np.sum(q * _log(q, eps)))
    for m in range(model.num_modalities):
        ll = np.zeros_like(qs[0])
        for f in range(model.num_factors):
            if model.A[m].shape[1] == model.num_states[f]:
                ll = ll + _log(model.A[m][obs[m], :], eps)
        F_val -= float(np.sum(qs[0] * ll))
    for f, q in enumerate(qs):
        F_val -= float(np.sum(q * _log(model.D[f], eps)))
    for f in range(model.num_factors):
        b_f = model.B[f]
        if action is not None and b_f.ndim == 3:
            b_f = b_f[:, :, action]
        if b_f.ndim == 2:
            F_val -= float(np.sum(qs[f] * _log(b_f @ qs_prev[f], eps)))
    return F_val


def run_fpi(model, obs, qs_prev=None, action=None, num_iter=16, tol=1e-4):
    log_A = model.log_A()
    log_B = model.log_B()
    log_D = model.log_D()
    qs = [d.copy() for d in model.D]
    if qs_prev is None:
        qs_prev = [d.copy() for d in model.D]
    fe_trace = []
    converged = False
    it = 0
    for it in range(num_iter):
        for f in range(model.num_factors):
            log_q = log_D[f].copy()
            for m in range(model.num_modalities):
                if model.A[m].shape[1] == model.num_states[f]:
                    log_q = log_q + log_A[m][obs[m], :]
            b_f = log_B[f]
            if action is not None and b_f.ndim == 3:
                b_f = b_f[:, :, action]
            if b_f.ndim == 2:
                log_q = log_q + _log(b_f @ qs_prev[f])
            qs[f] = _softmax(log_q)
        fe = free_energy(model, obs, qs, qs_prev, action)
        fe_trace.append(fe)
        if it > 0 and abs(fe_trace[-1] - fe_trace[-2]) < tol:
            converged = True
            break
    return FPIResult(qs=qs,
                     free_energy=fe_trace[-1] if fe_trace else 0.0,
                     iterations=it + 1,
                     converged=converged,
                     fe_trace=fe_trace)


@dataclass
class EFEBreakdown:
    total: float
    risk: float
    ambiguity: float
    epistemic: float
    policy: tuple = field(default_factory=tuple)


def _kl(p, q, eps=1e-16):
    p = p + eps; q = q + eps
    return float(np.sum(p * np.log(p / q)))


def _ent(p, eps=1e-16):
    p = p + eps
    return float(-np.sum(p * np.log(p)))


def predict_beliefs(model, qs, action):
    out = []
    for f in range(model.num_factors):
        b_f = model.B[f]
        if b_f.ndim == 3:
            b_f = b_f[:, :, action]
        out.append(b_f @ qs[f])
    return out


def compute_efe(model, qs, policy):
    qs_pred = [q.copy() for q in qs]
    risk = amb_tot = epist_tot = 0.0
    for a in policy:
        qs_pred = predict_beliefs(model, qs_pred, a)
        for m in range(model.num_modalities):
            q_o = model.A[m] @ qs_pred[0]
            risk += _kl(q_o, model.C[m])
            amb = 0.0
            for s in range(model.num_states[0]):
                amb += qs_pred[0][s] * _ent(model.A[m][:, s])
            amb_tot += amb
            epist_tot += _ent(q_o) - amb
    total = risk + (amb_tot - epist_tot)
    return EFEBreakdown(float(total), float(risk), float(amb_tot),
                        float(epist_tot), policy)


def compute_policy_posterior(model, qs, policies, gamma=1.0):
    g = np.array([compute_efe(model, qs, p).total for p in policies])
    logits = -gamma * g
    logits -= logits.max()
    e = np.exp(logits)
    return e / (e.sum() + 1e-16)


def select_action(model, qs, policies, gamma=1.0, rng=None):
    rng = rng or np.random.default_rng()
    post = compute_policy_posterior(model, qs, policies, gamma)
    idx = int(rng.choice(len(policies), p=post))
    policy = policies[idx]
    return policy, compute_efe(model, qs, policy), post


@dataclass
class LearningState:
    pA: list
    pB: list
    pD: list
    step: int = 0


def init_dirichlet(model, scale=1.0):
    return LearningState(
        pA=[a * scale + 1e-3 for a in model.A],
        pB=[b * scale + 1e-3 for b in model.B],
        pD=[d * scale + 1e-3 for d in model.D],
    )


def _exp_A(pA):
    return [p / (p.sum(axis=0, keepdims=True) + 1e-16) for p in pA]


def _exp_B(pB):
    out = []
    for b in pB:
        s, sp, a = b.shape
        flat = b.reshape(s, sp * a)
        flat = flat / (flat.sum(axis=0, keepdims=True) + 1e-16)
        out.append(flat.reshape(s, sp, a))
    return out


def _exp_D(pD):
    return [p / (p.sum() + 1e-16) for p in pD]


def learn_step(model, state, obs, qs, qs_prev, action, lr=0.1):
    for m in range(len(state.pA)):
        for o in range(state.pA[m].shape[0]):
            if o == obs[m]:
                state.pA[m][o, :] += lr * qs[0]
    for f in range(len(state.pB)):
        state.pB[f][:, :, action] += lr * np.outer(qs[f], qs_prev[f])
    for f in range(len(state.pD)):
        state.pD[f] += lr * qs[f]
    state.step += 1
    new_model = GenerativeModel(
        A=_exp_A(state.pA), B=_exp_B(state.pB), C=model.C,
        D=_exp_D(state.pD), E=model.E, normalize=False,
    )
    return new_model, state


@dataclass
class HierarchicalConfig:
    n_contexts: int = 3
    context_persistence: float = 0.9
    gamma: float = 1.0
    num_iter: int = 16
    seed: Optional[int] = 42


class HierarchicalAgent:
    def __init__(self, level1, config, policies):
        self.L1 = level1
        self.cfg = config
        self.policies = policies
        self.rng = np.random.default_rng(config.seed)
        self.L2 = self._build_level2()
        self.q_context = np.ones(config.n_contexts) / config.n_contexts
        self.qs_prev = None
        self.context_history = []
        self.action_history = []

    def _build_level2(self):
        n = self.cfg.n_contexts
        A = [np.eye(n) * 0.9 + 0.1 / n]
        B = [np.full((n, n, 1), 1.0 / n)]
        for c in range(n):
            B[0][c, c, 0] = self.cfg.context_persistence
            off = (1.0 - self.cfg.context_persistence) / max(n - 1, 1)
            for c2 in range(n):
                if c2 != c:
                    B[0][c2, c, 0] = off
        return GenerativeModel(A=A, B=B,
                                 C=[np.ones(n) / n], D=[np.ones(n) / n])

    def _infer_context(self, obs):
        res = run_fpi(self.L2, obs=[0], qs_prev=[self.q_context],
                        action=0, num_iter=self.cfg.num_iter)
        self.q_context = res.qs[0]
        self.context_history.append(self.q_context.copy())
        return res

    def _apply_top_down(self):
        n_states = self.L1.num_states[0]
        n_ctx = len(self.q_context)
        if n_states % n_ctx != 0:
            return
        block = n_states // n_ctx
        new_D = np.zeros(n_states)
        for c in range(n_ctx):
            new_D[c * block:(c + 1) * block] = self.q_context[c] / block
        self.L1.D = [new_D / (new_D.sum() + 1e-16)]

    def step(self, obs):
        ctx_res = self._infer_context(obs)
        self._apply_top_down()
        l1_res = run_fpi(self.L1, obs=obs, qs_prev=self.qs_prev,
                          action=None, num_iter=self.cfg.num_iter)
        self.qs_prev = l1_res.qs
        policy, bd, _ = select_action(self.L1, l1_res.qs, self.policies,
                                          gamma=self.cfg.gamma, rng=self.rng)
        self.action_history.append(policy[0])
        return policy[0], bd, l1_res, ctx_res

    def context_summary(self):
        if not self.context_history:
            return {}
        arr = np.array(self.context_history)
        return {"mean_entropy":
                float(np.nanmean(-np.sum(arr * np.log(arr + 1e-16), axis=1))),
                "final_context": int(np.argmax(self.q_context)),
                "n_steps": len(self.context_history)}


# ============================================================================
# 3. LLM COGNITIVE LAYER
# ============================================================================

@dataclass
class PromptPolicy:
    name: str
    template: str
    description: str = ""


def default_policies():
    return [
        PromptPolicy("explore",
                     "Task: {task}\n\nList what you do NOT yet know about this "
                     "task, then ask one concrete question whose answer would "
                     "reduce your uncertainty the most."),
        PromptPolicy("exploit",
                     "Task: {task}\n\nProduce your best concrete answer now. "
                     "Be specific and actionable. Do not hedge."),
        PromptPolicy("verify",
                     "Task: {task}\n\nList every claim in the work so far that "
                     "is not directly supported by evidence. For each, state "
                     "how it could be checked."),
        PromptPolicy("refine",
                     "Task: {task}\n\nImprove the previous output: fix errors, "
                     "tighten reasoning, remove filler. Return only the "
                     "improved version."),
    ]


def hash_embed(text: str, dim: int = 64) -> np.ndarray:
    """Deterministic bag-of-hashes embedding; no external dependencies."""
    vec = np.zeros(dim)
    for tok in re.findall(r"\w+", text.lower()):
        h = int(hashlib.blake2b(tok.encode(), digest_size=4).hexdigest(), 16)
        vec[h % dim] += 1.0
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


def discretize(vec: np.ndarray, n_bins: int = 4) -> int:
    """Map first 3 components into [0, n_bins^3)."""
    bins = np.linspace(-1.0, 1.0, n_bins + 1)
    idx = []
    for v in vec[:3]:
        b = int(np.digitize(v, bins) - 1)
        b = max(0, min(n_bins - 1, b))
        idx.append(b)
    return int(np.ravel_multi_index(idx, [n_bins] * 3))


def build_task_model(n_states=8, n_obs=64, n_actions=4, preferred=0, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.uniform(0.5, 1.5, size=(n_obs, n_states))
    A = A / A.sum(axis=0, keepdims=True)
    B = np.zeros((n_states, n_states, n_actions))
    for a in range(n_actions):
        M = rng.uniform(0.1, 1.0, size=(n_states, n_states))
        B[:, :, a] = M / M.sum(axis=0, keepdims=True)
    C = np.ones(n_obs) / n_obs * 0.1
    C[preferred % n_obs] = 0.9
    C = C / C.sum()
    D = np.ones(n_states) / n_states
    return GenerativeModel(A=[A], B=[B], C=[C], D=[D])


@dataclass
class TurnRecord:
    turn: int
    policy_name: str
    prompt: str
    response: str
    observation_idx: int
    efe: EFEBreakdown
    free_energy: float
    belief_argmax: int
    belief_entropy: float


class TathaLLMAgent:
    """Tatha cognitive layer driving an LLM."""

    def __init__(self, llm_call, model=None, policies=None,
                 embed=hash_embed, gamma=2.0, learn_lr=0.1, seed=0):
        self.llm_call = llm_call
        self.model = model or build_task_model()
        self.policies = policies or default_policies()
        self.policy_tuples = [(i,) for i in range(len(self.policies))]
        self.embed = embed
        self.gamma = gamma
        self.learn_lr = learn_lr
        self.rng = np.random.default_rng(seed)
        self.learning = init_dirichlet(self.model, 1.0)
        self.qs_prev = None
        self.last_obs = None
        self.history = []

    def _believe(self):
        obs = self.last_obs if self.last_obs is not None else 0
        res = run_fpi(self.model, obs=[obs], qs_prev=self.qs_prev,
                        num_iter=16)
        return res.qs, res.free_energy

    def step(self, task: str) -> TurnRecord:
        turn = len(self.history)
        qs_old = self.qs_prev
        qs, F_val = self._believe()
        self.qs_prev = qs
        belief_argmax = int(np.argmax(qs[0]))
        belief_entropy = float(-np.sum(qs[0] * np.log(qs[0] + 1e-16)))
        policy, efe, _ = select_action(self.model, qs, self.policy_tuples,
                                        gamma=self.gamma, rng=self.rng)
        action_idx = policy[0]
        chosen = self.policies[action_idx]
        prompt = chosen.template.format(task=task)
        response = self.llm_call(prompt)
        obs_vec = self.embed(response)
        obs_idx = discretize(obs_vec, n_bins=4) % self.model.num_obs[0]
        prev_qs = qs_old if qs_old is not None \
            else [np.ones_like(q) / q.size for q in qs]
        self.model, self.learning = learn_step(
            self.model, self.learning, obs=[obs_idx], qs=qs,
            qs_prev=prev_qs, action=action_idx, lr=self.learn_lr,
        )
        rec = TurnRecord(turn, chosen.name, prompt, response, obs_idx,
                          efe, F_val, belief_argmax, belief_entropy)
        self.history.append(rec)
        self.last_obs = obs_idx
        return rec

    def run(self, task, max_turns=6):
        for _ in range(max_turns):
            self.step(task)
        return self.history

    def summary(self):
        if not self.history:
            return {}
        return {
            "turns": len(self.history),
            "mean_free_energy": float(np.mean(
                [h.free_energy for h in self.history])),
            "mean_efe": float(np.mean([h.efe.total for h in self.history])),
            "mean_belief_entropy": float(np.mean(
                [h.belief_entropy for h in self.history])),
            "policy_counts": {p.name: sum(1 for h in self.history
                                          if h.policy_name == p.name)
                              for p in self.policies},
        }


def make_anthropic_llm(model="claude-opus-5-5", max_tokens=2048,
                       effort="medium"):
    """Returns a callable(prompt) -> str backed by Claude."""
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("pip install anthropic") from e
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def call(prompt):
        resp = client.messages.create(
            model=model, max_tokens=max_tokens,
            output_config={"effort": effort},
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    return call


# ============================================================================
# 4. GYM BRIDGE
# ============================================================================

@dataclass
class Discretizer:
    low: np.ndarray
    high: np.ndarray
    n_bins: int = 8

    def __post_init__(self):
        self.low = np.asarray(self.low, dtype=float)
        self.high = np.asarray(self.high, dtype=float)
        self.edges = [np.linspace(self.low[i], self.high[i], self.n_bins + 1)
                      for i in range(len(self.low))]

    def __call__(self, obs):
        idx = []
        for i, e in enumerate(self.edges):
            b = int(np.digitize(obs[i], e) - 1)
            b = max(0, min(self.n_bins - 1, b))
            idx.append(b)
        return int(np.ravel_multi_index(idx, [self.n_bins] * len(idx)))

    @property
    def n_codes(self):
        return self.n_bins ** len(self.low)


class GymWrapper:
    def __init__(self, env, discretizer=None, seed=None):
        if not _GYM_AVAILABLE:
            raise ImportError("pip install gymnasium")
        self.env = env if not isinstance(env, str) else gym.make(env)
        self.discretizer = discretizer
        self.rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._encode(obs), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        return self._encode(obs), float(r), term, trunc, info

    def _encode(self, obs):
        arr = np.atleast_1d(np.asarray(obs, dtype=float))
        if self.discretizer is not None:
            return self.discretizer(arr)
        return int(arr.ravel()[0])

    def close(self):
        self.env.close()


# ============================================================================
# 5. PLOTTING
# ============================================================================

@dataclass
class VisualizationConfig:
    figsize: tuple = (10.0, 6.0)
    dpi: int = 120
    cmap: str = "viridis"
    title_size: int = 13
    label_size: int = 11


def plot_belief_dynamics(beliefs, cfg=None, title="Belief dynamics"):
    cfg = cfg or VisualizationConfig()
    fig, ax = plt.subplots(figsize=cfg.figsize, dpi=cfg.dpi)
    im = ax.imshow(beliefs.T, aspect="auto", cmap=cfg.cmap,
                     interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_xlabel("timestep"); ax.set_ylabel("state")
    ax.set_title(title, fontsize=cfg.title_size)
    fig.colorbar(im, ax=ax, label="P(s)")
    fig.tight_layout()
    return fig


def plot_free_energy(fe_trace, cfg=None, title="Free energy"):
    cfg = cfg or VisualizationConfig()
    fig, ax = plt.subplots(figsize=cfg.figsize, dpi=cfg.dpi)
    ax.plot(range(len(fe_trace)), fe_trace, marker="o", linewidth=1.5)
    ax.set_xlabel("iteration"); ax.set_ylabel("F")
    ax.set_title(title, fontsize=cfg.title_size)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_dashboard(beliefs, fe_trace, errors, breakdowns=None, cfg=None):
    cfg = cfg or VisualizationConfig()
    fig, axes = plt.subplots(2, 2,
                               figsize=(cfg.figsize[0] * 1.4, cfg.figsize[1]),
                               dpi=cfg.dpi)
    axes[0, 0].imshow(beliefs.T, aspect="auto", cmap=cfg.cmap,
                        interpolation="nearest", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("beliefs", fontsize=cfg.title_size)
    axes[0, 1].plot(range(len(fe_trace)), fe_trace, marker="o")
    axes[0, 1].set_title("free energy"); axes[0, 1].grid(True, alpha=0.3)
    for c in range(errors.shape[1]):
        axes[1, 0].plot(range(errors.shape[0]), errors[:, c])
    axes[1, 0].set_title("prediction errors"); axes[1, 0].grid(True, alpha=0.3)
    if breakdowns:
        labels = [str(b.policy) for b in breakdowns]
        axes[1, 1].bar(labels, [b.total for b in breakdowns])
        axes[1, 1].set_title("EFE total"); axes[1, 1].grid(True, axis="y",
                                                                 alpha=0.3)
    else:
        axes[1, 1].axis("off")
    fig.tight_layout()
    return fig


# ============================================================================
# 6. HELPERS AND CLI
# ============================================================================

def build_grid_model(n=4):
    n_s = n * n
    A = np.eye(n_s) * 0.9 + 0.1 / n_s
    A = A / A.sum(axis=0, keepdims=True)
    B = np.zeros((n_s, n_s, 4))
    for a in range(4):
        for i in range(n):
            for j in range(n):
                cur = i * n + j
                if a == 0:   ni, nj = (i + 1) % n, j
                elif a == 1: ni, nj = (i - 1) % n, j
                elif a == 2: ni, nj = i, (j + 1) % n
                else:        ni, nj = i, (j - 1) % n
                B[ni * n + nj, cur, a] = 1.0
    B = B / B.sum(axis=0, keepdims=True)
    C = np.ones(n_s) * 0.1 / n_s; C[-1] = 0.9; C = C / C.sum()
    D = np.ones(n_s) / n_s
    return GenerativeModel(A=[A], B=[B], C=[C], D=[D])


def all_policies(n_actions, horizon):
    return list(product(range(n_actions), repeat=horizon))


def cmd_demo():
    print("=== Generative Model ===")
    model = build_grid_model(4)
    print(model.summary())
    print("\n=== FPI ===")
    res = run_fpi(model, obs=[0], num_iter=32)
    print(f"argmax={int(np.argmax(res.qs[0]))} F={res.free_energy:.4f} "
          f"iters={res.iterations} converged={res.converged}")
    print("\n=== EFE ===")
    for p in all_policies(4, 1):
        b = compute_efe(model, res.qs, p)
        print(f"policy={p} total={b.total:+.4f} risk={b.risk:+.4f} "
              f"amb={b.ambiguity:+.4f} epist={b.epistemic:+.4f}")
    print("\n=== Quantum ===")
    planner = PathIntegralPlanner(PathIntegralConfig(grid_size=16, n_slices=8))
    V = np.zeros((16, 16)); V[8, 8] = -5.0
    psi0 = np.zeros((16, 16), dtype=complex); psi0[0, 0] = 1.0
    psi = planner.normalize(planner.propagate(psi0, V))
    print(f"|psi|^2 sum = {np.sum(np.abs(psi) ** 2):.6f}")
    if _TORCH_AVAILABLE:
        hm = PrecisionHyperModel(HyperModelConfig(state_dim=32, n_channels=4))
        info = hm.step(torch.randn(8, 32), torch.randn(8, 4))
        print(f"hyper-model FE={info['free_energy']:.4f} "
              f"mean_prec={info['mean_precision']:.4f}")
    bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])
    def rastrigin(x):
        return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
    swarm = QEPSOSwarm(QEPSOConfig(n_particles=32, max_iter=100),
                         bounds, rastrigin)
    _, best_f = swarm.optimize()
    print(f"QEPSO best_f={best_f:.4f}")
    dm = DensityMatrixSwarm(DensitySwarmConfig(hilbert_dim=8))
    dm.evolve(20)
    print(f"density pop sum={dm.populations().sum():.4f} "
          f"coherence={dm.coherence():.4f} "
          f"H_ent={dm.entanglement_entropy(2):.4f}")


def cmd_agent(n=4, steps=20):
    model = build_grid_model(n)
    policies = all_policies(4, 2)
    rng = np.random.default_rng(0)
    qs_prev = None
    beliefs_hist, fe_hist, err_hist = [], [], []
    state = init_dirichlet(model, 1.0)
    true_state = 0
    for t in range(steps):
        obs = [true_state]
        res = run_fpi(model, obs, qs_prev=qs_prev, num_iter=16)
        beliefs_hist.append(res.qs[0].copy())
        fe_hist.extend(res.fe_trace)
        err_hist.append([float(np.linalg.norm(
            res.qs[0] - (qs_prev[0] if qs_prev else res.qs[0])))])
        policy, _, _ = select_action(model, res.qs, policies, rng=rng)
        action = policy[0]
        if t > 0 and qs_prev is not None:
            model, state = learn_step(model, state, obs, res.qs, qs_prev,
                                          action=action, lr=0.05)
        true_state = (true_state + (1 if action == 0 else -1)) % (n * n)
        qs_prev = res.qs
    fig = plot_dashboard(np.array(beliefs_hist), fe_hist[:40],
                           np.array(err_hist))
    fig.savefig("/tmp/tatha_agent.png")
    print(f"saved /tmp/tatha_agent.png  final F={fe_hist[-1]:.4f}")


def cmd_hier(steps=30):
    L1 = build_grid_model(4)
    agent = HierarchicalAgent(L1, HierarchicalConfig(n_contexts=3),
                                policies=all_policies(4, 1))
    for t in range(steps):
        a, bd, l1, _ = agent.step([int(t % 16)])
        if t % 5 == 0:
            print(f"t={t} action={a} EFE={bd.total:+.3f} "
                  f"F={l1.free_energy:+.3f} "
                  f"ctx={int(np.argmax(agent.q_context))}")
    print(agent.context_summary())


def cmd_gym(env_name="CartPole-v1", steps=50):
    if not _GYM_AVAILABLE:
        print("pip install gymnasium"); return
    disc = Discretizer(np.array([-2.4, -3.0, -0.2, -3.0]),
                         np.array([2.4, 3.0, 0.2, 3.0]), n_bins=4)
    env = GymWrapper(env_name, discretizer=disc, seed=0)
    obs, _ = env.reset(seed=0)
    print(f"encoded={obs} n_codes={disc.n_codes}")
    total = 0.0
    for t in range(steps):
        o, r, term, trunc, _ = env.step(t % 2)
        total += r
        if t % 10 == 0:
            print(f"t={t} obs={o} r={r:.3f} total={total:.3f}")
        if term or trunc:
            print(f"ended at t={t}"); break
    env.close()


def cmd_llm(task="Design a rate limiter for a REST API.", turns=6):
    if os.environ.get("TATHA_REAL_LLM") == "1":
        llm = make_anthropic_llm()
        print("(using Claude via ANTHROPIC_API_KEY)")
    else:
        def llm(prompt):
            if "do NOT yet know" in prompt:
                return "Unknown: target language, runtime, expected load."
            if "best concrete answer" in prompt:
                return "Token-bucket rate limiter in Python with a CLI."
            if "not directly supported by evidence" in prompt:
                return "Claim A unverified. Claim B unverified."
            return "Refined: " + prompt[:100]
        print("(using stub LLM; TATHA_REAL_LLM=1 for Claude)")

    agent = TathaLLMAgent(llm_call=llm, gamma=2.0, learn_lr=0.2)
    agent.run(task, max_turns=turns)
    for r in agent.history:
        print(f"turn={r.turn} policy={r.policy_name:<8} "
              f"EFE={r.efe.total:+.3f} F={r.free_energy:+.3f} "
              f"obs={r.observation_idx} H(q)={r.belief_entropy:.3f}")
    print("\nsummary:")
    for k, v in agent.summary().items():
        print(f"  {k}: {v}")


def main(argv=None):
    argv = argv or sys.argv
    if len(argv) < 2:
        print("usage:")
        print("  python tatha_master.py demo")
        print("  python tatha_master.py agent")
        print("  python tatha_master.py hier")
        print("  python tatha_master.py gym [env_name]")
        print("  python tatha_master.py llm [task]")
        return 0
    mode = argv[1]
    if mode == "demo":   cmd_demo()
    elif mode == "agent": cmd_agent()
    elif mode == "hier":  cmd_hier()
    elif mode == "gym":
        cmd_gym(argv[2] if len(argv) > 2 else "CartPole-v1")
    elif mode == "llm":
        task = " ".join(argv[2:]) if len(argv) > 2 \
            else "Design a rate limiter for a REST API."
        cmd_llm(task)
    else:
        print(f"unknown mode: {mode}"); return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
