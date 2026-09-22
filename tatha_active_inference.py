"""
tatha_active_inference.py
Complete active inference stack (single-file build).

Contents:
  1.  GenerativeModel           - A/B/C/D/E tensors with validation
  2.  FPI / variational inference - mean-field belief updating
  3.  EFE                        - risk/ambiguity/epistemic decomposition
  4.  Learning                   - Dirichlet updates for A, B, D
  5.  Gym adapter                - Gymnasium environment wrapper
  6.  Visualization              - belief/FE/error plots
  7.  HierarchicalAgent          - two-level model with top-down priors
  8.  Grid-world model builder   - canonical test environment
  9.  CLI entrypoint             - demo / agent / hier / gym modes

Dependencies:
  required: numpy, matplotlib
  optional: gymnasium (only for gym mode)

Environment:
  none required
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import gymnasium as gym
    _GYM_AVAILABLE = True
except ImportError:
    gym = None  # type: ignore
    _GYM_AVAILABLE = False


# ============================================================================
# SECTION 1: GENERATIVE MODEL
# ============================================================================

@dataclass
class GenerativeModel:
    """Discrete POMDP generative model with A, B, C, D, E tensors."""
    A: list[np.ndarray]
    B: list[np.ndarray]
    C: Optional[list[np.ndarray]] = None
    D: Optional[list[np.ndarray]] = None
    E: Optional[np.ndarray] = None
    num_controls: Optional[list[int]] = None
    control_fac_idx: Optional[list[int]] = None
    normalize: bool = True
    eps: float = 1e-16

    def __post_init__(self):
        self.num_modalities = len(self.A)
        self.num_factors = len(self.B)
        self.num_obs = [a.shape[0] for a in self.A]
        self.num_states = [b.shape[0] for b in self.B]

        if self.num_controls is None:
            self.num_controls = [b.shape[2] for b in self.B]
        if self.control_fac_idx is None:
            self.control_fac_idx = list(range(self.num_factors))

        if self.C is None:
            self.C = [np.ones(n) / n for n in self.num_obs]
        if self.D is None:
            self.D = [np.ones(n) / n for n in self.num_states]
        if self.E is None:
            self.E = np.ones(self.num_controls[0]) / self.num_controls[0]

        if self.normalize:
            self._normalize()
        self.validate()

    def _normalize(self) -> None:
        self.A = [a / (a.sum(axis=0, keepdims=True) + self.eps) for a in self.A]
        B_new = []
        for b in self.B:
            s, s_prev, a = b.shape
            flat = b.reshape(s, s_prev * a)
            flat = flat / (flat.sum(axis=0, keepdims=True) + self.eps)
            B_new.append(flat.reshape(s, s_prev, a))
        self.B = B_new
        self.C = [c / (c.sum() + self.eps) for c in self.C]
        self.D = [d / (d.sum() + self.eps) for d in self.D]
        self.E = self.E / (self.E.sum() + self.eps)

    def validate(self) -> None:
        for m, a in enumerate(self.A):
            if a.shape[1] != self.num_states[0]:
                raise ValueError(f"A[{m}] second dim {a.shape[1]} != num_states[0]")
            if not np.allclose(a.sum(axis=0), 1.0, atol=1e-6):
                raise ValueError(f"A[{m}] columns do not sum to 1")
        for f, b in enumerate(self.B):
            if b.shape[0] != self.num_states[f] or b.shape[1] != self.num_states[f]:
                raise ValueError(f"B[{f}] state dims mismatch")
            if not np.allclose(b.sum(axis=0), 1.0, atol=1e-6):
                raise ValueError(f"B[{f}] columns do not sum to 1")
        for d, arr in enumerate(self.D):
            if not np.isclose(arr.sum(), 1.0, atol=1e-6):
                raise ValueError(f"D[{d}] does not sum to 1")
        if not np.isclose(self.E.sum(), 1.0, atol=1e-6):
            raise ValueError("E does not sum to 1")

    def log_A(self) -> list[np.ndarray]:
        return [np.log(a + self.eps) for a in self.A]

    def log_B(self) -> list[np.ndarray]:
        return [np.log(b + self.eps) for b in self.B]

    def log_C(self) -> list[np.ndarray]:
        return [np.log(c + self.eps) for c in self.C]

    def log_D(self) -> list[np.ndarray]:
        return [np.log(d + self.eps) for d in self.D]

    def log_E(self) -> np.ndarray:
        return np.log(self.E + self.eps)

    def summary(self) -> dict:
        return {
            "num_modalities": self.num_modalities,
            "num_factors": self.num_factors,
            "num_obs": self.num_obs,
            "num_states": self.num_states,
            "num_controls": self.num_controls,
        }


# ============================================================================
# SECTION 2: VARIATIONAL INFERENCE (FPI)
# ============================================================================

@dataclass
class FPIResult:
    qs: list[np.ndarray]
    free_energy: float
    iterations: int
    converged: bool
    fe_trace: list[float]


def _log_stable(x: np.ndarray, eps: float = 1e-16) -> np.ndarray:
    return np.log(x + eps)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-16)


def free_energy(
    model: GenerativeModel,
    obs: list[int],
    qs: list[np.ndarray],
    qs_prev: Optional[list[np.ndarray]] = None,
    action: Optional[int] = None,
    eps: float = 1e-16,
) -> float:
    if qs_prev is None:
        qs_prev = [d.copy() for d in model.D]

    F = 0.0
    for q in qs:
        F += float(np.sum(q * _log_stable(q, eps)))

    for m in range(model.num_modalities):
        log_lik = np.zeros_like(qs[0])
        for f in range(model.num_factors):
            if model.A[m].shape[1] == model.num_states[f]:
                log_lik = log_lik + _log_stable(model.A[m][obs[m], :], eps)
        F -= float(np.sum(qs[0] * log_lik))

    for f, q in enumerate(qs):
        F -= float(np.sum(q * _log_stable(model.D[f], eps)))

    for f in range(model.num_factors):
        b_f = model.B[f]
        if action is not None and b_f.ndim == 3:
            b_f = b_f[:, :, action]
        if b_f.ndim == 2:
            log_B_expected = _log_stable(b_f @ qs_prev[f], eps)
            F -= float(np.sum(qs[f] * log_B_expected))

    return F


def run_fpi(
    model: GenerativeModel,
    obs: list[int],
    qs_prev: Optional[list[np.ndarray]] = None,
    action: Optional[int] = None,
    num_iter: int = 16,
    tol: float = 1e-4,
    eps: float = 1e-16,
) -> FPIResult:
    log_A = model.log_A()
    log_B = model.log_B()
    log_D = model.log_D()

    qs = [d.copy() for d in model.D]
    if qs_prev is None:
        qs_prev = [d.copy() for d in model.D]

    fe_trace: list[float] = []
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
                log_B_expected = b_f @ qs_prev[f]
                log_q = log_q + _log_stable(log_B_expected, eps)

            qs[f] = _softmax(log_q)

        fe = free_energy(model, obs, qs, qs_prev, action)
        fe_trace.append(fe)

        if it > 0 and abs(fe_trace[-1] - fe_trace[-2]) < tol:
            converged = True
            break

    return FPIResult(
        qs=qs,
        free_energy=fe_trace[-1] if fe_trace else 0.0,
        iterations=it + 1,
        converged=converged,
        fe_trace=fe_trace,
    )


def run_fpi_sequence(
    model: GenerativeModel,
    obs_seq: list[list[int]],
    actions: list[int],
    num_iter: int = 16,
    tol: float = 1e-4,
) -> list[FPIResult]:
    results: list[FPIResult] = []
    qs_prev = None
    for t, obs in enumerate(obs_seq):
        action = actions[t] if t < len(actions) else None
        res = run_fpi(model, obs, qs_prev=qs_prev, action=action,
                      num_iter=num_iter, tol=tol)
        results.append(res)
        qs_prev = res.qs
    return results


# ============================================================================
# SECTION 3: EXPECTED FREE ENERGY
# ============================================================================

@dataclass
class EFEBreakdown:
    total: float
    risk: float
    ambiguity: float
    epistemic: float
    policy: tuple = field(default_factory=tuple)

    def as_dict(self) -> dict[str, float]:
        return {
            "total": self.total,
            "risk": self.risk,
            "ambiguity": self.ambiguity,
            "epistemic": self.epistemic,
        }


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-16) -> float:
    p = p + eps
    q = q + eps
    return float(np.sum(p * np.log(p / q)))


def _entropy(p: np.ndarray, eps: float = 1e-16) -> float:
    p = p + eps
    return float(-np.sum(p * np.log(p)))


def predict_obs(model: GenerativeModel, qs: list[np.ndarray]) -> list[np.ndarray]:
    return [model.A[m] @ qs[0] for m in range(model.num_modalities)]


def predict_beliefs(
    model: GenerativeModel,
    qs: list[np.ndarray],
    action: int,
) -> list[np.ndarray]:
    qs_next = []
    for f in range(model.num_factors):
        b_f = model.B[f]
        if b_f.ndim == 3:
            b_f = b_f[:, :, action]
        qs_next.append(b_f @ qs[f])
    return qs_next


def compute_efe(
    model: GenerativeModel,
    qs: list[np.ndarray],
    policy: tuple,
) -> EFEBreakdown:
    qs_pred = [q.copy() for q in qs]
    total_risk = 0.0
    total_ambiguity = 0.0
    total_epistemic = 0.0

    for a in policy:
        qs_pred = predict_beliefs(model, qs_pred, a)
        q_o = predict_obs(model, qs_pred)

        for m in range(model.num_modalities):
            pref = model.C[m]
            total_risk += _kl(q_o[m], pref)

            amb = 0.0
            for s_idx in range(model.num_states[0]):
                p_s = qs_pred[0][s_idx]
                amb += p_s * _entropy(model.A[m][:, s_idx])
            total_ambiguity += amb

            h_qo = _entropy(q_o[m])
            total_epistemic += h_qo - amb

    ambiguity_net = total_ambiguity - total_epistemic
    total = total_risk + ambiguity_net

    return EFEBreakdown(
        total=float(total),
        risk=float(total_risk),
        ambiguity=float(total_ambiguity),
        epistemic=float(total_epistemic),
        policy=policy,
    )


def compute_policy_posterior(
    model: GenerativeModel,
    qs: list[np.ndarray],
    policies: list[tuple],
    gamma: float = 1.0,
) -> np.ndarray:
    breakdowns = [compute_efe(model, qs, p) for p in policies]
    g = np.array([b.total for b in breakdowns])
    logits = -gamma * g
    logits = logits - logits.max()
    e = np.exp(logits)
    return e / (e.sum() + 1e-16)


def select_action(
    model: GenerativeModel,
    qs: list[np.ndarray],
    policies: list[tuple],
    gamma: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> tuple[tuple, EFEBreakdown, np.ndarray]:
    rng = rng or np.random.default_rng()
    post = compute_policy_posterior(model, qs, policies, gamma)
    idx = int(rng.choice(len(policies), p=post))
    policy = policies[idx]
    breakdown = compute_efe(model, qs, policy)
    return policy, breakdown, post


# ============================================================================
# SECTION 4: DIRICHLET LEARNING
# ============================================================================

@dataclass
class LearningState:
    pA: list[np.ndarray]
    pB: list[np.ndarray]
    pD: list[np.ndarray]
    step: int = 0


def init_dirichlet(model: GenerativeModel, scale: float = 1.0) -> LearningState:
    pA = [a * scale + 1e-3 for a in model.A]
    pB = [b * scale + 1e-3 for b in model.B]
    pD = [d * scale + 1e-3 for d in model.D]
    return LearningState(pA=pA, pB=pB, pD=pD, step=0)


def update_A(
    pA: list[np.ndarray],
    obs: list[int],
    qs: list[np.ndarray],
    lr: float = 1.0,
    modalities: Any = "all",
) -> list[np.ndarray]:
    pA_new = [p.copy() for p in pA]
    mods = range(len(pA)) if modalities == "all" else modalities
    for m in mods:
        n_obs, n_s = pA_new[m].shape
        for o in range(n_obs):
            if o == obs[m]:
                pA_new[m][o, :] += lr * qs[0]
    return pA_new


def update_B(
    pB: list[np.ndarray],
    qs: list[np.ndarray],
    qs_prev: list[np.ndarray],
    action: int,
    lr: float = 1.0,
    factors: Any = "all",
) -> list[np.ndarray]:
    pB_new = [p.copy() for p in pB]
    facs = range(len(pB)) if factors == "all" else factors
    for f in facs:
        outer = np.outer(qs[f], qs_prev[f])
        pB_new[f][:, :, action] += lr * outer
    return pB_new


def update_D(
    pD: list[np.ndarray],
    qs: list[np.ndarray],
    lr: float = 1.0,
) -> list[np.ndarray]:
    pD_new = [p.copy() for p in pD]
    for f in range(len(pD)):
        pD_new[f] += lr * qs[f]
    return pD_new


def expected_A(pA: list[np.ndarray]) -> list[np.ndarray]:
    return [p / (p.sum(axis=0, keepdims=True) + 1e-16) for p in pA]


def expected_B(pB: list[np.ndarray]) -> list[np.ndarray]:
    out = []
    for b in pB:
        s, s_prev, a = b.shape
        flat = b.reshape(s, s_prev * a)
        flat = flat / (flat.sum(axis=0, keepdims=True) + 1e-16)
        out.append(flat.reshape(s, s_prev, a))
    return out


def expected_D(pD: list[np.ndarray]) -> list[np.ndarray]:
    return [p / (p.sum() + 1e-16) for p in pD]


def learn_step(
    model: GenerativeModel,
    state: LearningState,
    obs: list[int],
    qs: list[np.ndarray],
    qs_prev: list[np.ndarray],
    action: int,
    lr: float = 0.1,
) -> tuple[GenerativeModel, LearningState]:
    state.pA = update_A(state.pA, obs, qs, lr)
    state.pB = update_B(state.pB, qs, qs_prev, action, lr)
    state.pD = update_D(state.pD, qs, lr)
    state.step += 1

    new_model = GenerativeModel(
        A=expected_A(state.pA),
        B=expected_B(state.pB),
        C=model.C,
        D=expected_D(state.pD),
        E=model.E,
        normalize=False,
    )
    return new_model, state


# ============================================================================
# SECTION 5: GYMNASIUM ADAPTER
# ============================================================================

@dataclass
class Discretizer:
    """Uniform binning of continuous observations into discrete indices."""
    low: np.ndarray
    high: np.ndarray
    n_bins: int = 8

    def __post_init__(self):
        self.low = np.asarray(self.low, dtype=float)
        self.high = np.asarray(self.high, dtype=float)
        self.edges = [
            np.linspace(self.low[i], self.high[i], self.n_bins + 1)
            for i in range(len(self.low))
        ]

    def __call__(self, obs: np.ndarray) -> int:
        idx = []
        for i, e in enumerate(self.edges):
            b = int(np.digitize(obs[i], e) - 1)
            b = max(0, min(self.n_bins - 1, b))
            idx.append(b)
        return int(np.ravel_multi_index(idx, [self.n_bins] * len(idx)))

    @property
    def n_codes(self) -> int:
        return self.n_bins ** len(self.low)


class GymWrapper:
    """Wrap a Gymnasium environment for active inference agents."""

    def __init__(
        self,
        env: Any,
        discretizer: Optional[Discretizer] = None,
        normalize_observations: bool = True,
        uncertainty_penalty: float = 0.0,
        seed: Optional[int] = None,
    ):
        if not _GYM_AVAILABLE:
            raise ImportError("gymnasium is required for GymWrapper")
        self.env = env if not isinstance(env, str) else gym.make(env)
        self.discretizer = discretizer
        self.normalize = normalize_observations
        self.uncertainty_penalty = uncertainty_penalty
        self.rng = np.random.default_rng(seed)
        self._last_obs: Any = None

    def reset(self, **kwargs) -> tuple[int, dict]:
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return self._encode(obs), info

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs
        if self.uncertainty_penalty > 0 and "belief_entropy" in info:
            reward -= self.uncertainty_penalty * info["belief_entropy"]
        return self._encode(obs), float(reward), terminated, truncated, info

    def _encode(self, obs: Any) -> int:
        arr = np.atleast_1d(np.asarray(obs, dtype=float))
        if self.normalize and self.discretizer is not None:
            arr = (arr - self.discretizer.low) / (
                self.discretizer.high - self.discretizer.low + 1e-12
            )
        if self.discretizer is not None:
            return self.discretizer(arr)
        return int(arr.ravel()[0])

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def close(self) -> None:
        self.env.close()


# ============================================================================
# SECTION 6: VISUALIZATION
# ============================================================================

@dataclass
class VisualizationConfig:
    figsize: tuple = (10.0, 6.0)
    dpi: int = 120
    cmap: str = "viridis"
    grid: bool = True
    title_size: int = 13
    label_size: int = 11


def plot_belief_dynamics(
    beliefs: np.ndarray,
    cfg: Optional[VisualizationConfig] = None,
    state_labels: Optional[list[str]] = None,
    title: str = "Belief dynamics",
):
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


def plot_free_energy(
    fe_trace: list[float],
    cfg: Optional[VisualizationConfig] = None,
    title: str = "Free energy convergence",
):
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


def plot_efe_breakdown(
    breakdowns: list[EFEBreakdown],
    cfg: Optional[VisualizationConfig] = None,
    title: str = "EFE decomposition",
):
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


def plot_prediction_errors(
    errors: np.ndarray,
    cfg: Optional[VisualizationConfig] = None,
    channel_labels: Optional[list[str]] = None,
    title: str = "Prediction errors",
):
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


def plot_dashboard(
    beliefs: np.ndarray,
    fe_trace: list[float],
    errors: np.ndarray,
    breakdowns: Optional[list[EFEBreakdown]] = None,
    cfg: Optional[VisualizationConfig] = None,
):
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


# ============================================================================
# SECTION 7: HIERARCHICAL AGENT
# ============================================================================

@dataclass
class HierarchicalConfig:
    n_contexts: int = 3
    context_persistence: float = 0.9
    gamma: float = 1.0
    num_iter: int = 16
    seed: Optional[int] = 42


class HierarchicalAgent:
    def __init__(
        self,
        level1: GenerativeModel,
        config: HierarchicalConfig,
        policies: list[tuple],
    ):
        self.L1 = level1
        self.cfg = config
        self.policies = policies
        self.rng = np.random.default_rng(config.seed)

        self.L2 = self._build_level2()
        self.q_context = np.ones(config.n_contexts) / config.n_contexts
        self.qs_prev: Optional[list[np.ndarray]] = None
        self.context_history: list[np.ndarray] = []
        self.action_history: list[int] = []

    def _build_level2(self) -> GenerativeModel:
        n = self.cfg.n_contexts
        A = [np.eye(n) * 0.9 + 0.1 / n]
        B = [np.full((n, n, 1), 1.0 / n)]
        for c in range(n):
            B[0][c, c, 0] = self.cfg.context_persistence
            off = (1.0 - self.cfg.context_persistence) / max(n - 1, 1)
            for c2 in range(n):
                if c2 != c:
                    B[0][c2, c, 0] = off
        C = [np.ones(n) / n]
        D = [np.ones(n) / n]
        return GenerativeModel(A=A, B=B, C=C, D=D)

    def _infer_context(self, obs: list[int]) -> FPIResult:
        res = run_fpi(
            self.L2,
            obs=[0],
            qs_prev=[self.q_context],
            action=0,
            num_iter=self.cfg.num_iter,
        )
        self.q_context = res.qs[0]
        self.context_history.append(self.q_context.copy())
        return res

    def _apply_top_down_prior(self, q_context: np.ndarray) -> None:
        n_states = self.L1.num_states[0]
        n_ctx = len(q_context)
        if n_states % n_ctx != 0:
            return
        block = n_states // n_ctx
        new_D = np.zeros(n_states)
        for c in range(n_ctx):
            new_D[c * block:(c + 1) * block] = q_context[c] / block
        new_D = new_D / (new_D.sum() + 1e-16)
        self.L1.D = [new_D]

    def step(self, obs: list[int]) -> tuple[int, EFEBreakdown, FPIResult, FPIResult]:
        ctx_res = self._infer_context(obs)
        self._apply_top_down_prior(self.q_context)

        l1_res = run_fpi(
            self.L1,
            obs=obs,
            qs_prev=self.qs_prev,
            action=None,
            num_iter=self.cfg.num_iter,
        )
        self.qs_prev = l1_res.qs

        policy, breakdown, _ = select_action(
            self.L1, l1_res.qs, self.policies,
            gamma=self.cfg.gamma, rng=self.rng,
        )
        action = policy[0]
        self.action_history.append(action)
        return action, breakdown, l1_res, ctx_res

    def context_summary(self) -> dict[str, float]:
        if not self.context_history:
            return {}
        arr = np.array(self.context_history)
        return {
            "mean_entropy": float(np.mean(-np.sum(arr * np.log(arr + 1e-16), axis=1))),
            "final_context": int(np.argmax(self.q_context)),
            "n_steps": len(self.context_history),
        }


# ============================================================================
# SECTION 8: GRID-WORLD MODEL BUILDER
# ============================================================================

def build_grid_model(n: int = 4) -> GenerativeModel:
    n_states = n * n
    n_actions = 4
    A = np.eye(n_states) * 0.9 + 0.1 / n_states
    A = A / A.sum(axis=0, keepdims=True)

    B = np.zeros((n_states, n_states, n_actions))
    for a in range(n_actions):
        for i in range(n):
            for j in range(n):
                cur = i * n + j
                if a == 0:
                    ni, nj = (i + 1) % n, j
                elif a == 1:
                    ni, nj = (i - 1) % n, j
                elif a == 2:
                    ni, nj = i, (j + 1) % n
                else:
                    ni, nj = i, (j - 1) % n
                B[ni * n + nj, cur, a] = 1.0
    B = B / B.sum(axis=0, keepdims=True)

    C = np.ones(n_states) * 0.1 / n_states
    C[-1] = 0.9
    C = C / C.sum()

    D = np.ones(n_states) / n_states
    return GenerativeModel(A=[A], B=[B], C=[C], D=[D])


def all_policies(n_actions: int, horizon: int) -> list[tuple]:
    return list(product(range(n_actions), repeat=horizon))


# ============================================================================
# SECTION 9: CLI MODES
# ============================================================================

def demo() -> None:
    print("=== Generative Model ===")
    model = build_grid_model(4)
    print(model.summary())

    print("\n=== FPI ===")
    res = run_fpi(model, obs=[0], num_iter=32)
    print(f"qs argmax={int(np.argmax(res.qs[0]))} F={res.free_energy:.4f} "
          f"iters={res.iterations} converged={res.converged}")

    print("\n=== EFE ===")
    policies = all_policies(4, horizon=1)
    breakdowns = [compute_efe(model, res.qs, p) for p in policies]
    for b in breakdowns[:4]:
        print(f"policy={b.policy} total={b.total:.4f} "
              f"risk={b.risk:.4f} amb={b.ambiguity:.4f} epist={b.epistemic:.4f}")
    post = compute_policy_posterior(model, res.qs, policies)
    print(f"policy posterior argmax={int(np.argmax(post))}")

    print("\n=== Learning ===")
    state = init_dirichlet(model, scale=1.0)
    new_model, state = learn_step(
        model, state, [0], res.qs,
        [np.ones(model.num_states[0]) / model.num_states[0]],
        action=0, lr=0.5,
    )
    print(f"pA[0] diag after update: {np.diag(state.pA[0])[:4]}")

    print("\n=== Hierarchical ===")
    L1 = build_grid_model(4)
    agent = HierarchicalAgent(L1, HierarchicalConfig(n_contexts=3),
                              policies=all_policies(4, horizon=1))
    for t in range(5):
        a, bd, l1, l2 = agent.step([int(t % 16)])
        print(f"t={t} action={a} EFE={bd.total:.4f} "
              f"ctx={int(np.argmax(agent.q_context))}")
    print(agent.context_summary())


def agent_loop(n: int = 4, steps: int = 20, gamma: float = 1.0) -> None:
    model = build_grid_model(n)
    policies = all_policies(4, horizon=2)
    rng = np.random.default_rng(0)

    qs_prev = None
    beliefs_hist = []
    fe_hist = []
    errors_hist = []
    state = init_dirichlet(model, scale=1.0)

    true_state = 0
    for t in range(steps):
        obs = [true_state]
        res = run_fpi(model, obs, qs_prev=qs_prev, num_iter=16)
        beliefs_hist.append(res.qs[0].copy())
        fe_hist.extend(res.fe_trace)
        err = float(np.linalg.norm(
            res.qs[0] - (qs_prev[0] if qs_prev else res.qs[0])
        ))
        errors_hist.append([err])

        policy, breakdown, _ = select_action(
            model, res.qs, policies, gamma=gamma, rng=rng,
        )
        action = policy[0]

        if t > 0 and qs_prev is not None:
            model, state = learn_step(model, state, obs, res.qs, qs_prev,
                                      action=action, lr=0.05)

        true_state = int((true_state + (1 if action == 0 else -1)) % (n * n))
        qs_prev = res.qs

    beliefs = np.array(beliefs_hist)
    errors = np.array(errors_hist)
    fig = plot_dashboard(beliefs, fe_hist[:40], errors)
    fig.savefig("/tmp/tatha_agent_dashboard.png")
    print("saved /tmp/tatha_agent_dashboard.png")
    print(f"final belief argmax={int(np.argmax(beliefs[-1]))} "
          f"final F={fe_hist[-1]:.4f}")


def hierarchical_loop(steps: int = 30) -> None:
    L1 = build_grid_model(4)
    agent = HierarchicalAgent(L1, HierarchicalConfig(n_contexts=3),
                              policies=all_policies(4, horizon=1))
    for t in range(steps):
        obs = [int(t % 16)]
        a, bd, l1, l2 = agent.step(obs)
        if t % 5 == 0:
            print(f"t={t} action={a} EFE={bd.total:.4f} "
                  f"L1_F={l1.free_energy:.4f} ctx={int(np.argmax(agent.q_context))}")
    print(agent.context_summary())


def gym_loop(env_name: str = "CartPole-v1", steps: int = 50) -> None:
    if not _GYM_AVAILABLE:
        print("gymnasium not installed; pip install gymnasium")
        return

    disc = Discretizer(
        low=np.array([-2.4, -3.0, -0.2, -3.0]),
        high=np.array([2.4, 3.0, 0.2, 3.0]),
        n_bins=4,
    )
    env = GymWrapper(env_name, discretizer=disc, seed=0)
    obs, _ = env.reset(seed=0)
    print(f"encoded obs: {obs} n_codes: {disc.n_codes}")
    total_reward = 0.0
    for t in range(steps):
        action = t % 2
        o, r, term, trunc, _ = env.step(action)
        total_reward += r
        if t % 10 == 0:
            print(f"t={t} obs={o} reward={r:.3f} total={total_reward:.3f}")
        if term or trunc:
            print(f"episode ended at t={t}")
            break
    env.close()


def main(argv: Optional[list[str]] = None) -> int:
    argv = argv or sys.argv
    argv = argv or sys.argv
    if len(argv) < 2:
        print("usage:")
        print("  python tatha_active_inference.py demo")
        print("  python tatha_active_inference.py agent")
        print("  python tatha_active_inference.py hier")
        print("  python tatha_active_inference.py gym [env_name]")
        return 0

    mode = argv[1]
    if mode == "demo":
        demo()
    elif mode == "agent":
        agent_loop()
    elif mode == "hier":
        hierarchical_loop()
    elif mode == "gym":
        env_name = argv[2] if len(argv) > 2 else "CartPole-v1"
        gym_loop(env_name)
    else:
        print(f"unknown mode: {mode}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
