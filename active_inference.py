"""Active inference stack - FPI, EFE, generative model."""
from __future__ import annotations
import sys
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Optional

import numpy as np


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
                raise ValueError(f"A[{m}] second dim mismatch")
            if not np.allclose(a.sum(axis=0), 1.0, atol=1e-6):
                raise ValueError(f"A[{m}] columns do not sum to 1")
        for f, b in enumerate(self.B):
            if b.shape[0] != self.num_states[f] or b.shape[1] != self.num_states[f]:
                raise ValueError(f"B[{f}] state dims mismatch")
            if not np.allclose(b.sum(axis=0), 1.0, atol=1e-6):
                raise ValueError(f"B[{f}] columns do not sum to 1")
        for d in self.D:
            if not np.isclose(d.sum(), 1.0, atol=1e-6):
                raise ValueError("D does not sum to 1")
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
    model: GenerativeModel, obs: list[int], qs: list[np.ndarray],
    qs_prev: Optional[list[np.ndarray]] = None,
    action: Optional[int] = None, eps: float = 1e-16,
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
    model: GenerativeModel, obs: list[int],
    qs_prev: Optional[list[np.ndarray]] = None,
    action: Optional[int] = None,
    num_iter: int = 16, tol: float = 1e-4, eps: float = 1e-16,
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


@dataclass
class EFEBreakdown:
    total: float
    risk: float
    ambiguity: float
    epistemic: float
    policy: tuple = field(default_factory=tuple)

    def as_dict(self) -> dict[str, float]:
        return {"total": self.total, "risk": self.risk,
                "ambiguity": self.ambiguity, "epistemic": self.epistemic}


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-16) -> float:
    p = p + eps
    q = q + eps
    return float(np.sum(p * np.log(p / q)))


def _entropy(p: np.ndarray, eps: float = 1e-16) -> float:
    p = p + eps
    return float(-np.sum(p * np.log(p)))


def predict_obs(model: GenerativeModel, qs: list[np.ndarray]) -> list[np.ndarray]:
    return [model.A[m] @ qs[0] for m in range(model.num_modalities)]


def predict_beliefs(model: GenerativeModel, qs: list[np.ndarray],
                    action: int) -> list[np.ndarray]:
    qs_next = []
    for f in range(model.num_factors):
        b_f = model.B[f]
        if b_f.ndim == 3:
            b_f = b_f[:, :, action]
        qs_next.append(b_f @ qs[f])
    return qs_next


def compute_efe(model: GenerativeModel, qs: list[np.ndarray],
                policy: tuple) -> EFEBreakdown:
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
        total=float(total), risk=float(total_risk),
        ambiguity=float(total_ambiguity), epistemic=float(total_epistemic),
        policy=policy,
    )


def compute_policy_posterior(model: GenerativeModel, qs: list[np.ndarray],
                             policies: list[tuple], gamma: float = 1.0) -> np.ndarray:
    breakdowns = [compute_efe(model, qs, p) for p in policies]
    g = np.array([b.total for b in breakdowns])
    logits = -gamma * g
    logits = logits - logits.max()
    e = np.exp(logits)
    return e / (e.sum() + 1e-16)


def select_action(model: GenerativeModel, qs: list[np.ndarray],
                  policies: list[tuple], gamma: float = 1.0,
                  rng: Optional[np.random.Generator] = None
                  ) -> tuple[tuple, EFEBreakdown, np.ndarray]:
    rng = rng or np.random.default_rng()
    post = compute_policy_posterior(model, qs, policies, gamma)
    idx = int(rng.choice(len(policies), p=post))
    policy = policies[idx]
    breakdown = compute_efe(model, qs, policy)
    return policy, breakdown, post


@dataclass
class LearningState:
    pA: list[np.ndarray]
    pB: list[np.ndarray]
    pD: list[np.ndarray]
    step: int = 0


def init_dirichlet(model: GenerativeModel, scale: float = 1.0) -> LearningState:
    return LearningState(
        pA=[a * scale + 1e-3 for a in model.A],
        pB=[b * scale + 1e-3 for b in model.B],
        pD=[d * scale + 1e-3 for d in model.D], step=0,
    )


def update_A(pA: list[np.ndarray], obs: list[int], qs: list[np.ndarray],
             lr: float = 1.0, modalities: Any = "all") -> list[np.ndarray]:
    pA_new = [p.copy() for p in pA]
    mods = range(len(pA)) if modalities == "all" else modalities
    for m in mods:
        n_obs, n_s = pA_new[m].shape
        for o in range(n_obs):
            if o == obs[m]:
                pA_new[m][o, :] += lr * qs[0]
    return pA_new


def update_B(pB: list[np.ndarray], qs: list[np.ndarray],
             qs_prev: list[np.ndarray], action: int,
             lr: float = 1.0, factors: Any = "all") -> list[np.ndarray]:
    pB_new = [p.copy() for p in pB]
    facs = range(len(pB)) if factors == "all" else factors
    for f in facs:
        outer = np.outer(qs[f], qs_prev[f])
        pB_new[f][:, :, action] += lr * outer
    return pB_new


def update_D(pD: list[np.ndarray], qs: list[np.ndarray],
             lr: float = 1.0) -> list[np.ndarray]:
    return [p + lr * qs[f] for f, p in enumerate(pD)]


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


def learn_step(model: GenerativeModel, state: LearningState, obs: list[int],
               qs: list[np.ndarray], qs_prev: list[np.ndarray],
               action: int, lr: float = 0.1
               ) -> tuple[GenerativeModel, LearningState]:
    state.pA = update_A(state.pA, obs, qs, lr)
    state.pB = update_B(state.pB, qs, qs_prev, action, lr)
    state.pD = update_D(state.pD, qs, lr)
    state.step += 1
    new_model = GenerativeModel(
        A=expected_A(state.pA), B=expected_B(state.pB),
        C=model.C, D=expected_D(state.pD), E=model.E, normalize=False,
    )
    return new_model, state


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
