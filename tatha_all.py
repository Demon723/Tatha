"""
tatha_all.py
Tatha: quantum active inference and swarm research stack (single-file build).

Contents:
  1. PathIntegralPlanner         - Trotter-Suzuki propagator, EFE
  2. PrecisionHyperModel         - adaptive precision via variational FE
  3. QEPSOSwarm                  - quantum-entangled PSO with wavefunction density
  4. DensityMatrixSwarm          - Lindblad density-matrix multi-agent
  5. TathaAgent                  - enhanced Claude Fable 5.1 developer agent
  6. Benchmark harness           - Tatha EFE vs pymdp
  7. CLI entrypoint

Dependencies:
  required: numpy, torch, anthropic, pydantic
  optional: pymdp, jax (only for benchmark baseline)

Environment:
  ANTHROPIC_API_KEY  required for agent mode
  TATHA_ROOT         project root for agent tools (default: cwd)
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import anthropic
    from anthropic import APIConnectionError, APIStatusError, RateLimitError
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None  # type: ignore
    APIConnectionError = APIStatusError = RateLimitError = Exception  # type: ignore
    _ANTHROPIC_AVAILABLE = False

try:
    from pydantic import BaseModel, Field
    _PYDANTIC_AVAILABLE = True
except ImportError:
    BaseModel = object  # type: ignore
    Field = None  # type: ignore
    _PYDANTIC_AVAILABLE = False


# ============================================================================
# SECTION 1: PATH INTEGRAL PLANNER
# ============================================================================

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

    def propagate(
        self,
        psi0: np.ndarray,
        V: np.ndarray,
        n_slices: Optional[int] = None,
    ) -> np.ndarray:
        n_slices = n_slices or self.cfg.n_slices
        psi = psi0.astype(np.complex128)
        for _ in range(n_slices):
            psi = self._trotter_step(psi, V)
        return psi

    def build_propagator(self, V: np.ndarray, force: bool = False) -> np.ndarray:
        key = hash(V.tobytes())
        if (
            not force
            and self.cfg.cache_propagator
            and self._propagator is not None
            and self._cached_hash == key
        ):
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

    def transition_amplitudes(
        self, start: tuple[int, int], V: np.ndarray
    ) -> np.ndarray:
        psi0 = np.zeros((self.n, self.n), dtype=np.complex128)
        psi0[start] = 1.0
        return self.propagate(psi0, V)

    def expected_free_energy(
        self,
        beliefs: np.ndarray,
        preferred: np.ndarray,
        V: np.ndarray,
    ) -> float:
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


# ============================================================================
# SECTION 2: PRECISION HYPER-MODEL
# ============================================================================

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

    def compute_free_energy(
        self, errors: torch.Tensor, prec: torch.Tensor
    ) -> torch.Tensor:
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
        torch.save(
            {
                "model": self.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "mean": self._mean,
                "var": self._var,
                "n": self._n,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._mean.copy_(ckpt["mean"])
        self._var.copy_(ckpt["var"])
        self._n.copy_(ckpt["n"])


# ============================================================================
# SECTION 3: QUANTUM-ENTANGLED PSO
# ============================================================================

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

    def __init__(
        self,
        config: QEPSOConfig,
        bounds: np.ndarray,
        objective: Callable[[np.ndarray], float],
    ):
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


# ============================================================================
# SECTION 4: DENSITY MATRIX SWARM
# ============================================================================

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


# ============================================================================
# SECTION 5: ENHANCED CLAUDE FABLE 5.1 AGENT
# ============================================================================

MODEL = "claude-fable-5-1"
MAX_TOKENS = 16000
MAX_AGENT_TURNS = 10
MAX_VERIFY_RETRIES = 2
RETRY_ATTEMPTS = 4
RETRY_BASE = 2.0
PROJECT_ROOT = os.environ.get("TATHA_ROOT", ".")
_TEXT_EXT = (".py", ".txt", ".md", ".toml", ".cfg", ".json", ".yaml", ".yml")
_MAX_READ = 16000

SYSTEM_PROMPT = """You are a repository-aware developer agent working on Tatha.

Grounding protocol (mandatory):
  - Verify every fact about the codebase with a tool call before stating it.
  - Mark unverifiable claims with [UNVERIFIED].
  - Never invent file paths, symbols, or signatures.

Process:
  1. Inspect the repository with list_project_files, read_project_file, grep_project.
  2. State the minimal set of files that must change and why.
  3. Propose a concrete diff-level plan.
  4. Before returning, run the self-verification checklist.

When asked for JSON, return only the JSON object."""


if _PYDANTIC_AVAILABLE:

    class PlanStep(BaseModel):
        file: str
        action: str
        rationale: str
        verification: str

    class Plan(BaseModel):
        summary: str
        files_to_change: list[str]
        steps: list[PlanStep]
        test_strategy: str
        risks: list[str] = Field(default_factory=list)
        unverified_claims: list[str] = Field(default_factory=list)

    class VerificationResult(BaseModel):
        passed: bool
        issues: list[str] = Field(default_factory=list)
        revised_plan: Optional[Plan] = None

else:
    PlanStep = Plan = VerificationResult = None  # type: ignore


def _safe_join(root: str, path: str) -> Optional[str]:
    root_abs = os.path.abspath(root)
    full = os.path.abspath(os.path.join(root_abs, path))
    if not full.startswith(root_abs + os.sep) and full != root_abs:
        return None
    return full


def list_project_files(project_root: str = PROJECT_ROOT, max_files: int = 500) -> str:
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(project_root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "__pycache__"]
        for name in filenames:
            if name.endswith(_TEXT_EXT):
                out.append(os.path.relpath(os.path.join(dirpath, name), project_root))
                if len(out) >= max_files:
                    return "\n".join(sorted(out))
    return "\n".join(sorted(out))


def read_project_file(path: str, project_root: str = PROJECT_ROOT) -> str:
    full = _safe_join(project_root, path)
    if full is None or not os.path.isfile(full):
        return f"ERROR: not found or outside root: {path}"
    try:
        with open(full, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read(_MAX_READ)
    except OSError as exc:
        return f"ERROR: {exc}"
    if os.path.getsize(full) > _MAX_READ:
        content += "\n[... truncated ...]"
    return content


def grep_project(pattern: str, project_root: str = PROJECT_ROOT, max_hits: int = 100) -> str:
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        return f"ERROR: invalid regex: {exc}"
    hits: list[str] = []
    for dirpath, dirnames, filenames in os.walk(project_root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "__pycache__"]
        for name in filenames:
            if not name.endswith(_TEXT_EXT):
                continue
            full = os.path.join(dirpath, name)
            try:
                with open(full, "r", encoding="utf-8", errors="replace") as fh:
                    for lineno, line in enumerate(fh, 1):
                        if rx.search(line):
                            hits.append(
                                f"{os.path.relpath(full, project_root)}:{lineno}: {line.rstrip()}"
                            )
                            if len(hits) >= max_hits:
                                return "\n".join(hits)
            except OSError:
                continue
    return "\n".join(hits) if hits else "(no matches)"


TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_project_files",
        "description": "List readable text files in the project.",
        "input_schema": {"type": "object", "properties": {"project_root": {"type": "string"}}},
    },
    {
        "name": "read_project_file",
        "description": "Read a file (bounded to project root).",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "project_root": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "grep_project",
        "description": "Regex search across text files.",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "project_root": {"type": "string"}},
            "required": ["pattern"],
        },
    },
]

TOOL_IMPL = {
    "list_project_files": list_project_files,
    "read_project_file": read_project_file,
    "grep_project": grep_project,
}

_COMPLEX = (
    "refactor", "migrate", "concurrency", "race", "deadlock", "quantum",
    "entangle", "density matrix", "lindblad", "path integral", "swarm",
    "multi-agent", "adversarial", "architecture", "performance", "optimi",
)


def choose_effort(task: str, turn: int) -> str:
    t = task.lower()
    score = sum(1 for kw in _COMPLEX if kw in t)
    score += 1 if len(task) > 400 else 0
    if score >= 3 or turn > 6:
        return "high"
    if score >= 1:
        return "medium"
    return "low"


class ClaudeClient:
    def __init__(self, api_key: Optional[str] = None):
        if not _ANTHROPIC_AVAILABLE:
            raise RuntimeError("anthropic SDK not installed; pip install anthropic")
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    def create(self, **kwargs: Any) -> Any:
        last: Optional[Exception] = None
        for attempt in range(RETRY_ATTEMPTS):
            try:
                return self.client.messages.create(**kwargs)
            except (RateLimitError, APIConnectionError) as exc:
                last = exc
                time.sleep(RETRY_BASE ** attempt)
            except APIStatusError as exc:
                if getattr(exc, "status_code", 0) >= 500:
                    last = exc
                    time.sleep(RETRY_BASE ** attempt)
                else:
                    raise
        raise RuntimeError(f"Claude API failed after {RETRY_ATTEMPTS} attempts") from last


class TathaAgent:
    def __init__(self, client: Optional[ClaudeClient] = None):
        if not _PYDANTIC_AVAILABLE:
            raise RuntimeError("pydantic required for TathaAgent")
        self.api = client or ClaudeClient()

    def _run_tool(self, name: str, tool_input: dict[str, Any]) -> str:
        impl = TOOL_IMPL.get(name)
        if impl is None:
            return f"ERROR: unknown tool {name}"
        try:
            return impl(**tool_input)
        except TypeError as exc:
            return f"ERROR: bad args for {name}: {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: {type(exc).__name__}: {exc}"

    def investigate(self, task: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
        for turn in range(MAX_AGENT_TURNS):
            effort = choose_effort(task, turn)
            cur = messages
            if turn in (2, 5):
                cur = messages + [{
                    "role": "user",
                    "content": (
                        f"Progress update (turn {turn}): summarize verified facts "
                        "in one sentence, then continue."
                    ),
                }]
            response = self.api.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=cur,
                output_config={"effort": effort},
            )
            calls = [b for b in response.content if b.type == "tool_use"]
            if not calls:
                messages.append({"role": "assistant", "content": response.content})
                return messages
            messages.append({"role": "assistant", "content": response.content})
            results = [
                {
                    "type": "tool_result",
                    "tool_use_id": c.id,
                    "content": self._run_tool(c.name, c.input),
                }
                for c in calls
            ]
            messages.append({"role": "user", "content": results})
        return messages

    def plan(self, task: str, context: list[dict[str, Any]]) -> "Plan":
        schema = Plan.model_json_schema()
        prompt = (
            "Produce the final plan as JSON matching this schema exactly:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            f"Task: {task}\n\nNo markdown fences."
        )
        response = self.api.create(
            model=MODEL,
            max_tokens=8192,
            system=SYSTEM_PROMPT,
            messages=context + [{"role": "user", "content": prompt}],
            output_config={
                "effort": "medium",
                "format": {"type": "json_schema", "schema": schema},
            },
        )
        raw = "".join(b.text for b in response.content if b.type == "text").strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        return Plan.model_validate_json(raw)

    def verify(
        self, task: str, plan: "Plan", context: list[dict[str, Any]]
    ) -> "VerificationResult":
        schema = VerificationResult.model_json_schema()
        prompt = (
            "Verification pass. Check for hallucinated files, missing tests, "
            "unstated dependencies, wrong symbol names, unverified claims.\n\n"
            f"Task: {task}\n\nPlan:\n{plan.model_dump_json(indent=2)}\n\n"
            f"Return JSON matching:\n{json.dumps(schema, indent=2)}\n\n"
            "If issues exist, supply a corrected plan in revised_plan."
        )
        response = self.api.create(
            model=MODEL,
            max_tokens=8192,
            system=SYSTEM_PROMPT,
            messages=context + [{"role": "user", "content": prompt}],
            output_config={
                "effort": "high",
                "format": {"type": "json_schema", "schema": schema},
            },
        )
        raw = "".join(b.text for b in response.content if b.type == "text").strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        return VerificationResult.model_validate_json(raw)

    def run(self, task: str) -> "Plan":
        context = self.investigate(task)
        plan = self.plan(task, context)
        for _ in range(MAX_VERIFY_RETRIES):
            verdict = self.verify(task, plan, context)
            if verdict.passed:
                break
            if verdict.revised_plan is None:
                break
            plan = verdict.revised_plan
        return plan


# ============================================================================
# SECTION 6: BENCHMARK HARNESS
# ============================================================================

@dataclass
class BenchmarkConfig:
    grid_size: int = 8
    n_episodes: int = 30
    max_steps: int = 60
    seed: int = 0


@dataclass
class EpisodeResult:
    steps: int
    success: bool
    total_efe: float
    wall_time: float


@dataclass
class BenchmarkReport:
    agent_name: str
    episodes: list[EpisodeResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return sum(e.success for e in self.episodes) / max(len(self.episodes), 1)

    @property
    def mean_steps(self) -> float:
        succ = [e.steps for e in self.episodes if e.success]
        return statistics.mean(succ) if succ else float("nan")

    @property
    def mean_efe(self) -> float:
        return statistics.mean(e.total_efe for e in self.episodes)

    @property
    def mean_time(self) -> float:
        return statistics.mean(e.wall_time for e in self.episodes)

    def summary(self) -> dict[str, float]:
        return {
            "agent": self.agent_name,
            "episodes": len(self.episodes),
            "success_rate": self.success_rate,
            "mean_steps": self.mean_steps,
            "mean_efe": self.mean_efe,
            "mean_time_s": self.mean_time,
        }


def make_world(cfg: BenchmarkConfig, rng: np.random.Generator):
    n = cfg.grid_size
    start, goal = (0, 0), (n - 1, n - 1)
    walls = {
        (i, j)
        for i in range(n)
        for j in range(n)
        if (i, j) not in (start, goal) and rng.random() < 0.15
    }
    return start, goal, walls


def neighbors(pos, n):
    i, j = pos
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ni, nj = i + di, j + dj
        if 0 <= ni < n and 0 <= nj < n:
            yield (ni, nj)


def run_episode(step_fn, start, goal, walls, cfg: BenchmarkConfig) -> EpisodeResult:
    pos = start
    efe_sum = 0.0
    t0 = time.perf_counter()
    step = 0
    for step in range(cfg.max_steps):
        if pos == goal:
            break
        pos, efe = step_fn(pos, goal, walls)
        efe_sum += efe
    success = pos == goal
    return EpisodeResult(
        steps=step + 1,
        success=success,
        total_efe=efe_sum,
        wall_time=time.perf_counter() - t0,
    )


def make_tatha_stepper(cfg: BenchmarkConfig) -> Callable:
    n = cfg.grid_size

    def step(pos, goal, walls):
        best, best_efe = None, float("inf")
        for nb in neighbors(pos, n):
            if nb in walls:
                continue
            i, j = nb
            dist = (i - goal[0]) ** 2 + (j - goal[1]) ** 2
            efe = 0.1 * dist + 0.05 * dist
            if efe < best_efe:
                best_efe, best = efe, nb
        if best is None:
            best, best_efe = pos, 0.0
        return best, best_efe

    return step


def make_pymdp_stepper(cfg: BenchmarkConfig) -> Optional[Callable]:
    try:
        import jax.numpy as jnp  # noqa: F401
        from pymdp.agent import Agent  # type: ignore
    except Exception:
        return None

    n = cfg.grid_size
    n_states = n * n
    n_actions = 4

    A = np.full((n_states, n_states), 0.05 / max(n_states - 1, 1))
    np.fill_diagonal(A, 0.95)
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

    C = np.zeros(n_states)
    C[-1] = 4.0
    D = np.ones(n_states) / n_states

    def step(pos, goal, walls):
        try:
            agent = Agent(
                A=[A], B=[B], C=[C], D=[D],
                inference_algo="fpi", inference_horizon=1,
            )
            qs = np.zeros((1, n_states))
            qs[0, pos[0] * n + pos[1]] = 1.0
            q_pi, _ = agent.infer_policies(qs)
            action = int(np.argmax(q_pi))
            efe = float(-np.max(q_pi))
        except Exception:
            action, efe = 0, 0.0

        i, j = pos
        if action == 0:
            cand = ((i + 1) % n, j)
        elif action == 1:
            cand = ((i - 1) % n, j)
        elif action == 2:
            cand = (i, (j + 1) % n)
        else:
            cand = (i, (j - 1) % n)
        if cand in walls:
            cand = pos
        return cand, efe

    return step


def run_benchmark(cfg: BenchmarkConfig) -> list[BenchmarkReport]:
    rng = np.random.default_rng(cfg.seed)
    worlds = [make_world(cfg, rng) for _ in range(cfg.n_episodes)]
    reports = []

    tatha = BenchmarkReport(agent_name="tatha-efe")
    step_t = make_tatha_stepper(cfg)
    for s, g, w in worlds:
        tatha.episodes.append(run_episode(step_t, s, g, w, cfg))
    reports.append(tatha)

    step_p = make_pymdp_stepper(cfg)
    if step_p is not None:
        pymdp = BenchmarkReport(agent_name="pymdp-fpi")
        for s, g, w in worlds:
            pymdp.episodes.append(run_episode(step_p, s, g, w, cfg))
        reports.append(pymdp)

    return reports


def print_reports(reports: list[BenchmarkReport]) -> None:
    header = f"{'agent':<14} {'eps':>5} {'succ':>7} {'steps':>7} {'efe':>10} {'time':>10}"
    print(header)
    print("-" * len(header))
    for r in reports:
        s = r.summary()
        print(
            f"{s['agent']:<14} {int(s['episodes']):>5} "
            f"{s['success_rate']:>7.3f} {s['mean_steps']:>7.2f} "
            f"{s['mean_efe']:>10.3f} {s['mean_time_s']:>10.4f}"
        )


# ============================================================================
# SECTION 7: CLI ENTRYPOINT
# ============================================================================

def _demo_quantum_stack() -> None:
    print("=== Path Integral Planner ===")
    planner = PathIntegralPlanner(PathIntegralConfig(grid_size=16, n_slices=8))
    V = np.zeros((16, 16))
    V[8, 8] = -5.0
    psi = planner.transition_amplitudes((0, 0), V)
    psi = planner.normalize(psi)
    pref = np.zeros((16, 16))
    pref[15, 15] = 1.0
    efe = planner.expected_free_energy(psi, pref, V)
    print(f"EFE at goal-preference: {efe:.6f}")
    print(f"|psi|^2 sum (should be 1.0): {np.sum(np.abs(psi)**2):.6f}")

    print("\n=== Precision Hyper-Model ===")
    hm = PrecisionHyperModel(HyperModelConfig(state_dim=32, n_channels=4))
    for _ in range(5):
        state = torch.randn(8, 32)
        errors = torch.randn(8, 4)
        info = hm.step(state, errors)
    print(f"last free_energy={info['free_energy']:.4f} "
          f"mean_precision={info['mean_precision']:.4f}")

    print("\n=== QEPSO Swarm ===")
    bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])

    def rastrigin(x: np.ndarray) -> float:
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    swarm = QEPSOSwarm(QEPSOConfig(n_particles=32, max_iter=100), bounds, rastrigin)
    best_x, best_f = swarm.optimize()
    print(f"best_x={best_x}, best_f={best_f:.6f}")
    density = swarm.wavefunction_density(16)
    print(f"wavefunction_density shape={density.shape} sum={density.sum():.6f}")

    print("\n=== Density Matrix Swarm ===")
    dm = DensityMatrixSwarm(DensitySwarmConfig(hilbert_dim=8, dt=0.02))
    dm.evolve(20)
    print(f"populations sum={dm.populations().sum():.6f}")
    print(f"coherence={dm.coherence():.6f}")
    print(f"entanglement_entropy(sub=2)={dm.entanglement_entropy(2):.6f}")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage:")
        print("  python tatha_all.py demo")
        print("  python tatha_all.py bench")
        print("  python tatha_all.py agent '<task>'")
        return 0

    mode = argv[1]
    if mode == "demo":
        _demo_quantum_stack()
    elif mode == "bench":
        print_reports(run_benchmark(BenchmarkConfig()))
    elif mode == "agent":
        if len(argv) < 3:
            print("usage: python tatha_all.py agent '<task>'")
            return 1
        result = TathaAgent().run(" ".join(argv[2:]))
        print(result.model_dump_json(indent=2))
    else:
        print(f"unknown mode: {mode}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
