"""
tatha_utils.py
===============
Utility infrastructure for Tatha real-time agent system.

Provides:
  - Logging configuration
  - Config loading (YAML/env)
  - GPU backend abstraction
  - Error recovery
  - Validation
  - Metrics tracking
"""
from __future__ import annotations
import os
import sys
import json
import logging
import warnings
import statistics
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

def setup_logging(name: str = "tatha", level: str = "INFO",
                     log_file: str | None = None) -> logging.Logger:
    """Configure structured logging."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        '{"time": "%(asctime)s", "level": "%(levelname)s", '
        '"module": "%(name)s", "message": %(message)s}'
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

@dataclass
class TathaConfig:
    """Runtime configuration with env-var and YAML support."""

    GRID_SIZE: int = 10
    MAX_BRANCHES: int = 8
    HIDDEN_SIZE: int = 32
    BELIEF_SIZE: int = 8
    LEARNING_RATE: float = 0.005
    EXPLORATION_COEF: float = 0.2
    DECOHERENCE_RATE: float = 0.01
    DT: float = 0.1
    SETTLE_ITERS: int = 10
    REPLAY_CAPACITY: int = 10000
    BATCH_SIZE: int = 32
    CHECKPOINT_DIR: str = "./checkpoints"
    WS_HOST: str = "0.0.0.0"
    WS_PORT: int = 8765
    REST_PORT: int = 8766
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str | None = None
    USE_GPU: bool = False
    CUDA_DEVICE: int = 0
    PRIORITIZED_REPLAY: bool = False
    ALPHA: float = 0.6  # Prioritization strength
    BETA_START: float = 0.4  # Importance sampling
    CURRICULUM: bool = False
    MAX_EPISODES: int = 100
    ATTENTION_ENABLED: bool = False
    ATTENTION_HEADS: int = 4
    ATTENTION_DIM: int = 8
    REWARD_SHAPING: bool = False
    NEUROSCIENCE_METRICS: bool = False

    @classmethod
    def from_env(cls) -> "TathaConfig":
        """Load config from environment variables."""
        config = cls()
        env_map = {
            "TATHA_GRID_SIZE": "GRID_SIZE",
            "TATHA_MAX_BRANCHES": "MAX_BRANCHES",
            "TATHA_HIDDEN_SIZE": "HIDDEN_SIZE",
            "TATHA_BELIEF_SIZE": "BELIEF_SIZE",
            "TATHA_LEARNING_RATE": "LEARNING_RATE",
            "TATHA_EXPLORATION_COEF": "EXPLORATION_COEF",
            "TATHA_DECOHERENCE_RATE": "DECOHERENCE_RATE",
            "TATHA_DT": "DT",
            "TATHA_REPLAY_CAPACITY": "REPLAY_CAPACITY",
            "TATHA_BATCH_SIZE": "BATCH_SIZE",
            "TATHA_WS_PORT": "WS_PORT",
            "TATHA_REST_PORT": "REST_PORT",
            "TATHA_LOG_LEVEL": "LOG_LEVEL",
            "TATHA_USE_GPU": "USE_GPU",
            "TATHA_PRIORITIZED_REPLAY": "PRIORITIZED_REPLAY",
            "TATHA_CURRICULUM": "CURRICULUM",
            "TATHA_ATTENTION_ENABLED": "ATTENTION_ENABLED",
            "TATHA_REWARD_SHAPING": "REWARD_SHAPING",
            "TATHA_NEUROSCIENCE_METRICS": "NEUROSCIENCE_METRICS",
        }
        for env_key, attr in env_map.items():
            val = os.environ.get(env_key)
            if val is not None:
                try:
                    if attr in ("USE_GPU", "PRIORITIZED_REPLAY", "CURRICULUM",
                                "ATTENTION_ENABLED", "REWARD_SHAPING",
                                "NEUROSCIENCE_METRICS"):
                        setattr(config, attr, val.lower() in ("true", "1", "yes"))
                    elif attr in ("GRID_SIZE", "MAX_BRANCHES", "HIDDEN_SIZE",
                                  "BELIEF_SIZE", "SETTLE_ITERS", "REPLAY_CAPACITY",
                                  "BATCH_SIZE", "MAX_EPISODES", "CUDA_DEVICE",
                                  "ATTENTION_HEADS", "ATTENTION_DIM"):
                        setattr(config, attr, int(val))
                    elif attr in ("LEARNING_RATE", "EXPLORATION_COEF",
                                  "DECOHERENCE_RATE", "DT", "ALPHA", "BETA_START"):
                        setattr(config, attr, float(val))
                    elif attr in ("LOG_LEVEL", "LOG_FILE", "CHECKPOINT_DIR",
                                  "WS_HOST", "REST_PORT"):
                        setattr(config, attr, val)
                except (ValueError, TypeError):
                    pass
        return config

    @classmethod
    def from_yaml(cls, path: str) -> "TathaConfig":
        """Load config from YAML file."""
        try:
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            config = cls()
            for key, val in data.items():
                if hasattr(config, key):
                    setattr(config, key, val)
            return config
        except ImportError:
            return cls.from_env()

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def save(self, path: str) -> None:
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TathaConfig":
        import json
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


# ---------------------------------------------------------------------------
# GPU BACKEND ABSTRACTION
# ---------------------------------------------------------------------------

class Backend:
    """Abstract numpy/cupy backend for GPU acceleration."""
    def __init__(self, use_gpu: bool = False, device: int = 0):
        self.use_gpu = use_gpu and self._gpu_available()
        self.device = device
        self._xp = np  # default: numpy

    @staticmethod
    def _gpu_available() -> bool:
        try:
            import cupy
            return True
        except ImportError:
            return False

    @property
    def xp(self):
        if self.use_gpu:
            try:
                import cupy as cp
                return cp
            except ImportError:
                pass
        return np

    def array(self, *args, **kwargs):
        return self.xp.array(*args, **kwargs)

    def asarray(self, *args, **kwargs):
        return self.xp.asarray(*args, **kwargs)

    def matmul(self, a, b):
        return self.xp.dot(a, b)

    def expm(self, a):
        if self.use_gpu:
            from cupyx.scipy.linalg import expm
            return expm(a)
        from scipy.linalg import expm as _expm
        return _expm(a)

    def zeros(self, shape, dtype=float):
        return self.xp.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=float):
        return self.xp.ones(shape, dtype=dtype)

    def random_normal(self, shape, dtype=float):
        if self.use_gpu:
            return self.xp.random.standard_normal(shape, dtype=dtype)
        return np.random.standard_normal(shape).astype(dtype)


def get_backend(use_gpu: bool = False) -> Backend:
    """Get the computation backend."""
    return Backend(use_gpu=use_gpu)


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------

class ObservationValidator:
    """Validates observations before agent processing."""
    def __init__(self, expected_size: int, min_val: float = -1e6,
                 max_val: float = 1e6, allow_nan: bool = False):
        self.expected_size = expected_size
        self.min_val = min_val
        self.max_val = max_val
        self.allow_nan = allow_nan
        self._violations = 0

    def validate(self, obs: list | np.ndarray) -> tuple[bool, str | None]:
        """Validate observation. Returns (is_valid, error_message)."""
        if obs is None:
            return False, "Observation is None"

        arr = np.asarray(obs)

        # Check size
        if arr.size < self.expected_size:
            return False, f"Obs size {arr.size} < expected {self.expected_size}"

        # Check bounds
        if np.nanmax(arr) > self.max_val:
            return False, f"Obs max {np.nanmax(arr)} > {self.max_val}"
        if np.nanmin(arr) < self.min_val:
            return False, f"Obs min {np.nanmin(arr)} < {self.min_val}"

        # Check NaN
        if not self.allow_nan and np.any(np.isnan(arr)):
            return False, "Observation contains NaN"

        # Check inf
        if np.any(np.isinf(arr)):
            return False, "Observation contains Inf"

        return True, None

    def record_violation(self):
        self._violations += 1

    @property
    def violations(self):
        return self._violations


# ---------------------------------------------------------------------------
# ERROR RECOVERY
# ---------------------------------------------------------------------------

class ErrorRecovery:
    """Manages error recovery for agent operations."""
    def __init__(self, max_retries: int = 3, reset_on_failure: bool = True):
        self.max_retries = max_retries
        self.reset_on_failure = reset_on_failure
        self._retry_count = 0
        self._errors: list[dict] = []

    def execute_with_recovery(self, func, *args, **kwargs):
        """Execute function with automatic recovery."""
        for attempt in range(self.max_retries):
            try:
                result = func(*args, **kwargs)
                self._retry_count = 0  # reset on success
                return result
            except Exception as e:
                self._errors.append({
                    'attempt': attempt,
                    'error': str(e),
                    'type': type(e).__name__,
                })
                if attempt < self.max_retries - 1:
                    if self.reset_on_failure:
                        self._soft_reset()
                    continue
                raise RuntimeError(f"Max retries ({self.max_retries}) exceeded")

    def _soft_reset(self):
        """Soft reset: recover without losing state."""
        pass  # Override in subclass

    def get_error_history(self) -> list[dict]:
        return list(self._errors)

    @property
    def healthy(self) -> bool:
        return self._retry_count < self.max_retries


# ---------------------------------------------------------------------------
# NEUROSCIENCE METRICS
# ---------------------------------------------------------------------------

class NeuroscienceMetrics:
    """Track computational neuroscience metrics."""
    def __init__(self):
        self.free_energy_history: list[float] = []
        self.precision_history: list[float] = []
        self.prediction_error_history: list[float] = []
        self.complexity_history: list[float] = []
        self.active_information: list[float] = []

    def record_free_energy(self, fe: float):
        self.free_energy_history.append(fe)

    def record_precision(self, precision: float):
        self.precision_history.append(precision)

    def record_prediction_error(self, pe: float):
        self.prediction_error_history.append(pe)

    def record_complexity(self, c: float):
        self.complexity_history.append(c)

    def record_active_information(self, ai: float):
        self.active_information.append(ai)

    def summary(self) -> dict:
        return {
            'mean_free_energy': float(np.mean(self.free_energy_history[-100:])) if self.free_energy_history else 0.0,
            'mean_prediction_error': float(np.mean(self.prediction_error_history[-100:])) if self.prediction_error_history else 0.0,
            'mean_precision': float(np.mean(self.precision_history[-100:])) if self.precision_history else 0.0,
            'total_steps': len(self.free_energy_history),
        }


# ---------------------------------------------------------------------------
# BENCHMARK UTILITY
# ---------------------------------------------------------------------------

class BenchmarkSuite:
    """Standardized benchmarks across all environments."""

    @staticmethod
    def measure_steps_to_completion(env) -> int:
        """Measure steps to reach goal."""
        return 0  # Override per environment

    @staticmethod
    def measure_reward_per_episode(env) -> float:
        """Measure average reward per episode."""
        return 0.0

    @staticmethod
    def measure_convergence(env) -> bool:
        """Check if agent has converged."""
        return False

    @staticmethod
    def measure_generalization(env, test_envs: list) -> float:
        """Measure performance on unseen environments."""
        return 0.0


# --- New benchmark types from tatha_all.py ---

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
