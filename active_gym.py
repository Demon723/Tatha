"""Gymnasium adapter for active inference agents."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional

try:
    import gymnasium as gym
    _GYM_AVAILABLE = True
except ImportError:
    gym = None
    _GYM_AVAILABLE = False


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

    def __init__(self, env: Any, discretizer: Optional[Discretizer] = None,
                 normalize_observations: bool = True,
                 uncertainty_penalty: float = 0.0, seed: Optional[int] = None):
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
