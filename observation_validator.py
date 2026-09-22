from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class ValidationResult:
    passed: bool
    issues: list = field(default_factory=list)
    def __bool__(self) -> bool:
        return self.passed
    def raise_if_failed(self) -> None:
        if not self.passed:
            raise ValueError("; ".join(self.issues))


@dataclass
class ObservationSpec:
    n_obs: Optional[int] = None
    n_dims: Optional[int] = None
    kind: str = "discrete"
    low: Optional[float] = None
    high: Optional[float] = None
    allow_nan: bool = False
    allow_inf: bool = False
    min_entropy: Optional[float] = None
    max_entropy: Optional[float] = None


def _check_finite(obs, spec, issues):
    if not spec.allow_nan and np.any(np.isnan(obs)):
        issues.append("contains NaN")
    if not spec.allow_inf and np.any(np.isinf(obs)):
        issues.append("contains Inf")


def _check_shape(obs, spec, issues):
    if spec.n_obs is not None and obs.ndim > 1 and obs.shape[1] != spec.n_obs:
        issues.append(f"observation size {obs.shape[1]} != expected {spec.n_obs}")
    if spec.n_dims is not None and obs.ndim != spec.n_dims:
        issues.append(f"ndim {obs.ndim} != expected {spec.n_dims}")


def _check_range(obs, spec, issues):
    if spec.low is not None and np.any(obs < spec.low):
        issues.append(f"values below {spec.low}")
    if spec.high is not None and np.any(obs > spec.high):
        issues.append(f"values above {spec.high}")


def _check_discrete(obs, spec, issues):
    if not np.allclose(obs, np.round(obs)):
        issues.append("non-integer values in discrete observation")
    if spec.n_obs is not None:
        if np.any(obs < 0) or np.any(obs >= spec.n_obs):
            issues.append(f"index out of range [0, {spec.n_obs})")


def _check_probability(obs, spec, issues):
    if obs.ndim != 1:
        issues.append("probability observation must be 1-D")
        return
    if np.any(obs < 0):
        issues.append("negative probability")
    s = obs.sum()
    if not np.isclose(s, 1.0, atol=1e-4):
        issues.append(f"probabilities sum to {s:.6f}, expected 1.0")
    if np.all(obs == 0):
        issues.append("degenerate all-zero distribution")


def _check_entropy(obs, spec, issues):
    if spec.min_entropy is None and spec.max_entropy is None:
        return
    if obs.ndim != 1:
        return
    p = np.clip(obs, 1e-12, None)
    p = p / p.sum()
    h = float(-np.sum(p * np.log(p)))
    if spec.min_entropy is not None and h < spec.min_entropy:
        issues.append(f"entropy {h:.4f} < min {spec.min_entropy}")
    if spec.max_entropy is not None and h > spec.max_entropy:
        issues.append(f"entropy {h:.4f} > max {spec.max_entropy}")


def validate_observation(obs: Any, spec: ObservationSpec, strict: bool = False) -> ValidationResult:
    issues: list[str] = []
    try:
        arr = np.asarray(obs, dtype=float)
    except (TypeError, ValueError) as exc:
        return ValidationResult(False, [f"cannot coerce to array: {exc}"])
    if arr.size == 0:
        return ValidationResult(False, ["empty observation"])
    _check_finite(arr, spec, issues)
    if issues:
        return ValidationResult(False, issues)
    _check_shape(arr, spec, issues)
    _check_range(arr, spec, issues)
    if spec.kind == "discrete":
        _check_discrete(arr, spec, issues)
    elif spec.kind == "probability":
        _check_probability(arr, spec, issues)
        _check_entropy(arr, spec, issues)
    result = ValidationResult(len(issues) == 0, issues)
    if strict and not result.passed:
        result.raise_if_failed()
    return result


def validate_batch(obs_batch: Any, spec: ObservationSpec, strict: bool = False) -> ValidationResult:
    arr = np.asarray(obs_batch)
    if arr.ndim == 0:
        return ValidationResult(False, ["batch is scalar"])
    all_issues = []
    for i in range(arr.shape[0]):
        r = validate_observation(arr[i], spec)
        if not r.passed:
            all_issues.append(f"row {i}: {'; '.join(r.issues)}")
    result = ValidationResult(len(all_issues) == 0, all_issues)
    if strict and not result.passed:
        result.raise_if_failed()
    return result


def grid_spec(n: int = 4) -> ObservationSpec:
    return ObservationSpec(n_obs=n * n, n_dims=1, kind="discrete", low=0, high=n * n - 1)


def probability_spec(n: int) -> ObservationSpec:
    return ObservationSpec(n_obs=n, n_dims=1, kind="probability", low=0.0, high=1.0, min_entropy=1e-6)


def continuous_spec(n_dims: int, low: float = -np.inf, high: float = np.inf) -> ObservationSpec:
    return ObservationSpec(n_dims=n_dims, kind="continuous", low=low, high=high)
