from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class ValidationResult:
    passed: bool
    issues: list = field(default_factory=list)
    def __bool__(self):
        return self.passed
    def raise_if_failed(self):
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


def _finite(arr, spec, issues):
    if not spec.allow_nan and np.any(np.isnan(arr)):
        issues.append("contains NaN")
    if not spec.allow_inf and np.any(np.isinf(arr)):
        issues.append("contains Inf")


def _shape(arr, spec, issues):
    if spec.n_obs is not None and arr.ndim == 1 and arr.size != spec.n_obs and arr.size > 1:
        issues.append(f"size {arr.size} != {spec.n_obs}")
    if spec.n_dims is not None and arr.ndim != spec.n_dims:
        issues.append(f"ndim {arr.ndim} != {spec.n_dims}")


def _range(arr, spec, issues):
    if spec.low is not None and np.any(arr < spec.low):
        issues.append(f"values < {spec.low}")
    if spec.high is not None and np.any(arr > spec.high):
        issues.append(f"values > {spec.high}")


def _discrete(arr, spec, issues):
    if not np.allclose(arr, np.round(arr)):
        issues.append("non-integer in discrete obs")
    if spec.n_obs is not None and (np.any(arr < 0) or np.any(arr >= spec.n_obs)):
        issues.append(f"out of range [0, {spec.n_obs}): {int(np.max(arr[np.nonzero((arr<0)|(arr>=spec.n_obs))]))}")


def _prob(arr, spec, issues):
    if arr.ndim != 1:
        issues.append("probability must be 1-D")
        return
    if np.any(arr < 0):
        issues.append("negative probability")
    s = arr.sum()
    if not np.isclose(s, 1.0, atol=1e-4):
        issues.append(f"sum {s:.6f} != 1.0")
    if np.all(arr == 0):
        issues.append("all-zero distribution")


def _entropy(arr, spec, issues):
    if spec.min_entropy is None and spec.max_entropy is None:
        return
    if arr.ndim != 1:
        return
    p = np.clip(arr, 1e-12, None)
    p = p / p.sum()
    h = float(-np.sum(p * np.log(p)))
    if spec.min_entropy is not None and h < spec.min_entropy:
        issues.append(f"H={h:.4f} < {spec.min_entropy}")
    if spec.max_entropy is not None and h > spec.max_entropy:
        issues.append(f"H={h:.4f} > {spec.max_entropy}")


def validate_observation(obs: Any, spec: ObservationSpec,
                         strict: bool = False) -> ValidationResult:
    issues = []
    try:
        arr = np.asarray(obs, dtype=float)
    except (TypeError, ValueError) as exc:
        return ValidationResult(False, [f"coerce failed: {exc}"])
    if arr.size == 0:
        return ValidationResult(False, ["empty observation"])
    _finite(arr, spec, issues)
    if issues:
        return ValidationResult(False, issues)
    _shape(arr, spec, issues)
    _range(arr, spec, issues)
    if spec.kind == "discrete":
        _discrete(arr, spec, issues)
    elif spec.kind == "probability":
        _prob(arr, spec, issues)
        _entropy(arr, spec, issues)
    result = ValidationResult(len(issues) == 0, issues)
    if strict and not result.passed:
        result.raise_if_failed()
    return result




def validate_batch(obs_batch: Any, spec: ObservationSpec,
                   strict: bool = False) -> ValidationResult:
    arr = np.asarray(obs_batch)
    if arr.ndim == 0:
        return ValidationResult(False, ['batch is scalar'])
    all_issues = []
    for i in range(arr.shape[0]):
        r = validate_observation(arr[i], spec)
        if not r.passed:
            all_issues.append(f"row {i}: {"; ".join(r.issues)}")
    result = ValidationResult(len(all_issues) == 0, all_issues)
    if strict and not result.passed:
        result.raise_if_failed()
    return result


def grid_spec(n: int = 4) -> ObservationSpec:
    return ObservationSpec(n_obs=n * n, n_dims=1, kind="discrete",
                           low=0, high=n * n - 1)


def probability_spec(n: int) -> ObservationSpec:
    return ObservationSpec(n_obs=n, n_dims=1, kind="probability",
                           low=0.0, high=1.0, min_entropy=1e-6)


def continuous_spec(n_dims: int, low=-np.inf, high=np.inf) -> ObservationSpec:
    return ObservationSpec(n_dims=n_dims, kind="continuous",
                           low=low, high=high)


if __name__ == "__main__":
    spec = grid_spec(4)
    assert validate_observation(np.array([5]), spec).passed
    assert not validate_observation(np.array([99]), spec).passed
    assert not validate_observation(np.array([3.5]), spec).passed
    pspec = probability_spec(4)
    assert validate_observation(np.array([0.25] * 4), pspec).passed
    assert not validate_observation(np.array([0.5, 0.5, 0.5, 0.5]), pspec).passed
    try:
        validate_observation(np.array([99]), spec, strict=True)
        raise AssertionError("strict mode did not raise")
    except ValueError:
        pass
    print("observation_validator.py: OK")
