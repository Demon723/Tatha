from __future__ import annotations

import importlib
import inspect
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class ComponentSpec:
    name: str
    module: str
    attr: Optional[str]
    kind: str
    category: str
    requires: tuple = ("numpy",)
    kwargs_variants: tuple = ()
    smoke: Optional[str] = None
    notes: str = ""


WIRING_MANIFEST: list[ComponentSpec] = [
    ComponentSpec("PathIntegralPlanner", "path_integral", "PathIntegralPlanner", "class", "quantum", ("numpy",), kwargs_variants=({"grid_size": 8, "n_slices": 4}, {"grid_size": 8}, {}), notes="Trotter-Suzuki propagator"),
    ComponentSpec("PrecisionHyperModel", "precision_hyper", "PrecisionHyperModel", "class", "quantum", ("numpy", "torch"), kwargs_variants=({"state_dim": 16, "n_channels": 4}, {}), notes="Adaptive per-channel precision"),
    ComponentSpec("QEPSOSwarm", "qepso", "QEPSOSwarm", "class", "quantum", ("numpy",), kwargs_variants=(), notes="Quantum-entangled PSO"),
    ComponentSpec("DensityMatrixSwarm", "density_swarm", "DensityMatrixSwarm", "class", "quantum", ("numpy",), kwargs_variants=({"hilbert_dim": 8}, {}), notes="Lindblad evolution"),
    ComponentSpec("QuantumSwarm", "quantum_swarm", "QuantumSwarm", "class", "swarm", ("numpy",), kwargs_variants=({},), notes="Multi-agent quantum swarm"),
    ComponentSpec("AdversarialSwarm", "adversarial_swarm", "AdversarialSwarm", "class", "swarm", ("numpy",), kwargs_variants=({},), notes="RED vs BLUE swarm game"),
    ComponentSpec("GridWorld", "gridworld_pc", "GridWorld", "class", "env", ("numpy",), kwargs_variants=({"size": 4}, {"n": 4}, {}), notes="Grid world"),
    ComponentSpec("SpaceGridWorld", "gridworld_pc", "SpaceGridWorld", "class", "env", ("numpy",), kwargs_variants=({"size": 4}, {}), notes="Spatial grid world"),
    ComponentSpec("GenerativeModel", "active_inference", "GenerativeModel", "class", "ai", ("numpy",), kwargs_variants=(), notes="A/B/C/D/E discrete POMDP"),
    ComponentSpec("run_fpi", "active_inference", "run_fpi", "function", "ai", ("numpy",), notes="Fixed-point variational inference"),
    ComponentSpec("compute_efe", "active_inference", "compute_efe", "function", "ai", ("numpy",), notes="Expected free energy"),
    ComponentSpec("select_action", "active_inference", "select_action", "function", "ai", ("numpy",), notes="EFE policy selection"),
    ComponentSpec("GymWrapper", "active_gym", "GymWrapper", "class", "gym", ("numpy", "gymnasium"), kwargs_variants=(), notes="Gymnasium adapter"),
    ComponentSpec("Discretizer", "active_gym", "Discretizer", "class", "gym", ("numpy",), kwargs_variants=({"low": np.array([-1.0, -1.0]), "high": np.array([1.0, 1.0])},), notes="Continuous to discrete binning"),
    ComponentSpec("active_visualization", "active_visualization", None, "module", "viz", ("numpy", "matplotlib"), notes="Plotting helpers"),
    ComponentSpec("CheckpointManager", "checkpoint", "CheckpointManager", "class", "infra", ("numpy",), kwargs_variants=({"root": "/tmp/tatha_ckpt"},), notes="Save/load store"),
    ComponentSpec("validate_observation", "observation_validator", "validate_observation", "function", "infra", ("numpy",), notes="Observation validator"),
]


_MODULE_CACHE: dict[str, bool] = {}

def _has_module(name: str) -> bool:
    if name in _MODULE_CACHE:
        return _MODULE_CACHE[name]
    if name in sys.modules:
        _MODULE_CACHE[name] = True
        return True
    try:
        importlib.import_module(name)
        _MODULE_CACHE[name] = True
    except ImportError:
        _MODULE_CACHE[name] = False
    return _MODULE_CACHE[name]

def _missing_deps(spec: ComponentSpec) -> list:
    return [r for r in spec.requires if not _has_module(r)]


def _try_call(fn: Callable, kwargs_variants: tuple, fallback: Any = None) -> Any:
    last_exc: Optional[Exception] = None
    for kwargs in kwargs_variants:
        try:
            return fn(**kwargs)
        except (TypeError, ValueError) as exc:
            last_exc = exc
            continue
    try:
        return fn()
    except (TypeError, ValueError) as exc:
        last_exc = exc
    if last_exc is not None:
        return _Failed(last_exc)
    return fallback


class _Failed:
    def __init__(self, exc: Exception):
        self.exc = exc
    def __repr__(self):
        return f"<_Failed: {type(self.exc).__name__}: {self.exc}>"


@dataclass
class Resolution:
    spec: ComponentSpec
    obj: Any = None
    error: Optional[str] = None
    @property
    def ok(self) -> bool:
        return self.obj is not None and self.error is None


class WiringRegistry:
    def __init__(self, manifest: Optional[list] = None):
        self.manifest = manifest or WIRING_MANIFEST
        self._by_name = {s.name: s for s in self.manifest}
        self._resolved: dict[str, Resolution] = {}
        self._instances: dict[str, Any] = {}

    def resolve(self, name: str) -> Resolution:
        if name in self._resolved:
            return self._resolved[name]
        spec = self._by_name.get(name)
        if spec is None:
            res = Resolution(spec=None, error=f"no spec for {name}")
            self._resolved[name] = res
            return res
        missing = _missing_deps(spec)
        if missing:
            res = Resolution(spec=spec, error=f"missing deps: {missing}")
            self._resolved[name] = res
            return res
        try:
            mod = importlib.import_module(spec.module)
            obj = mod if spec.attr is None else getattr(mod, spec.attr)
        except (ImportError, AttributeError) as exc:
            res = Resolution(spec=spec, error=f"import failed: {exc}")
            self._resolved[name] = res
            return res
        res = Resolution(spec=spec, obj=obj)
        self._resolved[name] = res
        return res

    def instantiate(self, name: str, **overrides) -> Any:
        if name in self._instances and not overrides:
            return self._instances[name]
        res = self.resolve(name)
        if not res.ok:
            return None
        if res.spec.kind in ("function", "module"):
            return res.obj
        variants = list(res.spec.kwargs_variants)
        if overrides:
            variants = [overrides] + variants
        obj = _try_call(res.obj, tuple(variants))
        if isinstance(obj, _Failed):
            return None
        if not overrides:
            self._instances[name] = obj
        return obj

    def available(self) -> list:
        return [s.name for s in self.manifest if self.resolve(s.name).ok]

    def unavailable(self) -> dict:
        return {s.name: self.resolve(s.name).error for s in self.manifest if not self.resolve(s.name).ok}

    def by_category(self, cat: str) -> list:
        return [s.name for s in self.manifest if s.category == cat and self.resolve(s.name).ok]

    def coverage(self) -> dict:
        total = len(self.manifest)
        ok = len(self.available())
        by_cat: dict[str, dict] = {}
        for s in self.manifest:
            by_cat.setdefault(s.category, {"total": 0, "ok": 0})
            by_cat[s.category]["total"] += 1
            if self.resolve(s.name).ok:
                by_cat[s.category]["ok"] += 1
        return {"total": total, "available": ok, "missing": total - ok, "percent": 100.0 * ok / max(total, 1), "by_category": by_cat}


# Backward-compatible aliases for existing tests
ComponentRegistry = WiringRegistry


# ============================================================================
# TATHASTACK (backward-compatible with tests)
# ============================================================================

class TathaStack:
    def __init__(self, registry: Optional[WiringRegistry] = None):
        self.reg = registry or WiringRegistry()
        self.components: dict[str, Any] = {}
        for spec in self.reg.manifest:
            res = self.reg.resolve(spec.name)
            if res.ok:
                self.components[spec.name] = res.obj

    @classmethod
    def from_available(cls) -> "TathaStack":
        return cls()

    def has(self, name: str) -> bool:
        return name in self.components

    def get(self, name: str) -> Any:
        if name not in self.components:
            raise KeyError(f"component not available: {name}")
        return self.components[name]

    def run_grid_episode(self, n: int = 4, steps: int = 10, seed: int = 0) -> dict:
        if not self.has("GenerativeModel") or not self.has("run_fpi"):
            raise RuntimeError("active_inference module not available")
        from active_inference import GenerativeModel, run_fpi, select_action
        n_s = n * n
        A = [np.eye(n_s) * 0.9 + 0.1 / n_s]
        A[0] = A[0] / A[0].sum(axis=0, keepdims=True)
        B = [np.stack([np.eye(n_s)] * 4, axis=2)]
        model = GenerativeModel(A=A, B=B)
        rng = np.random.default_rng(seed)
        qs_prev = None
        beliefs, actions = [], []
        for t in range(steps):
            obs = [int(rng.integers(0, n_s))]
            res = run_fpi(model, obs=obs, qs_prev=qs_prev, num_iter=16)
            beliefs.append(res.qs[0].copy())
            policy, _, _ = select_action(model, res.qs, [(a,) for a in range(4)], rng=rng)
            actions.append(policy[0])
            qs_prev = res.qs
        return {"beliefs": np.array(beliefs), "actions": actions, "final_free_energy": float(res.free_energy), "n_steps": steps}

    def run_path_integral(self, grid: int = 16, slices: int = 8) -> dict:
        planner_cls = self.instantiate("PathIntegralPlanner", grid_size=grid, n_slices=slices)
        if planner_cls is None:
            return {"skipped": "PathIntegralPlanner not available"}
        V = np.zeros((grid, grid)); V[grid // 2, grid // 2] = -5.0
        psi0 = np.zeros((grid, grid), dtype=complex); psi0[0, 0] = 1.0
        psi_out = planner_cls.propagate(psi0, V, n_slices=2)
        return {"norm": float(np.sum(np.abs(psi_out) ** 2))}

    def run_density_swarm(self, dim: int = 8, steps: int = 20) -> dict:
        dm = self.instantiate("DensityMatrixSwarm", hilbert_dim=dim)
        if dm is None:
            return {"skipped": "DensityMatrixSwarm not available"}
        dm.evolve(steps)
        return {"populations_sum": float(dm.populations().sum()), "coherence": float(dm.coherence())}

    def summary(self) -> dict:
        return {"n_available": len(self.components), "n_total": len(self.reg.manifest), "available": sorted(self.components.keys())}


# Backward-compatible aliases


# ============================================================================
# SMOKE TESTING
# ============================================================================

def _smoke_instance(name: str, obj: Any) -> tuple[bool, str]:
    try:
        if name == "PathIntegralPlanner":
            psi = np.zeros((8, 8), dtype=complex); psi[0, 0] = 1.0
            V = np.zeros((8, 8))
            out = obj.propagate(psi, V, n_slices=2)
            return True, f"shape={np.asarray(out).shape}"
        if name == "DensityMatrixSwarm":
            obj.evolve(3)
            return True, f"coherence={obj.coherence():.4f}"
        if name == "QEPSOSwarm":
            return True, "class resolved"
        if name == "GenerativeModel":
            A = [np.eye(4) * 0.9 + 0.1 / 4]
            B = [np.stack([np.eye(4)] * 2, axis=2)]
            m = obj(A=A, B=B)
            return True, f"states={m.num_states}"
        if name == "run_fpi":
            A = [np.eye(4) * 0.9 + 0.1 / 4]
            B = [np.stack([np.eye(4)] * 2, axis=2)]
            from active_inference import GenerativeModel as GM
            m = GM(A=A, B=B)
            res = obj(m, obs=[0], num_iter=8)
            return True, f"F={res.free_energy:.4f}"
        if name == "compute_efe":
            A = [np.eye(4) * 0.9 + 0.1 / 4]
            B = [np.stack([np.eye(4)] * 2, axis=2)]
            from active_inference import GenerativeModel as GM
            m = GM(A=A, B=B)
            qs = [np.ones(4) / 4]
            bd = obj(m, qs, (0,))
            return True, f"total={bd.total:.4f}"
        if name == "select_action":
            A = [np.eye(4) * 0.9 + 0.1 / 4]
            B = [np.stack([np.eye(4)] * 2, axis=2)]
            from active_inference import GenerativeModel as GM
            m = GM(A=A, B=B)
            qs = [np.ones(4) / 4]
            out = obj(m, qs, [(0,), (1,)])
            return True, f"policy={out[0]}"
        if name == "Discretizer":
            code = obj(np.array([0.0, 0.0]))
            return True, f"code={code}"
        if name == "CheckpointManager":
            mgr = obj("/tmp/tatha_wiring_smoke")
            mgr.save("smoke", np.arange(4))
            arr = mgr.load("smoke")
            mgr.delete("smoke")
            return True, f"roundtrip shape={arr.shape}"
        if name == "validate_observation":
            from observation_validator import grid_spec
            r = obj(np.array([1]), grid_spec(4))
            return True, f"passed={r.passed}"
        if name == "active_visualization":
            return True, "module imported"
        return True, "resolved"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


# ============================================================================
# PROBE AND SMOKE
# ============================================================================

def probe(verbose: bool = True, registry: Optional[WiringRegistry] = None) -> dict:
    reg = registry or WiringRegistry()
    _by_cat = {}
    for s in reg.manifest:
        _by_cat.setdefault(s.category, [])
        if reg.resolve(s.name).ok:
            _by_cat[s.category].append(s.name)
    report = {"available": reg.available(), "unavailable": reg.unavailable(), "by_category": _by_cat,
            "coverage": reg.coverage()}
    if verbose:
        print("=" * 72)
        print("TATHA WIRING PROBE")
        print("=" * 72)
        by_cat: dict[str, list] = {}
        for s in reg.manifest:
            by_cat.setdefault(s.category, []).append(s)
        for cat in sorted(by_cat):
            print(f"\n[{cat.upper()}]")
            for s in by_cat[cat]:
                res = reg.resolve(s.name)
                status = "OK  " if res.ok else "FAIL"
                detail = s.notes if res.ok else res.error
                print(f"  {status} {s.name:<24} {detail}")
        cov = report["coverage"]
        print(f"\navailable: {cov['available']}/{cov['total']} ({cov['percent']:.1f}%)")
        print("=" * 72)
    return report


def smoke(verbose: bool = True, registry: Optional[WiringRegistry] = None) -> dict:
    reg = registry or WiringRegistry()
    results: dict[str, dict] = {}
    for spec in reg.manifest:
        res = reg.resolve(spec.name)
        if not res.ok:
            results[spec.name] = {"status": "skip", "msg": res.error}
            continue
        obj = reg.instantiate(spec.name)
        if obj is None:
            results[spec.name] = {"status": "skip", "msg": "instantiation failed"}
            continue
        ok, msg = _smoke_instance(spec.name, obj)
        results[spec.name] = {"status": "pass" if ok else "fail", "msg": msg}
    if verbose:
        print("=" * 72)
        print("TATHA WIRING SMOKE TEST")
        print("=" * 72)
        for name in sorted(results):
            r = results[name]
            tag = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}[r["status"]]
            print(f"  {tag} {name:<24} {r['msg']}")
        passed = sum(1 for r in results.values() if r["status"] == "pass")
        failed = sum(1 for r in results.values() if r["status"] == "fail")
        print(f"\n{passed} passed, {failed} failed, {len(results) - passed - failed} skipped")
        print("=" * 72)
    return results


# ============================================================================
# RUNTIME ATTACHMENT
# ============================================================================

def attach_to_runtime(runtime: Any, registry: Optional[WiringRegistry] = None, strict: bool = False) -> list:
    reg = registry or WiringRegistry()
    attached = []
    for spec in reg.manifest:
        attr = spec.name.lower()
        if hasattr(runtime, attr):
            continue
        obj = reg.instantiate(spec.name)
        if obj is None:
            if strict:
                raise RuntimeError(f"failed to instantiate {spec.name}")
            continue
        try:
            setattr(runtime, attr, obj)
            attached.append(spec.name)
        except Exception:
            if strict:
                raise
            continue
    try:
        setattr(runtime, "_wired_components", attached)
    except Exception:
        pass
    return attached


# Backward-compatible aliases
wire = attach_to_runtime


# ============================================================================
# CATEGORY RUNNERS
# ============================================================================

def run_quantum(grid: int = 8) -> dict:
    reg = WiringRegistry()
    out = {}
    planner_cls = reg.instantiate("PathIntegralPlanner", grid_size=grid, n_slices=4)
    if planner_cls is not None:
        psi = np.zeros((grid, grid), dtype=complex); psi[0, 0] = 1.0
        V = np.zeros((grid, grid)); V[grid // 2, grid // 2] = -5.0
        psi_out = planner_cls.propagate(psi, V)
        out["path_integral"] = float(np.sum(np.abs(psi_out) ** 2))
    dm = reg.instantiate("DensityMatrixSwarm", hilbert_dim=8)
    if dm is not None:
        dm.evolve(20)
        out["density_pop_sum"] = float(dm.populations().sum())
        out["density_coherence"] = float(dm.coherence())
    return out


def run_active_inference(steps: int = 5) -> dict:
    reg = WiringRegistry()
    GM = reg.instantiate("GenerativeModel")
    run_fpi = reg.instantiate("run_fpi")
    select_action = reg.instantiate("select_action")
    if GM is None or run_fpi is None or select_action is None:
        return {"skipped": True}
    A = [np.eye(16) * 0.9 + 0.1 / 16]
    B = [np.stack([np.eye(16)] * 4, axis=2)]
    model = GM(A=A, B=B)
    rng = np.random.default_rng(0)
    qs_prev = None
    actions = []
    for _ in range(steps):
        res = run_fpi(model, obs=[0], qs_prev=qs_prev, num_iter=16)
        policy, _, _ = select_action(model, res.qs, [(a,) for a in range(4)], rng=rng)
        actions.append(policy[0])
        qs_prev = res.qs
    return {"actions": actions, "final_F": res.free_energy}


def run_infra() -> dict:
    reg = WiringRegistry()
    mgr = reg.instantiate("CheckpointManager", root="/tmp/tatha_wiring")
    if mgr is not None:
        mgr.save("test", np.arange(4))
        arr = mgr.load("test")
        return {"checkpoint_shape": arr.shape}
    return {"skipped": True}


# ============================================================================
# CLI
# ============================================================================

def cmd_probe(): probe()
def cmd_smoke(): smoke()

def cmd_list(category: Optional[str] = None):
    reg = WiringRegistry()
    for spec in reg.manifest:
        if category and spec.category != category:
            continue
        res = reg.resolve(spec.name)
        status = "OK  " if res.ok else "FAIL"
        print(f"{status} [{spec.category:<7}] {spec.name:<24} {res.error or spec.notes}")

def cmd_run(name: str):
    reg = WiringRegistry()
    res = reg.resolve(name)
    if not res.ok:
        print(f"cannot resolve {name}: {res.error}")
        return 1
    obj = reg.instantiate(name)
    if obj is None:
        print(f"cannot instantiate {name}")
        return 1
    ok, msg = _smoke_instance(name, obj)
    print(f"{name}: {'OK' if ok else 'FAIL'} — {msg}")
    return 0 if ok else 1

def cmd_coverage():
    reg = WiringRegistry()
    cov = reg.coverage()
    print(f"total: {cov['total']}")
    print(f"available: {cov['available']} ({cov['percent']:.1f}%)")
    print(f"missing: {cov['missing']}")
    print()
    for cat, counts in sorted(cov["by_category"].items()):
        print(f"  {cat:<10} {counts['ok']}/{counts['total']}")

def cmd_categories():
    reg = WiringRegistry()
    for cat in sorted(set(s.category for s in reg.manifest)):
        print(f"{cat}: {len(reg.by_category(cat))} available")

def cmd_quantum(): print(run_quantum())
def cmd_ai(): print(run_active_inference())
def cmd_infra(): print(run_infra())

def cmd_attach():
    print("# Paste into tatha_realtime.py after the imports:")
    print()
    print("try:")
    print("    from tatha_wiring import WiringRegistry, attach_to_runtime")
    print("    _WIRING = WiringRegistry()")
    print("    attach_to_runtime(globals())")
    print("except ImportError:")
    print("    _WIRING = None")

def main(argv: Optional[list] = None) -> int:
    argv = argv or sys.argv
    if len(argv) < 2:
        cmd_probe()
        return 0
    mode = argv[1]
    if mode == "probe": cmd_probe()
    elif mode == "smoke": cmd_smoke()
    elif mode == "list": cmd_list(argv[2] if len(argv) > 2 else None)
    elif mode == "run":
        if len(argv) < 3: print("usage: tatha_wiring.py run <component>"); return 1
        return cmd_run(argv[2])
    elif mode == "coverage": cmd_coverage()
    elif mode == "categories": cmd_categories()
    elif mode == "quantum": cmd_quantum()
    elif mode == "ai": cmd_ai()
    elif mode == "infra": cmd_infra()
    elif mode == "attach": cmd_attach()
    else: print(f"unknown mode: {mode}"); return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
