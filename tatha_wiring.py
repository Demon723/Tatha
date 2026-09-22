from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class ComponentSpec:
    name: str
    module: str
    attr: Optional[str]
    kind: str
    requires: list = field(default_factory=list)
    category: str = "misc"
    notes: str = ""


WIRING_MANIFEST: list[ComponentSpec] = [
    ComponentSpec(name="PathIntegralPlanner", module="path_integral", attr="PathIntegralPlanner", kind="class", requires=["numpy"], category="quantum", notes="Trotter-Suzuki propagator for the EFE computation"),
    ComponentSpec(name="PrecisionHyperModel", module="precision_hyper", attr="PrecisionHyperModel", kind="class", requires=["numpy", "torch"], category="quantum", notes="Adaptive per-channel precision"),
    ComponentSpec(name="QEPSOSwarm", module="qepso", attr="QEPSOSwarm", kind="class", requires=["numpy"], category="quantum", notes="Quantum-entangled particle swarm optimization"),
    ComponentSpec(name="DensitySwarmConfig", module="density_swarm", attr="DensitySwarmConfig", kind="class", requires=["numpy"], category="quantum", notes="Lindblad evolution config"),
    ComponentSpec(name="DensityMatrixSwarm", module="density_swarm", attr="DensityMatrixSwarm", kind="class", requires=["numpy"], category="quantum", notes="Lindblad evolution of a single density operator"),
    ComponentSpec(name="QuantumSwarm", module="quantum_swarm", attr="QuantumSwarm", kind="class", requires=["numpy"], category="swarm", notes="Multi-agent quantum swarm"),
    ComponentSpec(name="AdversarialSwarm", module="adversarial_swarm", attr="AdversarialSwarm", kind="class", requires=["numpy"], category="swarm", notes="RED vs BLUE zero-sum swarm game"),
    ComponentSpec(name="GridWorld", module="gridworld_pc", attr="GridWorld", kind="class", requires=["numpy"], category="env", notes="Grid world for predictive coding"),
    ComponentSpec(name="SpaceGridWorld", module="gridworld_pc", attr="SpaceGridWorld", kind="class", requires=["numpy"], category="env", notes="Spatial grid world variant"),
    ComponentSpec(name="GenerativeModel", module="active_inference", attr="GenerativeModel", kind="class", requires=["numpy"], category="ai", notes="A/B/C/D/E discrete POMDP"),
    ComponentSpec(name="run_fpi", module="active_inference", attr="run_fpi", kind="function", requires=["numpy"], category="ai", notes="Fixed-point variational inference"),
    ComponentSpec(name="compute_efe", module="active_inference", attr="compute_efe", kind="function", requires=["numpy"], category="ai", notes="Expected free energy decomposition"),
    ComponentSpec(name="GymWrapper", module="active_gym", attr="GymWrapper", kind="class", requires=["numpy", "gymnasium"], category="gym", notes="Gymnasium environment adapter"),
    ComponentSpec(name="Discretizer", module="active_gym", attr="Discretizer", kind="class", requires=["numpy"], category="gym", notes="Continuous to discrete observation binning"),
    ComponentSpec(name="active_visualization", module="active_visualization", attr=None, kind="module", requires=["numpy", "matplotlib"], category="viz", notes="Belief/FE/error plots"),
    ComponentSpec(name="Tatha", module="unified_pc_space", attr="Tatha", kind="class", requires=["numpy"], category="core", notes="Base predictive coding agent"),
    ComponentSpec(name="QuantumTatha", module="unified_pc_space", attr="QuantumTatha", kind="class", requires=["numpy"], category="core", notes="Quantum predictive coding agent"),
    ComponentSpec(name="AdaptiveQuantumTatha", module="unified_pc_space", attr="AdaptiveQuantumTatha", kind="class", requires=["numpy"], category="core", notes="Adaptive quantum predictive coding agent"),
    ComponentSpec(name="CheckpointManager", module="checkpoint", attr="CheckpointManager", kind="class", requires=["numpy"], category="infra", notes="Generic save/load for all components"),
    ComponentSpec(name="validate_observation", module="observation_validator", attr="validate_observation", kind="function", requires=["numpy"], category="infra", notes="Runtime observation validation"),
]


def _has_module(name: str) -> bool:
    if name in sys.modules:
        return True
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def _has_all(requires: list) -> tuple[bool, list[str]]:
    missing = [r for r in requires if not _has_module(r)]
    return len(missing) == 0, missing


class ComponentRegistry:
    def __init__(self, manifest: Optional[list] = None):
        self.manifest = manifest or WIRING_MANIFEST
        self._cache: dict[str, Any] = {}
        self._failures: dict[str, str] = {}

    def resolve(self, name: str) -> Optional[Any]:
        if name in self._cache:
            return self._cache[name]
        if name in self._failures:
            return None
        spec = next((s for s in self.manifest if s.name == name), None)
        if spec is None:
            self._failures[name] = f"no spec for {name}"
            return None
        ok, missing = _has_all(spec.requires)
        if not ok:
            self._failures[name] = f"missing deps: {missing}"
            return None
        try:
            mod = importlib.import_module(spec.module)
            obj = mod if spec.attr is None else getattr(mod, spec.attr)
        except (ImportError, AttributeError) as exc:
            self._failures[name] = f"import failed: {exc}"
            return None
        self._cache[name] = obj
        return obj

    def available(self) -> list[str]:
        return [s.name for s in self.manifest if self.resolve(s.name) is not None]

    def unavailable(self) -> dict[str, str]:
        out = {}
        for s in self.manifest:
            if self.resolve(s.name) is None:
                out[s.name] = self._failures.get(s.name, "unknown")
        return out

    def by_category(self, category: str) -> list[str]:
        return [s.name for s in self.manifest if s.category == category and self.resolve(s.name) is not None]


def probe(verbose: bool = True, registry: Optional[ComponentRegistry] = None) -> dict:
    reg = registry or ComponentRegistry()
    report = {"available": reg.available(), "unavailable": reg.unavailable(), "by_category": {}}
    for s in reg.manifest:
        report["by_category"].setdefault(s.category, [])
        if reg.resolve(s.name) is not None:
            report["by_category"][s.category].append(s.name)
    if verbose:
        print("=" * 72)
        print("TATHA WIRING PROBE")
        print("=" * 72)
        for cat in sorted(report["by_category"]):
            print(f"\n[{cat}]")
            for name in report["by_category"][cat]:
                spec = next(s for s in reg.manifest if s.name == name)
                print(f"  OK   {name:<28} {spec.notes}")
        if report["unavailable"]:
            print("\n[unavailable]")
            for name, reason in sorted(report["unavailable"].items()):
                print(f"  FAIL {name:<28} {reason}")
        print()
        print(f"available: {len(report['available'])} / {len(reg.manifest)}")
        print("=" * 72)
    return report


class TathaStack:
    def __init__(self, registry: Optional[ComponentRegistry] = None):
        self.reg = registry or ComponentRegistry()
        self.components: dict[str, Any] = {}
        for spec in self.reg.manifest:
            obj = self.reg.resolve(spec.name)
            if obj is not None:
                self.components[spec.name] = obj

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
        if not self.has("GenerativeModel"):
            raise RuntimeError("GenerativeModel not available")
        GM = self.get("GenerativeModel")
        run_fpi = self.get("run_fpi")
        compute_efe = self.get("compute_efe")
        from tatha_master import select_action
        n_s = n * n
        A = [np.eye(n_s) * 0.9 + 0.1 / n_s]
        A[0] = A[0] / A[0].sum(axis=0, keepdims=True)
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
        C = [np.ones(n_s) * 0.1 / n_s]
        C[0][-1] = 0.9
        C[0] = C[0] / C[0].sum()
        D = [np.ones(n_s) / n_s]
        B = [B]
        model = GM(A=A, B=B, C=C, D=D)
        rng = np.random.default_rng(seed)
        qs_prev = None
        beliefs, actions = [], []
        for t in range(steps):
            obs = [int(rng.integers(0, n_s))]
            res = run_fpi(model, obs=obs, qs_prev=qs_prev, num_iter=16)
            beliefs.append(res.qs[0].copy())
            policy, bd, _ = select_action(model, res.qs, [(a,) for a in range(4)], rng=rng)
            actions.append(policy[0])
            qs_prev = res.qs
        return {"beliefs": np.array(beliefs), "actions": actions, "final_free_energy": float(res.free_energy), "n_steps": steps}

    def run_path_integral(self, grid: int = 16, slices: int = 8) -> dict:
        if not self.has("PathIntegralPlanner"):
            return {"skipped": "PathIntegralPlanner not available"}
        planner = self.get("PathIntegralPlanner")(grid_size=grid, n_slices=slices)
        V = np.zeros((grid, grid)); V[grid // 2, grid // 2] = -5.0
        psi0 = np.zeros((grid, grid), dtype=complex); psi0[0, 0] = 1.0
        psi = planner.normalize(planner.propagate(psi0, V))
        return {"norm": float(np.sum(np.abs(psi) ** 2))}

    def run_density_swarm(self, dim: int = 8, steps: int = 20) -> dict:
        if not self.has("DensityMatrixSwarm"):
            return {"skipped": "DensityMatrixSwarm not available"}
        cfg_cls = self.get("DensitySwarmConfig") if self.has("DensitySwarmConfig") else None
        if cfg_cls is not None:
            dm = self.get("DensityMatrixSwarm")(cfg_cls(hilbert_dim=dim))
        else:
            from density_swarm import DensitySwarmConfig
            dm = self.get("DensityMatrixSwarm")(DensitySwarmConfig(hilbert_dim=dim))
        dm.evolve(steps)
        return {"populations_sum": float(dm.populations().sum()), "coherence": float(dm.coherence())}

    def summary(self) -> dict:
        return {"n_available": len(self.components), "n_total": len(self.reg.manifest), "available": sorted(self.components.keys())}


def wire(agent: Any, registry: Optional[ComponentRegistry] = None) -> Any:
    reg = registry or ComponentRegistry()
    attached = []
    for spec in reg.manifest:
        attr = spec.name.lower()
        if hasattr(agent, attr):
            continue
        obj = reg.resolve(spec.name)
        if obj is None:
            continue
        try:
            setattr(agent, attr, obj)
            attached.append(spec.name)
        except Exception:
            continue
    agent._wired_components = attached
    return agent


def main(argv: Optional[list] = None) -> int:
    argv = argv or sys.argv
    if len(argv) < 2:
        probe()
        return 0
    mode = argv[1]
    if mode == "probe":
        probe()
    elif mode == "stack":
        stack = TathaStack.from_available()
        print("stack summary:", stack.summary())
        result = stack.run_grid_episode()
        print(f"grid episode: {result['n_steps']} steps, final F={result['final_free_energy']:.4f}")
        pi = stack.run_path_integral()
        print("path integral:", pi)
        ds = stack.run_density_swarm()
        print("density swarm:", ds)
    else:
        print(f"unknown mode: {mode}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
