"""Verifies the wiring fixture and the two previously-missing modules."""

import numpy as np
import pytest


# --- checkpoint -------------------------------------------------------------

def test_checkpoint_roundtrip(tmp_path):
    from checkpoint import CheckpointManager
    mgr = CheckpointManager(str(tmp_path))
    arr = np.arange(12).reshape(3, 4).astype(np.float32)
    mgr.save("arr", arr)
    arr2 = mgr.load("arr")
    assert np.allclose(arr, arr2)


def test_checkpoint_dataclass(tmp_path):
    import dataclasses
    from checkpoint import CheckpointManager

    @dataclasses.dataclass
    class Cfg:
        n: int
        lr: float

    mgr = CheckpointManager(str(tmp_path))
    mgr.save("cfg", Cfg(n=8, lr=3e-4))
    cfg = mgr.load("cfg", target_class=Cfg)
    assert cfg.n == 8 and cfg.lr == 3e-4


def test_checkpoint_list_and_delete(tmp_path):
    from checkpoint import CheckpointManager
    mgr = CheckpointManager(str(tmp_path))
    mgr.save("a", 1)
    mgr.save("b", 2)
    assert set(mgr.list()) == {"a", "b"}
    mgr.delete("a")
    assert mgr.list() == ["b"]


# --- observation_validator --------------------------------------------------

def test_validator_discrete_pass():
    from observation_validator import validate_observation, grid_spec
    spec = grid_spec(4)
    assert validate_observation(np.array([5]), spec).passed


def test_validator_discrete_fail_range():
    from observation_validator import validate_observation, grid_spec
    spec = grid_spec(4)
    r = validate_observation(np.array([99]), spec)
    assert not r.passed
    assert any("out of range" in i for i in r.issues)


def test_validator_strict_raises():
    from observation_validator import validate_observation, grid_spec
    spec = grid_spec(4)
    with pytest.raises(ValueError):
        validate_observation(np.array([99]), spec, strict=True)


def test_validator_probability():
    from observation_validator import validate_observation, probability_spec
    spec = probability_spec(4)
    assert validate_observation(np.array([0.25] * 4), spec).passed
    # sum != 1
    assert not validate_observation(np.array([0.5, 0.5, 0.5, 0.5]), spec).passed


def test_validator_batch():
    from observation_validator import validate_batch, grid_spec
    spec = grid_spec(4)
    good = np.array([[0], [1], [2], [3]])
    assert validate_batch(good, spec).passed
    bad = np.array([[0], [99], [2]])
    assert not validate_batch(bad, spec).passed


# --- wiring -----------------------------------------------------------------

def test_registry_resolves_present_components():
    from tatha_wiring import ComponentRegistry
    reg = ComponentRegistry()
    for name in ("GenerativeModel", "run_fpi", "compute_efe",
                 "PathIntegralPlanner", "QEPSOSwarm"):
        obj = reg.resolve(name)
        if obj is None:
            pytest.skip(f"{name} not present in repo")


def test_registry_reports_unavailable():
    from tatha_wiring import ComponentRegistry
    reg = ComponentRegistry()
    avail = reg.available()
    unavail = reg.unavailable()
    assert isinstance(avail, list)
    assert isinstance(unavail, dict)


def test_stack_runs_grid_episode():
    from tatha_wiring import TathaStack
    stack = TathaStack.from_available()
    if not stack.has("GenerativeModel") or not stack.has("run_fpi"):
        pytest.skip("active_inference module not available")
    result = stack.run_grid_episode(n=4, steps=5)
    assert result["beliefs"].shape[0] == 5
    assert len(result["actions"]) == 5
    assert all(0 <= a < 4 for a in result["actions"])


def test_wire_attaches_attributes():
    from tatha_wiring import wire

    class Dummy:
        pass

    agent = Dummy()
    wire(agent)
    assert hasattr(agent, "_wired_components")
    assert isinstance(agent._wired_components, list)


def test_probe_returns_report(capsys):
    from tatha_wiring import probe
    report = probe(verbose=False)
    assert "available" in report
    assert "unavailable" in report
    assert "by_category" in report
