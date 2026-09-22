from __future__ import annotations

import dataclasses
import os
import pickle
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import torch
    _TORCH = True
except ImportError:
    torch = None
    _TORCH = False


CHECKPOINT_VERSION = 1


class CheckpointError(RuntimeError):
    pass


def _is_dc(obj):
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def _is_torch_mod(obj):
    return _TORCH and isinstance(obj, torch.nn.Module)


def to_payload(obj: Any) -> dict:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return {"kind": "scalar", "value": obj}
    if isinstance(obj, np.ndarray):
        return {"kind": "ndarray", "value": obj}
    if isinstance(obj, np.generic):
        return {"kind": "scalar", "value": obj.item()}
    if _is_torch_mod(obj):
        return {
            "kind": "torch_module",
            "class": type(obj).__name__,
            "state_dict": {k: v.detach().cpu().numpy()
                           for k, v in obj.state_dict().items()},
        }
    if _TORCH and isinstance(obj, torch.optim.Optimizer):
        return {"kind": "torch_optimizer", "state_dict": obj.state_dict()}
    if _is_dc(obj):
        return {
            "kind": "dataclass",
            "class": type(obj).__name__,
            "fields": {f.name: to_payload(getattr(obj, f.name))
                       for f in dataclasses.fields(obj)},
        }
    if isinstance(obj, dict):
        return {"kind": "dict",
                "items": {str(k): to_payload(v) for k, v in obj.items()}}
    if isinstance(obj, (list, tuple)):
        return {"kind": "list" if isinstance(obj, list) else "tuple",
                "items": [to_payload(x) for x in obj]}
    return {"kind": "pickled", "bytes": pickle.dumps(obj)}


def from_payload(payload: dict, target_class: Optional[type] = None) -> Any:
    kind = payload["kind"]
    if kind == "scalar":
        return payload["value"]
    if kind == "ndarray":
        return np.asarray(payload["value"])
    if kind == "torch_module":
        if not _TORCH:
            raise CheckpointError("torch not available")
        if target_class is None:
            raise CheckpointError(
                f"loading torch module {payload['class']} needs target_class")
        model = target_class.__new__(target_class)
        sd = {k: torch.from_numpy(np.asarray(v))
              for k, v in payload["state_dict"].items()}
        model.load_state_dict(sd)
        return model
    if kind == "torch_optimizer":
        return payload["state_dict"]
    if kind == "dataclass":
        if target_class is None:
            raise CheckpointError(
                f"loading dataclass {payload['class']} needs target_class")
        return target_class(**{k: from_payload(v)
                               for k, v in payload["fields"].items()})
    if kind == "dict":
        return {k: from_payload(v) for k, v in payload["items"].items()}
    if kind == "list":
        return [from_payload(x) for x in payload["items"]]
    if kind == "tuple":
        return tuple(from_payload(x) for x in payload["items"])
    if kind == "pickled":
        return pickle.loads(payload["bytes"])
    raise CheckpointError(f"unknown kind: {kind}")


class CheckpointManager:
    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
    def _path(self, name: str) -> Path:
        return self.root / f"{name}.ckpt.pkl"
    def save(self, name: str, obj: Any,
             metadata: Optional[dict] = None) -> Path:
        payload = {"version": CHECKPOINT_VERSION, "name": name,
                    "timestamp": time.time(), "metadata": metadata or {},
                    "payload": to_payload(obj)}
        path = self._path(name)
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        return path
    def load(self, name: str, target_class: Optional[type] = None) -> Any:
        path = self._path(name)
        if not path.exists():
            raise CheckpointError(f"not found: {path}")
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        if payload["version"] != CHECKPOINT_VERSION:
            raise CheckpointError("version mismatch")
        return from_payload(payload["payload"], target_class=target_class)
    def list(self) -> list:
        return sorted(p.stem.replace(".ckpt", "")
                      for p in self.root.glob("*.ckpt.pkl"))
    def delete(self, name: str) -> None:
        p = self._path(name)
        if p.exists():
            p.unlink()
    def metadata(self, name: str) -> dict:
        with open(self._path(name), "rb") as fh:
            payload = pickle.load(fh)
        return {"version": payload["version"],
                "timestamp": payload["timestamp"],
                "metadata": payload["metadata"]}


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        mgr = CheckpointManager(tmp)
        arr = np.arange(12).reshape(3, 4).astype(np.float32)
        mgr.save("arr", arr, metadata={"tag": "test"})
        assert np.allclose(arr, mgr.load("arr"))
        @dataclasses.dataclass
        class C:
            n: int
            lr: float
        mgr.save("cfg", C(n=8, lr=3e-4))
        c2 = mgr.load("cfg", target_class=C)
        assert c2.n == 8 and c2.lr == 3e-4
        print("checkpoint.py: OK, files:", mgr.list())
