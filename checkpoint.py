from __future__ import annotations

import dataclasses
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False


CHECKPOINT_VERSION = 1


class CheckpointError(RuntimeError):
    pass


def _is_dataclass_instance(obj: Any) -> bool:
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def _is_torch_module(obj: Any) -> bool:
    return _TORCH_AVAILABLE and isinstance(obj, torch.nn.Module)


def _is_torch_optimizer(obj: Any) -> bool:
    return _TORCH_AVAILABLE and isinstance(
        obj, torch.optim.Optimizer
    )


def to_payload(obj: Any) -> dict:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return {"kind": "scalar", "value": obj}
    if isinstance(obj, np.ndarray):
        return {
            "kind": "ndarray",
            "value": obj,
            "dtype": str(obj.dtype),
            "shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {"kind": "scalar", "value": obj.item()}
    if _is_torch_module(obj):
        return {
            "kind": "torch_module",
            "class": type(obj).__name__,
            "state_dict": {k: v.detach().cpu().numpy()
                           for k, v in obj.state_dict().items()},
        }
    if _is_torch_optimizer(obj):
        return {
            "kind": "torch_optimizer",
            "class": type(obj).__name__,
            "state_dict": obj.state_dict(),
        }
    if _is_dataclass_instance(obj):
        return {
            "kind": "dataclass",
            "class": type(obj).__name__,
            "fields": {f.name: to_payload(getattr(obj, f.name))
                       for f in dataclasses.fields(obj)},
        }
    if isinstance(obj, dict):
        return {"kind": "dict", "items": {str(k): to_payload(v) for k, v in obj.items()}}
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
        if not _TORCH_AVAILABLE:
            raise CheckpointError("torch not available")
        if target_class is None:
            raise CheckpointError(f"loading torch module {payload['class']} requires target_class")
        model = target_class.__new__(target_class)
        return model
    if kind == "torch_optimizer":
        if not _TORCH_AVAILABLE:
            raise CheckpointError("torch not available")
        return payload["state_dict"]
    if kind == "dataclass":
        if target_class is None:
            raise CheckpointError(f"loading dataclass {payload['class']} requires target_class")
        kwargs = {k: from_payload(v) for k, v in payload["fields"].items()}
        return target_class(**kwargs)
    if kind == "dict":
        return {k: from_payload(v) for k, v in payload["items"].items()}
    if kind == "list":
        return [from_payload(x) for x in payload["items"]]
    if kind == "tuple":
        return tuple(from_payload(x) for x in payload["items"])
    if kind == "pickled":
        return pickle.loads(payload["bytes"])
    raise CheckpointError(f"unknown payload kind: {kind}")


class CheckpointManager:
    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
    def _path(self, name: str) -> Path:
        return self.root / f"{name}.ckpt.pkl"
    def save(self, name: str, obj: Any, metadata: Optional[dict] = None) -> Path:
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
            raise CheckpointError(f"checkpoint not found: {path}")
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        if payload["version"] != CHECKPOINT_VERSION:
            raise CheckpointError(f"checkpoint version mismatch: file={payload['version']} code={CHECKPOINT_VERSION}")
        obj = from_payload(payload["payload"], target_class=target_class)
        if _is_torch_module(obj) and payload["payload"]["kind"] == "torch_module":
            sd = {k: torch.from_numpy(np.asarray(v)) for k, v in payload["payload"]["state_dict"].items()}
            obj.load_state_dict(sd)
        return obj
    def metadata(self, name: str) -> dict:
        path = self._path(name)
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        return {"version": payload["version"], "timestamp": payload["timestamp"], "metadata": payload["metadata"]}
    def list(self) -> list[str]:
        return sorted(p.stem.replace(".ckpt", "") for p in self.root.glob("*.ckpt.pkl"))
    def delete(self, name: str) -> None:
        path = self._path(name)
        if path.exists():
            path.unlink()
    def clear(self) -> None:
        for p in self.root.glob("*.ckpt.pkl"):
            p.unlink()


def save_checkpoint(path: str, obj: Any, metadata: Optional[dict] = None) -> None:
    CheckpointManager(os.path.dirname(path) or ".").save(
        os.path.basename(path).replace(".ckpt.pkl", ""), obj, metadata=metadata)


def load_checkpoint(path: str, target_class: Optional[type] = None) -> Any:
    return CheckpointManager(os.path.dirname(path) or ".").load(
        os.path.basename(path).replace(".ckpt.pkl", ""), target_class=target_class)
