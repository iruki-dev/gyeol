"""Atomic training-state checkpoints and RNG capture (revision B3).

A *training state* holds everything needed to continue a run exactly where it
stopped: model(s), optimizer(s), scheduler(s), Python / NumPy / Torch RNG
states, and the run position (epoch, global step, position inside the epoch's
data order, early-stopping bookkeeping).  It is written atomically: first to
a temporary file in the same directory, then moved into place with
:func:`os.replace`, so an interruption mid-write never leaves a truncated
checkpoint behind.

Training states are *resume* files.  Release checkpoints with license
provenance are written by :func:`gyeol.train.checkpoint.save_checkpoint`.
"""

from __future__ import annotations

import os
import random
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


def atomic_save(obj: Any, path: str | Path) -> Path:
    """``torch.save`` to a temp file next to ``path``, fsync, then ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            torch.save(obj, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with _suppress():
            os.unlink(tmp)
        raise
    return path


def atomic_write_text(text: str, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with _suppress():
            os.unlink(tmp)
        raise
    return path


class _suppress:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True


def rng_state() -> dict:
    """Python, NumPy (global) and Torch (CPU and, if present, CUDA) RNG states."""
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict) -> None:
    random.setstate(_to_tuple(st["python"]))
    np.random.set_state(_to_tuple(st["numpy"]))
    torch.set_rng_state(st["torch"].cpu() if isinstance(st["torch"], torch.Tensor) else torch.as_tensor(st["torch"], dtype=torch.uint8))
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def _to_tuple(x):
    if isinstance(x, list):
        return tuple(_to_tuple(v) for v in x)
    return x


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


@dataclass
class RunPosition:
    epoch: int = 0
    step: int = 0  # optimizer steps taken
    micro_step: int = 0  # batches consumed (gradient accumulation counts each)
    data_position: int = 0  # items of the current epoch's order already consumed
    best_metric: float | None = None
    best_step: int = -1
    bad_validations: int = 0  # validations since the last improvement (early stopping)
    elapsed_s: float = 0.0
    finished: bool = False
    history: list[dict] = field(default_factory=list)  # validation records

    def as_dict(self) -> dict:
        return asdict(self)


def save_training_state(path: str | Path, *, modules: dict[str, torch.nn.Module], optimizers: dict[str, torch.optim.Optimizer],
                        schedulers: dict[str, Any], position: RunPosition, config: dict, extra: dict | None = None) -> Path:
    blob = {
        "format": "gyeol-train-state/1",
        "modules": {k: m.state_dict() for k, m in modules.items()},
        "optimizers": {k: o.state_dict() for k, o in optimizers.items()},
        "schedulers": {k: s.state_dict() for k, s in schedulers.items() if s is not None},
        "rng": rng_state(),
        "position": position.as_dict(),
        "config": config,
        "extra": extra or {},
    }
    return atomic_save(blob, path)


def load_training_state(path: str | Path, *, modules: dict[str, torch.nn.Module], optimizers: dict[str, torch.optim.Optimizer],
                        schedulers: dict[str, Any], map_location: str | torch.device = "cpu", restore_rng: bool = True) -> tuple[RunPosition, dict]:
    blob = torch.load(path, map_location=map_location, weights_only=False)  # our own file: contains RNG tuples
    if blob.get("format") != "gyeol-train-state/1":
        raise ValueError(f"{path} is not a gyeol training state")
    for k, m in modules.items():
        m.load_state_dict(blob["modules"][k])
    for k, o in optimizers.items():
        o.load_state_dict(blob["optimizers"][k])
    for k, s in schedulers.items():
        if s is not None and k in blob["schedulers"]:
            s.load_state_dict(blob["schedulers"][k])
    if restore_rng:
        set_rng_state(blob["rng"])
    return RunPosition(**blob["position"]), blob


__all__ = ["RunPosition", "atomic_save", "atomic_write_text", "load_training_state", "rng_state", "save_training_state",
           "seed_everything", "set_rng_state"]
