"""Checkpoints with provenance and config hash.

Every checkpoint written by :func:`save_checkpoint` records, as information:

* ``sources``: the datasets and weights it was built from — each with its listed license and source from
  :mod:`gyeol.core.assets` (``{"name", "license", "source", …}``; unlisted names are recorded as such),
* ``parents``: gyeol checkpoints it was initialised from (name and their sources),
* ``config_hash``: SHA-256 of the canonical JSON training config.

:func:`load_checkpoint` reads gyeol checkpoints and plain state dicts alike.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import torch

from ..core.assets import describe


def config_hash(config: Any) -> str:
    if is_dataclass(config):
        config = asdict(config)
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


@dataclass
class CheckpointInfo:
    name: str
    sources: list[dict]  # describe(name) of every dataset / weight file it was built from
    config_hash: str
    parents: list[dict] = field(default_factory=list)  # {"name", "sources"} of gyeol checkpoints it was initialised from

    @property
    def source_names(self) -> list[str]:
        return [s["name"] for s in self.sources]


def save_checkpoint(path: str | Path, state_dict: dict, *, name: str, sources: list[str], config: Any,
                    parents: list[CheckpointInfo] | None = None, extra: dict | None = None) -> CheckpointInfo:
    """Write a checkpoint with its provenance: ``sources`` are asset names (datasets, weights)."""
    from .state import atomic_save

    lineage = [{"name": p.name, "sources": list(p.sources)} for p in (parents or [])]
    info = CheckpointInfo(name, [describe(s) for s in sources], config_hash(config), lineage)
    atomic_save({"state_dict": state_dict, "gyeol": {**asdict(info), **(extra or {})}}, path)
    return info


def load_checkpoint(path: str | Path, *, map_location: str = "cpu") -> tuple[dict, CheckpointInfo]:
    """(state dict, provenance) of a gyeol checkpoint; a plain state dict loads with empty provenance."""
    blob = torch.load(path, map_location=map_location, weights_only=True)
    meta = blob.get("gyeol") if isinstance(blob, dict) else None
    if meta is None:
        sd = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
        return sd, CheckpointInfo(Path(path).stem, [], "")
    sources = [s if isinstance(s, dict) else describe(s) for s in meta.get("sources", [])]
    return blob["state_dict"], CheckpointInfo(meta["name"], sources, meta.get("config_hash", ""), list(meta.get("parents", [])))


def load_weights(path: str | Path, *, map_location: str = "cpu") -> Any:
    """Read a third-party weight file (e.g. a fetched RMVPE or HuBERT checkpoint)."""
    return torch.load(path, map_location=map_location, weights_only=True)
