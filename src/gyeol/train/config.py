"""Training-run configuration (YAML) for ``gyeol train <task> --config <yaml>`` (revision B2).

Example (``configs/cpu-smoke/autoencoder.yaml``)::

    task: autoencoder
    seed: 0
    profile: commercial
    device: auto          # auto → CUDA if available, else CPU (float32)
    threads: 0            # 0 = PyTorch default
    data:
      cache: runs/cpu-smoke/cache
      manifests: []       # gyeol manifests (adapters output); prepared into the cache on first use
      synthetic: {n_singers: 6, seconds: 1.2}
      prepare: {sr: 22050, hop: 256, separation: "off", features: [dsp]}
      split: {train: 0.6, val: 0.2, test: 0.2}
      crop_frames: [32, 64]
      batch_size: 4
      num_workers: 0
    optim: {lr: 2.0e-3, grad_accum: 2, grad_clip: 1.0, schedule: constant}
    run: {out: runs/cpu-smoke/autoencoder, max_steps: 30, val_every: 10, save_every: 10, log_every: 5, patience: 5}
    model: {size: tiny}
    components: {vocoder: scratch}
    init: {}
    checkpointing: []

Unknown keys are errors (a typo must not silently fall back to a default).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .components import ComponentPlan, InitSpec

TASKS = ("heads", "autoencoder", "vocoder", "pitch", "ssl")


@dataclass
class DataConfig:
    cache: str = "runs/cache"
    manifests: list[str] = field(default_factory=list)
    synthetic: dict | None = None  # {n_singers, seconds, takes_per_cell, seed}: generate a synthetic corpus (CI / smoke)
    prepare: dict = field(default_factory=dict)  # PrepareConfig fields
    split: dict = field(default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1})
    split_seed: int = 0
    crop_frames: list[int] = field(default_factory=lambda: [64, 192])
    batch_size: int = 8
    num_workers: int = 0
    datasets: list[str] = field(default_factory=list)  # restrict to these dataset names (default: all prepared)


@dataclass
class OptimConfig:
    lr: float = 2e-4
    weight_decay: float = 0.0
    betas: list[float] = field(default_factory=lambda: [0.8, 0.99])
    grad_accum: int = 1
    grad_clip: float = 1.0
    schedule: str = "constant"  # constant | cosine | exponential
    warmup_steps: int = 0
    gamma: float = 0.999  # exponential decay per step
    d_lr: float | None = None  # discriminator learning rate (default: lr)


@dataclass
class RunConfig:
    out: str = "runs/train"
    max_steps: int = 1000
    max_epochs: int = 1_000_000
    val_every: int = 200
    save_every: int = 200
    log_every: int = 20
    sample_every: int = 0  # generative tasks: save audio samples every N steps (0 = at validation)
    n_samples: int = 2
    patience: int = 10  # validations without improvement before stopping (0 = never stop early)
    min_delta: float = 0.0
    val_batches: int = 0  # 0 = the whole validation split


@dataclass
class TrainConfig:
    task: str
    seed: int = 0
    profile: str = "commercial"
    device: str = "auto"
    threads: int = 0
    mixed_precision: bool = False  # CUDA only; CPU always float32
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    run: RunConfig = field(default_factory=RunConfig)
    model: dict = field(default_factory=dict)  # task-specific (see gyeol.train.tasks)
    components: dict[str, str] = field(default_factory=dict)
    init: dict[str, dict] = field(default_factory=dict)
    finetune_lr_scale: float = 0.1
    checkpointing: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"unknown task {self.task!r}; one of {', '.join(TASKS)}")
        if self.optim.grad_accum < 1 or self.data.batch_size < 1:
            raise ValueError("grad_accum and batch_size must be ≥ 1")

    def plan(self) -> ComponentPlan:
        return ComponentPlan(dict(self.components), {k: InitSpec(**v) for k, v in self.init.items()}, self.finetune_lr_scale,
                             tuple(self.checkpointing))

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def _build(cls, d: dict, where: str):
    if not isinstance(d, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(d).__name__}")
    names = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(d) - set(names))
    if unknown:
        raise ValueError(f"{where}: unknown keys {unknown}; allowed: {sorted(names)}")
    kw = {}
    for k, v in d.items():
        sub = {"data": DataConfig, "optim": OptimConfig, "run": RunConfig}.get(k) if cls is TrainConfig else None
        kw[k] = _build(sub, v or {}, f"{where}.{k}") if sub else v
    return cls(**kw)


def config_from_dict(d: dict[str, Any]) -> TrainConfig:
    return _build(TrainConfig, dict(d), "config")


def load_config(path: str | Path, task: str | None = None, overrides: list[str] | None = None) -> TrainConfig:
    """Read a YAML (or JSON) run config; ``task`` must match the file's task if both are given.

    ``overrides``: ``["run.max_steps=5", "optim.lr=1e-3"]`` (values parsed as YAML scalars).
    """
    text = Path(path).read_text(encoding="utf-8")
    if str(path).endswith(".json"):
        d = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - pyyaml is in the [train] extra
            raise ImportError("reading YAML configs needs PyYAML: pip install 'gyeol[train]'") from exc
        d = yaml.safe_load(text) or {}
    if task is not None:
        if d.get("task", task) != task:
            raise ValueError(f"{path} configures task {d.get('task')!r}, not {task!r}")
        d["task"] = task
    for ov in overrides or []:
        key, _, raw = ov.partition("=")
        if not _:
            raise ValueError(f"override {ov!r} must look like section.key=value")
        try:
            import yaml

            val = yaml.safe_load(raw)
        except ImportError:  # pragma: no cover
            val = json.loads(raw)
        node = d
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return config_from_dict(d)


__all__ = ["DataConfig", "OptimConfig", "RunConfig", "TASKS", "TrainConfig", "config_from_dict", "load_config"]
