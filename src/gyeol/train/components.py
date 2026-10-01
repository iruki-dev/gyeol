"""Per-component training modes and license-gated initial weights (revision B5).

Every trainable component of a task has a mode:

* ``scratch`` — random initialisation, trained at the base learning rate;
* ``finetune`` — initialised from a local checkpoint (``init`` is required),
  trained at ``base_lr · finetune_lr_scale``;
* ``freeze`` — weights fixed (``requires_grad=False``) and kept in eval mode
  (no dropout, frozen normalisation statistics).  It may start from a
  checkpoint or from its random initialisation (useful in smoke tests).

Components: ``ssl`` (SSL encoder), ``pitch`` (RMVPE), ``residual_encoder``,
``singer_encoder``, ``env_encoder``, ``acoustic``, ``vocoder``,
``discriminators``, and ``heads``.

Initial weights always come from a **local** file, through the license gate:

* a gyeol checkpoint (written by :func:`gyeol.train.checkpoint.save_checkpoint`)
  is opened with :func:`load_checkpoint` under the run's profile — its
  embedded license tag becomes part of the new run's lineage;
* a registered third-party checkpoint (``asset: rmvpe``, ``asset: hubert_fairseq``…)
  is opened with :func:`load_third_party` after :func:`require_allowed` for
  that asset.

Nothing is downloaded.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..core.license import LicenseTag, Profile, lookup
from .checkpoint import CheckpointInfo, load_checkpoint, load_third_party

MODES = ("scratch", "finetune", "freeze")
COMPONENTS = ("ssl", "pitch", "residual_encoder", "singer_encoder", "env_encoder", "acoustic", "vocoder", "discriminators", "heads")


@dataclass
class InitSpec:
    path: str
    asset: str | None = None  # registry name for third-party weights; None = gyeol checkpoint
    prefix: str = ""  # take only keys under this prefix (and strip it), e.g. "vocoder."
    strict: bool = True


@dataclass
class ComponentPlan:
    modes: dict[str, str] = field(default_factory=dict)
    init: dict[str, InitSpec] = field(default_factory=dict)
    finetune_lr_scale: float = 0.1
    checkpointing: tuple[str, ...] = ()

    def mode(self, name: str) -> str:
        return self.modes.get(name, "scratch")

    def validate(self, available: tuple[str, ...]) -> None:
        for name, m in self.modes.items():
            if name not in COMPONENTS:
                raise ValueError(f"unknown component {name!r}; known: {', '.join(COMPONENTS)}")
            if m not in MODES:
                raise ValueError(f"component {name!r}: mode {m!r} is not one of {MODES}")
        for name in list(self.modes) + list(self.init) + list(self.checkpointing):
            if name not in available:
                raise ValueError(f"component {name!r} is not part of this task (it has {', '.join(available)})")
        for name in available:
            if self.mode(name) == "finetune" and name not in self.init:
                raise ValueError(f"component {name!r} is set to finetune but has no init checkpoint")


@dataclass
class Lineage:
    """Sources a run's weights derive from: dataset / asset names and parent gyeol checkpoints."""

    assets: list[str] = field(default_factory=list)
    parents: list[CheckpointInfo] = field(default_factory=list)

    def tags(self) -> list[LicenseTag]:
        return [lookup(a).tag for a in self.assets] + [p.license for p in self.parents]


def _select(sd: dict, prefix: str) -> dict:
    if not prefix:
        return sd
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def load_initial_weights(module: nn.Module, spec: InitSpec, profile: Profile, lineage: Lineage,
                         loader: Callable[[nn.Module, dict], None] | None = None) -> None:
    """Load ``spec`` into ``module`` after the license gate and record it in ``lineage``."""
    path = Path(spec.path).expanduser()
    if not path.is_file():
        hint = f" (fetch it with `gyeol fetch {spec.asset}`)" if spec.asset else ""
        raise FileNotFoundError(f"initial weights not found: {path}{hint}")
    if spec.asset:
        sd = load_third_party(path, spec.asset, profile)
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        lineage.assets.append(spec.asset)
    else:
        sd, info = load_checkpoint(path, profile)
        lineage.parents.append(info)
    sd = _select(sd, spec.prefix)
    if loader is not None:
        loader(module, sd)
        return
    res = module.load_state_dict(sd, strict=False)
    if spec.strict and (res.missing_keys or res.unexpected_keys):
        raise ValueError(f"{path}: weights do not match the component: missing {res.missing_keys[:8]}, unexpected {res.unexpected_keys[:8]}")


def apply_mode(module: nn.Module, mode: str) -> None:
    if mode == "freeze":
        for p in module.parameters():
            p.requires_grad_(False)
        module.eval()
    else:
        for p in module.parameters():
            p.requires_grad_(True)


def keep_frozen_in_eval(modules: dict[str, nn.Module], plan: ComponentPlan) -> None:
    """Call after ``model.train()``: frozen components stay in eval mode."""
    for name, m in modules.items():
        if plan.mode(name) == "freeze":
            m.eval()


def param_groups(components: dict[str, nn.Module], plan: ComponentPlan, base_lr: float) -> list[dict]:
    """Optimizer parameter groups: scratch at ``base_lr``, finetune scaled, frozen left out (shared params counted once)."""
    groups, seen = [], set()
    for name, m in components.items():
        mode = plan.mode(name)
        if mode == "freeze":
            continue
        params = [p for p in m.parameters() if p.requires_grad and id(p) not in seen]
        seen.update(id(p) for p in params)
        if params:
            groups.append({"params": params, "lr": base_lr * (plan.finetune_lr_scale if mode == "finetune" else 1.0), "name": name})
    return groups


# ---------------------------------------------------------------- gradient checkpointing (B4)


def _checkpointed(module: nn.Module, forward, *args, **kwargs):
    if module.training and torch.is_grad_enabled() and any(p.requires_grad for p in module.parameters()):
        return checkpoint(forward, *args, use_reentrant=False, **kwargs)
    return forward(*args, **kwargs)


def enable_gradient_checkpointing(module: nn.Module) -> int:
    """Recompute activations of each block in the module's ``ModuleList``\\ s during backward.

    Blocks are wrapped by replacing their bound ``forward`` (the state dict
    is unchanged).  Modules with their own switch (``checkpointing``
    attribute, e.g. the BS-RoFormer) use it instead.  Returns the number of
    wrapped blocks.
    """
    if hasattr(module, "checkpointing"):
        module.checkpointing = True
        return 1
    n = 0

    def visit(m: nn.Module) -> None:
        nonlocal n
        for child in m.children():
            if isinstance(child, nn.ModuleList):
                for block in child:
                    if any(True for _ in block.parameters()) and not getattr(block, "_gyeol_ckpt", False):
                        block.forward = functools.partial(_checkpointed, block, block.forward)
                        block._gyeol_ckpt = True
                        n += 1
            else:
                visit(child)

    visit(module)
    return n


__all__ = ["COMPONENTS", "MODES", "ComponentPlan", "InitSpec", "Lineage", "apply_mode", "enable_gradient_checkpointing",
           "keep_frozen_in_eval", "load_initial_weights", "param_groups"]
