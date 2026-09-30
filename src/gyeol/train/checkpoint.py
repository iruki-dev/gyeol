"""Checkpoints with embedded license provenance and config hash.

Every checkpoint stores

* ``license``: the effective tag of the checkpoint — the *most restrictive*
  tag among all datasets and parent checkpoints it was trained from,
* ``sources``: the names of those assets,
* ``config_hash``: SHA-256 of the canonical JSON training config,
* ``profile``: the profile it was trained under.

:func:`load_checkpoint` refuses, under the commercial profile, any
checkpoint whose effective tag is not commercially usable — including
third-party checkpoints registered in :mod:`gyeol.core.license`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import torch

from ..core.license import AssetKind, LicensedAsset, LicenseError, LicenseTag, Profile, lookup, require_allowed

_ORDER = [LicenseTag.COMMERCIAL_OK, LicenseTag.COMMERCIAL_OK_CONDITIONAL, LicenseTag.UNKNOWN, LicenseTag.NONCOMMERCIAL, LicenseTag.COPYLEFT]


def most_restrictive(tags: list[LicenseTag]) -> LicenseTag:
    """Combine tags: the result is as restrictive as the worst input."""
    if not tags:
        return LicenseTag.UNKNOWN
    return max((LicenseTag(t) for t in tags), key=_ORDER.index)


def config_hash(config: Any) -> str:
    if is_dataclass(config):
        config = asdict(config)
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


@dataclass
class CheckpointInfo:
    name: str
    license: LicenseTag
    sources: list[str]
    config_hash: str
    profile: Profile
    conditions: list[str]

    def as_asset(self) -> LicensedAsset:
        return LicensedAsset(self.name, AssetKind.CHECKPOINT, self.license, self.license.value, conditions=tuple(self.conditions), verified=True)


def save_checkpoint(path: str | Path, state_dict: dict, *, name: str, sources: list[str], config: Any, profile: Profile) -> CheckpointInfo:
    assets = [lookup(s) for s in sources]
    tag = most_restrictive([a.tag for a in assets])
    if Profile(profile) is Profile.COMMERCIAL and tag not in (LicenseTag.COMMERCIAL_OK, LicenseTag.COMMERCIAL_OK_CONDITIONAL):
        raise LicenseError(f"a commercial-profile checkpoint cannot be built from {tag.value} sources")
    conditions = sorted({c for a in assets for c in a.conditions})
    info = CheckpointInfo(name, tag, list(sources), config_hash(config), Profile(profile), conditions)
    torch.save({"state_dict": state_dict, "gyeol": {**asdict(info), "license": tag.value, "profile": Profile(profile).value}}, path)
    return info


def load_checkpoint(path: str | Path, profile: Profile | str, *, map_location: str = "cpu") -> tuple[dict, CheckpointInfo]:
    """Load a gyeol checkpoint, enforcing the license gate for ``profile``."""
    blob = torch.load(path, map_location=map_location, weights_only=True)
    meta = blob.get("gyeol") if isinstance(blob, dict) else None
    if meta is None:
        raise LicenseError(f"{path}: no embedded license metadata; register the checkpoint and wrap it with gyeol metadata first")
    info = CheckpointInfo(
        name=meta["name"], license=LicenseTag(meta["license"]), sources=list(meta["sources"]),
        config_hash=meta["config_hash"], profile=Profile(meta["profile"]), conditions=list(meta.get("conditions", [])),
    )
    require_allowed(info.as_asset(), Profile(profile))
    return blob["state_dict"], info


def load_third_party(path: str | Path, asset_name: str, profile: Profile | str, *, map_location: str = "cpu") -> Any:
    """Load a registered third-party checkpoint after the license gate."""
    require_allowed(lookup(asset_name), Profile(profile))
    return torch.load(path, map_location=map_location, weights_only=True)
