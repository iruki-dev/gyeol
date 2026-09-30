"""Dataset manifests with license tags, and the license-gated loader.

A manifest is a JSON file::

    {"dataset": "vocalset",               # key in gyeol.core.license.REGISTRY
     "root": "/data/vocalset",
     "items": [{"path": "f1/arpeggios/belt/f1_arpeggios_belt_a.wav",
                "singer": "f1", "labels": {"technique": "belt"}}, ...]}

:func:`open_manifest` resolves the dataset's :class:`LicensedAsset` and calls
:func:`~gyeol.core.license.require_allowed` for the active profile *before*
any item is exposed, so a commercial training run cannot touch
non-commercial data.  Dataset-specific adapters (M2) produce manifests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from ..core.license import LicensedAsset, Profile, lookup, require_allowed


@dataclass
class ManifestItem:
    path: str
    singer: str | None = None
    labels: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Manifest:
    dataset: str
    root: str
    items: list[ManifestItem]

    @classmethod
    def read(cls, path: str | Path) -> "Manifest":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["dataset"], d.get("root", ""), [ManifestItem(**i) for i in d.get("items", [])])

    def write(self, path: str | Path) -> None:
        payload = {"dataset": self.dataset, "root": self.root, "items": [i.__dict__ for i in self.items]}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


class LicensedDataset:
    """Iterable view of a manifest that has passed the license gate."""

    def __init__(self, manifest: Manifest, asset: LicensedAsset, profile: Profile):
        self.manifest = manifest
        self.asset = asset
        self.profile = profile

    def __len__(self) -> int:
        return len(self.manifest.items)

    def __iter__(self) -> Iterator[ManifestItem]:
        return iter(self.manifest.items)

    def resolve(self, item: ManifestItem) -> Path:
        return Path(self.manifest.root) / item.path


def open_manifest(manifest: Manifest | str | Path, profile: Profile | str) -> LicensedDataset:
    """Open a manifest under ``profile``; raises LicenseError if not allowed."""
    m = manifest if isinstance(manifest, Manifest) else Manifest.read(manifest)
    asset = lookup(m.dataset)
    require_allowed(asset, Profile(profile))
    return LicensedDataset(m, asset, Profile(profile))
