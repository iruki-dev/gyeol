"""Dataset manifests and the dataset view.

A manifest is a JSON file::

    {"dataset": "vocalset",               # ideally a name from gyeol.core.assets (for provenance records)
     "root": "/data/vocalset",
     "items": [{"path": "f1/arpeggios/belt/f1_arpeggios_belt_a.wav",
                "singer": "f1", "labels": {"technique": "belt"}}, ...]}

:func:`open_manifest` returns a :class:`Dataset` view.  Which datasets may be
used for what is up to the user; :func:`gyeol.core.assets.describe` gives the
listed license of a dataset for the record.  Dataset-specific adapters (M2)
produce manifests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from ..core.assets import describe


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


class Dataset:
    """Iterable view of a manifest."""

    def __init__(self, manifest: Manifest):
        self.manifest = manifest
        self.info = describe(manifest.dataset)  # listed license / source, for provenance records

    def __len__(self) -> int:
        return len(self.manifest.items)

    def __iter__(self) -> Iterator[ManifestItem]:
        return iter(self.manifest.items)

    def resolve(self, item: ManifestItem) -> Path:
        return Path(self.manifest.root) / item.path


def open_manifest(manifest: Manifest | str | Path) -> Dataset:
    """A :class:`Dataset` over ``manifest`` (an object or a JSON path)."""
    return Dataset(manifest if isinstance(manifest, Manifest) else Manifest.read(manifest))
