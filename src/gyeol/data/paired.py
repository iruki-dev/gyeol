"""Paired on/off loader with alignment.

Paired data — the same phrase sung with a technique off (control) and on
(e.g. GTSinger control/technique groups, own "breathy vs normal" takes) —
supervises attribute heads (M3) and conditional directions (M7).

For each (off, on) pair the loader

1. loads both files through the dataset view,
2. analyses them with the signal layer (f0, loudness, content, …),
3. aligns *on* to *off* with the content-only warp; separate performances
   are not sing-along takes, so the band is widened to cover their length
   difference,
4. returns both representations plus τ, so f0 / loudness can be regressed
   out before contrasting (M7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np

from ..align.warp import Warp, WarpConfig, estimate_warp
from ..core.containers import Recording, Representation
from ..core.status import Result
from ..io.audio import load_audio
from .manifest import Dataset, ManifestItem
from .splits import pairs


@dataclass
class PairedExample:
    pair_id: str
    off: Representation
    on: Representation
    warp: Warp  # maps "on" frames to "off" frames
    labels_off: dict
    labels_on: dict
    meta: dict = field(default_factory=dict)


class PairedLoader:
    def __init__(self, dataset: Dataset, analyzer: Callable[[Recording], Result[Representation]] | None = None,
                 min_band_seconds: float = 1.0):
        from ..attributes.extract import analyze

        self.dataset = dataset
        self.analyzer = analyzer or analyze
        self.min_band_seconds = min_band_seconds
        self.pairs = pairs(dataset.manifest)

    def __len__(self) -> int:
        return len(self.pairs)

    def _rep(self, item: ManifestItem) -> Result[Representation]:
        try:
            x, sr = load_audio(self.dataset.resolve(item))
        except Exception as exc:  # noqa: BLE001
            return Result.failure(f"cannot read {item.path}: {exc}")
        rec = Recording(x, sr, meta={"dataset": self.dataset.manifest.dataset, "path": item.path})
        return self.analyzer(rec)

    def load(self, index: int) -> Result[PairedExample]:
        off_i, on_i = self.pairs[index]
        off, on = self._rep(off_i), self._rep(on_i)
        if not off.usable or not on.usable:
            return Result.failure(f"analysis failed: off={off.reason!r} on={on.reason!r}")
        a, b = off.value, on.value
        if not a.grid.same_axis(b.grid):
            return Result.failure("paired files have different sample rates / hops")
        diff_s = abs(a.grid.n_frames - b.grid.n_frames) * a.grid.hop_seconds
        cfg = WarpConfig(band_seconds=max(self.min_band_seconds, diff_s + 0.5))
        ca, cb = a.curves["content"].values, b.curves["content"].values
        w = estimate_warp(cb, ca, b.grid, cfg)
        if not w.ok:
            return Result.failure(f"alignment failed: {w.reason}")
        return Result.success(PairedExample(off_i.meta["pair_id"], a, b, w.value, off_i.labels, on_i.labels,
                                            {"alignment_confidence": float(np.mean(w.value.confidence))}))

    def __iter__(self) -> Iterator[Result[PairedExample]]:
        for i in range(len(self)):
            yield self.load(i)
