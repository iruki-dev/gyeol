"""Human-annotated real-singing pitch sets for *evaluation* (revision D3): Vocadito and MIR-1K.

Both adapters only build :class:`~gyeol.data.manifest.Manifest`\\ s (datasets ``vocadito`` and ``mir1k``; their
listed licenses are in :mod:`gyeol.core.assets`).

Layouts (as distributed):

* **Vocadito** — ``Audio/vocadito_<n>.wav`` and ``Annotations/F0/vocadito_<n>_f0.csv`` (``time, frequency`` in Hz;
  0 = unvoiced).  Any CSV whose name contains ``f0`` and starts with the audio file's stem is accepted.
* **MIR-1K** — ``Wavfile/<singer>_<song>_<n>.wav`` (16 kHz stereo: accompaniment left, voice right) and
  ``PitchLabel/<same stem>.pv`` (one MIDI semitone value per 20 ms frame, the first centred at 20 ms; 0 = unvoiced).
  The voice channel is evaluated by default; ``channel=None`` evaluates the mixture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.status import Result, Status
from .manifest import Manifest, ManifestItem

AUDIO_EXT = (".wav", ".flac")


def scan_vocadito(root: str | Path) -> Result[Manifest]:
    root = Path(root)
    audio = sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXT)
    csvs = sorted(p for p in root.rglob("*.csv") if "f0" in p.stem.lower())
    items, missing = [], []
    for a in audio:
        # "vocadito_1_f0" belongs to vocadito_1, not "vocadito_10_f0": the stem must continue with a non-digit
        ann = [c for c in csvs if c.stem.startswith(a.stem) and not c.stem[len(a.stem):][:1].isdigit()]
        if not ann:
            missing.append(a.name)
            continue
        items.append(ManifestItem(str(a.relative_to(root)), a.stem, {}, {"f0": str(ann[0].relative_to(root)), "f0_format": "csv_time_hz"}))
    if not items:
        return Result.failure(f"no Vocadito audio with an f0 annotation under {root}")
    return Result(Status.OK, Manifest("vocadito", str(root), items), "",
                  [f"{len(missing)} audio files without f0 annotation"] if missing else [])


def scan_mir1k(root: str | Path, channel: int | None = 1) -> Result[Manifest]:
    root = Path(root)
    items = []
    for a in sorted(root.rglob("*.wav")):
        pv = next(iter(sorted(root.rglob(f"{a.stem}.pv"))), None)
        if pv is None:
            continue
        items.append(ManifestItem(str(a.relative_to(root)), a.stem.split("_")[0], {},
                                  {"f0": str(pv.relative_to(root)), "f0_format": "mir1k_pv", "channel": channel}))
    if not items:
        return Result.failure(f"no MIR-1K wav files with PitchLabel .pv annotations under {root}")
    return Result.success(Manifest("mir1k", str(root), items))


def read_reference_f0(root: str | Path, item: ManifestItem) -> tuple[np.ndarray, np.ndarray]:
    """(times s, f0 Hz with 0 = unvoiced) of an annotated item."""
    path = Path(root) / item.meta["f0"]
    fmt = item.meta.get("f0_format", "csv_time_hz")
    if fmt == "mir1k_pv":
        semis = np.array([float(x) for x in path.read_text().split()], float)
        t = 0.02 * (np.arange(len(semis)) + 1)
        return t, np.where(semis > 0, 440.0 * 2 ** ((semis - 69.0) / 12.0), 0.0)
    rows = []
    for line in path.read_text().splitlines():
        parts = [p for p in line.replace("\t", ",").replace(" ", ",").split(",") if p]
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except (ValueError, IndexError):
            continue  # header or blank line
    a = np.array(rows, float).reshape(-1, 2)
    return a[:, 0], np.maximum(a[:, 1], 0.0)


def load_item_audio(path: str | Path, item: ManifestItem) -> tuple[np.ndarray, int]:
    import soundfile as sf

    x, sr = sf.read(str(path), always_2d=True, dtype="float64")
    ch = item.meta.get("channel")
    if ch is not None and x.shape[1] > ch:
        return x[:, ch], int(sr)
    return x.mean(axis=1), int(sr)


__all__ = ["load_item_audio", "read_reference_f0", "scan_mir1k", "scan_vocadito"]
