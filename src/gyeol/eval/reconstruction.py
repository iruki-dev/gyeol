"""Reconstruction metrics, vocoder benchmark and re-encoding consistency (evaluation §2).

* :func:`vocoder_benchmark` — run an analysis–synthesis function over items
  and report, **per technique** (breathy / rough / falsetto / …) and **per
  frame class** (voiced vs consonant = unvoiced but audible):
  multi-resolution log-spectral distance, f0 RMSE (cents) and voicing error
  from signal-layer re-analysis, and aperiodic-ratio error.
* :func:`reencoding_consistency` — curves of ``analyze(decode(x))`` against
  ``analyze(x)`` on frames confident in both (the non-differentiable
  counterpart of the training-time r consistency).

A listening-test protocol template lives in ``docs/protocols/listening_test.md``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ..core.consent import Provenance
from ..core.containers import Recording, Representation
from ..core.status import Result

EPS = 1e-10


def log_spectral_distance(x: np.ndarray, y: np.ndarray, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    """Per-frame LSD (dB) between two signals, frames centred on i·hop."""
    n = min(len(x), len(y))

    def spec(a):
        a = np.pad(a[:n], (n_fft // 2, n_fft // 2))
        idx = np.arange(n_fft)[None, :] + hop * np.arange(1 + n // hop)[:, None]
        idx = np.minimum(idx, len(a) - 1)
        return 10 * np.log10(np.abs(np.fft.rfft(a[idx] * np.hanning(n_fft), axis=1)) ** 2 + EPS)

    return np.sqrt(np.mean((spec(x) - spec(y)) ** 2, axis=1))


@dataclass
class BenchmarkItem:
    audio: np.ndarray
    sr: int
    labels: dict = field(default_factory=dict)  # e.g. {"technique": "breathy"}


@dataclass
class CategoryReport:
    n_items: int
    lsd_voiced_db: float
    lsd_consonant_db: float
    f0_rmse_cents: float
    voicing_error: float
    aperiodic_mae_db: float


def vocoder_benchmark(resynth: Callable[[np.ndarray, int], np.ndarray], items: Sequence[BenchmarkItem],
                      analyzer: Callable[[Recording], Result[Representation]] | None = None, group_by: str = "technique") -> dict[str, CategoryReport]:
    from ..attributes.extract import analyze

    analyzer = analyzer or analyze
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for it in items:
        y = np.asarray(resynth(it.audio, it.sr), float)
        a = analyzer(Recording(it.audio, it.sr, Provenance.SYNTHETIC))
        b = analyzer(Recording(np.pad(y, (0, max(0, len(it.audio) - len(y))))[: len(it.audio)], it.sr, Provenance.SYNTHETIC))
        cat = str(it.labels.get(group_by, "unlabelled"))
        if not a.usable or not b.usable:
            acc[cat]["failed"].append(1.0)
            continue
        ra, rb = a.value, b.value
        lsd = log_spectral_distance(it.audio, y, hop=ra.grid.hop)
        T = min(len(lsd), ra.grid.n_frames, rb.grid.n_frames)
        va = np.isfinite(ra.curves["f0_cents"].values[:T])
        vb = np.isfinite(rb.curves["f0_cents"].values[:T])
        loud = np.nan_to_num(ra.curves["loudness_rel"].values[:T], nan=-99)
        consonant = ~va & (loud > -40)
        both = va & vb
        d = ra.curves["f0_cents"].values[:T][both] - rb.curves["f0_cents"].values[:T][both]
        apa, apb = ra.curves["aperiodic_ratio"], rb.curves["aperiodic_ratio"]
        okap = (apa.confidence[:T] > 0) & (apb.confidence[:T] > 0)
        acc[cat]["lsd_v"].extend(lsd[:T][va].tolist())
        acc[cat]["lsd_c"].extend(lsd[:T][consonant].tolist())
        acc[cat]["f0"].extend((d**2).tolist())
        acc[cat]["vuv"].append(float(np.mean(va != vb)))
        acc[cat]["ap"].extend(np.abs(apa.values[:T][okap] - apb.values[:T][okap]).tolist())
        acc[cat]["n"].append(1.0)
    mean = lambda v: float(np.mean(v)) if len(v) else float("nan")  # noqa: E731
    return {cat: CategoryReport(int(sum(d["n"])), mean(d["lsd_v"]), mean(d["lsd_c"]), float(np.sqrt(mean(d["f0"]))),
                                mean(d["vuv"]), mean(d["ap"])) for cat, d in acc.items()}


CONSISTENCY_CURVES = ("pitch_center", "loudness_rel", "aperiodic_ratio", "vibrato_extent")


def reencoding_consistency(original: Representation, reencoded: Representation, curves: Sequence[str] = CONSISTENCY_CURVES,
                           min_confidence: float = 0.3) -> dict[str, float]:
    """Median |c(decode(x)) − c(x)| per curve over frames confident in both (NaN if none)."""
    out = {}
    T = min(original.grid.n_frames, reencoded.grid.n_frames)
    for n in curves:
        a, b = original.curves[n], reencoded.curves[n]
        ok = (a.confidence[:T] >= min_confidence) & (b.confidence[:T] >= min_confidence)
        ok &= np.isfinite(a.values[:T]) & np.isfinite(b.values[:T])
        out[n] = float(np.median(np.abs(a.values[:T][ok] - b.values[:T][ok]))) if ok.any() else float("nan")
    return out
