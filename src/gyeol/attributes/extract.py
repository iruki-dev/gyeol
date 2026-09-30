"""Signal-layer analysis: recording → :class:`Representation` (M1, no training).

Pipeline

1. frontend quality checks on the **raw** input (clipping, SNR, bandwidth,
   optional backing-track bleed);
2. pitch consensus (octave-repaired) → ``f0_cents``, ``voicing``;
3. loudness and harmonic/noise analysis → ``loudness``, ``loudness_rel``,
   ``periodic_db``, ``aperiodic_db``, ``aperiodic_ratio``,
   ``subharmonic_ratio``;
4. pitch-derived curves → ``pitch_center``, ``vibrato_rate``,
   ``vibrato_extent``; note segmentation and ornament events;
5. content features for alignment → ``content`` (T, D).

Frontend flags lower confidences: noise-sensitive curves are multiplied by
the per-frame SNR factor, and clipping zeroes aperiodicity confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..align.content import ContentFeatures, MFCCContent
from ..core.containers import AttributeCurve, AttributeCurves, Recording, Representation
from ..core.grid import DEFAULT_HOP, FrameGrid
from ..core.license import Profile
from ..core.status import Result
from ..frontend.quality import QualityPolicy, assess
from ..pitch.adapters import default_trackers
from ..pitch.base import PitchTracker
from ..pitch.consensus import ConsensusConfig, consensus
from ..dsp.notes import segment_notes
from .pitch_curves import EventConfig, detect_events, notes_from_pitch, pitch_center, vibrato_curves
from .signal import harmonic_noise, loudness, relative_loudness


@dataclass
class AnalysisConfig:
    hop: int = DEFAULT_HOP
    profile: Profile = Profile.COMMERCIAL
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    quality: QualityPolicy = field(default_factory=QualityPolicy)
    events: EventConfig = field(default_factory=EventConfig)


def analyze(recording: Recording, *, trackers: Sequence[PitchTracker] | None = None, content: ContentFeatures | None = None,
            backing: np.ndarray | None = None, config: AnalysisConfig | None = None) -> Result[Representation]:
    """Build the interpretable layer for one recording.

    Returns ``FAILED`` for unusable input (too short, no voiced frames) with
    the reason; quality problems that still allow analysis are reported in
    ``representation.quality`` and reflected in curve confidences.
    """
    cfg = config or AnalysisConfig()
    x, sr = recording.audio, recording.sr
    if len(x) < int(0.3 * sr):
        return Result.failure("recording shorter than 300 ms")
    if not np.all(np.isfinite(x)) or np.max(np.abs(x)) == 0:
        return Result.failure("recording is silent or contains NaN/inf")
    grid = FrameGrid.for_samples(len(x), sr, cfg.hop)
    xc = x - np.mean(x)

    pr = consensus(xc, sr, grid, list(trackers) if trackers else default_trackers(cfg.profile), cfg.consensus)
    if not pr.usable:
        return Result.failure(f"pitch analysis failed: {pr.reason}")
    p = pr.value
    voiced = p.voiced
    if voiced.sum() < 5:
        return Result.failure("no voiced frames: nothing sung was detected")

    q = assess(x, sr, grid, voiced, backing=backing, policy=cfg.quality)
    ff = q.frame_factor if q.frame_factor is not None else np.ones(grid.n_frames)
    clip_ok = 0.0 if "clipping" in q.flags else 1.0

    curves = AttributeCurves(grid)
    add = lambda name, v, c, unit, **kw: curves.add(AttributeCurve(name, v, c, grid, unit, **kw))  # noqa: E731
    cents = p.cents
    add("f0_cents", cents, p.f0_conf, "cents re A4")
    add("voicing", p.voiced_prob, np.ones(grid.n_frames), "probability")
    add("subharmonic_ratio", p.subharmonic_ratio, p.f0_conf * ff, "ratio")

    loud = loudness(xc, sr, grid)
    add("loudness", loud, ff * (loud > -90), "dBFS(A)")
    add("loudness_rel", relative_loudness(loud, voiced), np.where(voiced, ff, 0.0), "dB re median voiced")

    per, ape, meas = harmonic_noise(xc, sr, grid, p.f0_hz)
    hn_conf = p.f0_conf * ff * meas * clip_ok
    add("periodic_db", per, hn_conf, "dB")
    add("aperiodic_db", ape, hn_conf, "dB")
    add("aperiodic_ratio", ape - per, hn_conf, "dB")

    raw_notes = segment_notes(cents, voiced, grid.hop_seconds)
    notes = notes_from_pitch(cents, voiced, grid, pitch_center(cents, voiced, grid, segments=raw_notes))
    center = pitch_center(cents, voiced, grid, segments=notes)
    add("pitch_center", center, p.f0_conf, "cents re A4")
    rate, extent, vconf = vibrato_curves(cents, center, voiced, grid, p.f0_conf)
    add("vibrato_rate", rate, vconf, "Hz")
    add("vibrato_extent", extent, vconf, "cents")

    feats = (content or MFCCContent()).extract(xc, sr, grid)
    add("content", feats, np.ones(grid.n_frames), "normalised", labels=tuple(f"c{i}" for i in range(feats.shape[1])))

    events = detect_events(cents, center, voiced, p.f0_conf, vconf, notes, grid, cfg.events)
    rep = Representation(
        grid=grid, curves=curves, recording_id=recording.recording_id, provenance=recording.provenance, events=events,
        quality={"flags": dict(q.flags), "snr_db": None if q.snr is None else q.snr.snr_db,
                 "bandwidth_hz": None if q.bandwidth is None else q.bandwidth.bandwidth_hz,
                 "clipping_fraction": q.clipping.fraction, "bleed_db": None if q.bleed is None else q.bleed.bleed_db},
        meta={"notes": [(n.start, n.end) for n in notes], "trackers": [t.name for t in p.tracks],
              "failed_trackers": p.failed_trackers, "octave_repaired_fraction": float(p.octave_repaired[voiced].mean())},
    )
    warnings = [f"{k}: {v}" for k, v in q.flags.items()] + pr.warnings
    return Result(pr.status, rep, "", warnings)
