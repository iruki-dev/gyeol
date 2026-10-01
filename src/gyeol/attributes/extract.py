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

import hashlib
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..align.content import ContentFeatures, MFCCContent
from ..core.consent import Provenance
from ..core.containers import AttributeCurve, AttributeCurves, Recording, Representation
from ..core.grid import DEFAULT_HOP, FrameGrid
from ..core.license import Profile
from ..core.status import Result
from ..frontend.quality import QualityPolicy, assess
from ..frontend.separation import (
    AccompanimentPolicy,
    SeparationPolicy,
    TakePolicy,
    default_separator,
    make_separator,
    unseparated_factor,
    estimate_accompaniment,
    separation_quality,
)
from ..pitch.adapters import default_trackers
from ..pitch.base import PitchTracker
from ..pitch.consensus import ConsensusConfig, consensus
from .pitch_curves import EventConfig, detect_events, notes_from_pitch, pitch_center, vibrato_curves
from .signal import harmonic_noise, loudness, relative_loudness


@dataclass
class AnalysisConfig:
    hop: int = DEFAULT_HOP
    profile: Profile = Profile.COMMERCIAL
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    quality: QualityPolicy = field(default_factory=QualityPolicy)
    events: EventConfig = field(default_factory=EventConfig)
    #: optional learned heads (M3) and the frame encoder that feeds them
    heads: object | None = None  # CalibratedHeads
    feature_encoder: object | None = None  # FrameEncoder or DSPFrameFeatures
    #: revision A1: accompaniment detection and separation-quality policies
    accompaniment: AccompanimentPolicy = field(default_factory=AccompanimentPolicy)
    separation_policy: SeparationPolicy = field(default_factory=SeparationPolicy)
    #: keep the separated vocal in ``rep.meta["separated_audio"]`` (data preparation caches it)
    keep_separated_audio: bool = False
    #: revision D1: user takes with a known accompaniment are separated only when bleed is detected
    take: TakePolicy = field(default_factory=TakePolicy)


SEPARATION_MODES = ("auto", "always", "off")


def _separate(recording: Recording, mode: str, separator, backing, cfg: AnalysisConfig,
              accompaniment_ref: np.ndarray | None = None) -> Result[tuple[np.ndarray, dict, list[str]]]:
    """Decide on separation and run it → (signal to analyse, report, warnings)."""
    from ..frontend.quality import detect_bleed

    x, sr = recording.audio, recording.sr
    report: dict = {"mode": mode, "applied": False}
    warnings: list[str] = []
    if mode == "off":
        report["reason"] = "separation disabled"
        return Result.success((x, report, warnings))
    need = mode == "always" or backing is not None
    if mode == "auto" and accompaniment_ref is not None and backing is None:
        # a headphone take of a known song: separate only when the accompaniment bleeds into the microphone
        b = detect_bleed(x, accompaniment_ref, sr)
        if b.usable:
            bleed, prom = b.value.bleed_db, b.value.lag_prominence
            found = bool(np.isfinite(bleed) and bleed >= cfg.take.bleed_threshold_db and prom >= cfg.take.min_lag_prominence)
            report["bleed"] = {"bleed_db": bleed, "lag_prominence": prom, "lag_s": b.value.lag_s, "threshold_db": cfg.take.bleed_threshold_db,
                               "detected": found}
            report["accompaniment"] = {"may_contain": found, "residual_db": bleed, "reasons": [f"bleed {bleed:.1f} dB"] if found else []}
            need = need or found
        else:
            report["bleed"] = {"detected": None, "reason": b.reason}
            report["accompaniment"] = {"may_contain": None, "reasons": [f"undetermined: {b.reason}"]}
        if not need:
            report["reason"] = "no accompaniment bleed detected"
            return Result.success((x, report, warnings))
        if separator is None or isinstance(separator, str):
            separator_r = make_separator(separator or cfg.take.separator, cfg.profile, accompaniment=accompaniment_ref, accompaniment_sr=sr)
            if not separator_r.ok:
                report["reason"] = f"bleed detected but no separator available: {separator_r.reason}"
                warnings.append("accompaniment bleed detected and not separated: confidences reduced")
                return Result.success((x, report, warnings))
            separator = separator_r.value
    elif mode == "auto":
        acc = estimate_accompaniment(x, sr, cfg.accompaniment)
        if acc.usable:
            a = acc.value
            report["accompaniment"] = {"may_contain": a.may_contain, "residual_db": a.residual_db, "residual_flatness": a.residual_flatness,
                                       "tonal_gaps": a.tonal_gaps, "low_freq_fraction": a.low_freq_fraction, "reasons": a.reasons}
            need = need or a.may_contain
        else:  # cannot tell (e.g. a very short clip): separate if a separator exists, but do not penalise
            report["accompaniment"] = {"may_contain": None, "reasons": [f"undetermined: {acc.reason}"]}
            need = True
    if not need:
        report["reason"] = "no accompaniment detected"
        return Result.success((x, report, warnings))
    if isinstance(separator, str):
        sep = make_separator(separator, cfg.profile, accompaniment=backing, accompaniment_sr=sr)
    else:
        sep = Result.success(separator) if separator is not None else default_separator(cfg.profile, backing, sr)
    if not sep.ok:
        if mode == "always":
            return Result.failure(f"separation required but unavailable: {sep.reason}")
        report["reason"] = f"accompaniment suspected but no separator available: {sep.reason}"
        warnings.append("accompaniment suspected and not separated: confidences reduced")
        return Result.success((x, report, warnings))
    out = sep.value.separate(x, sr)
    if not out.ok:
        if mode == "always":
            return Result.failure(f"separation failed: {out.reason}")
        report["reason"] = f"separator failed: {out.reason}"
        warnings.append("accompaniment suspected and separation failed: confidences reduced")
        return Result.success((x, report, warnings))
    report.update(applied=True, separator=getattr(sep.value, "name", type(sep.value).__name__))
    return Result.success((np.asarray(out.value, float), report, warnings))


def analyze(recording: Recording, *, trackers: Sequence[PitchTracker] | None = None, content: ContentFeatures | None = None,
            backing: np.ndarray | None = None, config: AnalysisConfig | None = None, separation: str = "auto",
            separator=None, accompaniment_ref: np.ndarray | None = None) -> Result[Representation]:
    """Build the interpretable layer for one recording.

    ``separation``: ``"auto"`` (default) separates the vocal first whenever the
    input may contain accompaniment (:func:`~gyeol.frontend.separation.estimate_accompaniment`,
    or a known ``backing`` track); ``"always"`` requires separation (fails if no
    separator is available); ``"off"`` analyses the input as is.  ``separator``
    is any object with ``separate(audio, sr) -> Result`` or a name from
    :data:`~gyeol.frontend.separation.SEPARATORS`; by default the backing
    canceller (when ``backing`` is given) or fetched BS-RoFormer weights.

    ``accompaniment_ref`` (revision D1): the cached accompaniment of the target
    song (:func:`~gyeol.frontend.separation.separate_target`), at this
    recording's rate.  A user take recorded on headphones is then separated in
    ``auto`` mode only when that accompaniment bleeds into the microphone
    (``config.take.bleed_threshold_db``), with the light separator
    ``config.take.separator`` (default: subtract the known accompaniment).

    Returns ``FAILED`` for unusable input (too short, no voiced frames) with
    the reason; quality problems that still allow analysis are reported in
    ``representation.quality`` and reflected in curve confidences.
    """
    cfg = config or AnalysisConfig()
    if separation not in SEPARATION_MODES:
        raise ValueError(f"separation must be one of {SEPARATION_MODES}, got {separation!r}")
    raw, sr = recording.audio, recording.sr
    if len(raw) < int(0.3 * sr):
        return Result.failure("recording shorter than 300 ms")
    if not np.all(np.isfinite(raw)) or np.max(np.abs(raw)) == 0:
        return Result.failure("recording is silent or contains NaN/inf")
    grid = FrameGrid.for_samples(len(raw), sr, cfg.hop)
    timings: dict[str, float] = {}
    clock = [time.perf_counter()]

    def lap(stage: str) -> None:
        now = time.perf_counter()
        timings[stage] = now - clock[0]
        clock[0] = now

    sr_ = _separate(recording, separation, separator, backing, cfg, accompaniment_ref)
    if not sr_.ok:
        return Result.failure(sr_.reason)
    x, sep_report, sep_warnings = sr_.value
    sep_factor = np.ones(grid.n_frames)
    if sep_report.get("applied"):
        ref = backing if backing is not None else accompaniment_ref
        sq = separation_quality(raw, x, sr, grid, backing=ref, separator=sep_report.get("separator", ""), policy=cfg.separation_policy)
        sep_factor = sq.frame_factor
        sep_report.update(residual_db=sq.residual_db, residual_reference=sq.residual_reference, flags=sq.flags,
                          median_frame_sir_db=float(np.median(sq.frame_sir_db)))
    elif sep_report.get("accompaniment", {}).get("may_contain") and separation != "off":
        f = unseparated_factor(sep_report["accompaniment"].get("residual_db"), cfg.separation_policy)
        sep_report["unseparated_factor"] = f
        sep_factor = np.full(grid.n_frames, f)
    lap("separation")
    xc = x - np.mean(x)

    pr = consensus(xc, sr, grid, list(trackers) if trackers else default_trackers(cfg.profile), cfg.consensus)
    if not pr.usable:
        return Result.failure(f"pitch analysis failed: {pr.reason}")
    p = pr.value
    voiced = p.voiced
    if voiced.sum() < 5:
        return Result.failure("no voiced frames: nothing sung was detected")

    lap("pitch")
    q = assess(raw, sr, grid, voiced, backing=backing, policy=cfg.quality, analysis=x)
    lap("quality")
    ff = (q.frame_factor if q.frame_factor is not None else np.ones(grid.n_frames)) * sep_factor
    for k, v in sep_report.get("flags", {}).items():
        q.flags[k] = v
    if sep_report.get("accompaniment", {}).get("may_contain") and not sep_report.get("applied") and separation != "off":
        q.flags["accompaniment_unseparated"] = sep_report.get("reason", "accompaniment suspected")
    f0_conf = p.f0_conf * sep_factor
    clip_ok = 0.0 if "clipping" in q.flags else 1.0

    curves = AttributeCurves(grid)
    add = lambda name, v, c, unit, **kw: curves.add(AttributeCurve(name, v, c, grid, unit, **kw))  # noqa: E731
    cents = p.cents
    add("f0_cents", cents, f0_conf, "cents re A4")
    add("voicing", p.voiced_prob, np.ones(grid.n_frames), "probability")
    add("subharmonic_ratio", p.subharmonic_ratio, p.f0_conf * ff, "ratio")  # ff already carries the separation factor

    loud = loudness(xc, sr, grid)
    add("loudness", loud, ff * (loud > -90), "dBFS(A)")
    add("loudness_rel", relative_loudness(loud, voiced), np.where(voiced, ff, 0.0), "dB re median voiced")

    lap("loudness")
    per, ape, meas = harmonic_noise(xc, sr, grid, p.f0_hz)
    lap("harmonic_noise")
    hn_conf = p.f0_conf * ff * meas * clip_ok
    add("periodic_db", per, hn_conf, "dB")
    add("aperiodic_db", ape, hn_conf, "dB")
    add("aperiodic_ratio", ape - per, hn_conf, "dB")

    notes = notes_from_pitch(cents, voiced, grid)
    center = pitch_center(cents, voiced, grid, segments=notes)
    add("pitch_center", center, f0_conf, "cents re A4")
    rate, extent, vconf = vibrato_curves(cents, center, voiced, grid, f0_conf)
    add("vibrato_rate", rate, vconf, "Hz")
    add("vibrato_extent", extent, vconf, "cents")

    lap("pitch_curves")
    feats = (content or MFCCContent()).extract(xc, sr, grid)
    lap("content")
    add("content", feats, np.ones(grid.n_frames), "normalised", labels=tuple(f"c{i}" for i in range(feats.shape[1])))

    events = detect_events(cents, center, voiced, f0_conf, vconf, notes, grid, cfg.events)
    lap("events")
    rep = Representation(
        grid=grid, curves=curves, recording_id=recording.recording_id, provenance=recording.provenance, events=events,
        quality={"flags": dict(q.flags), "separation": sep_report, "snr_db": None if q.snr is None else q.snr.snr_db,
                 "bandwidth_hz": None if q.bandwidth is None else q.bandwidth.bandwidth_hz,
                 "clipping_fraction": q.clipping.fraction, "bleed_db": None if q.bleed is None else q.bleed.bleed_db},
        meta={"notes": [(n.start, n.end) for n in notes], "trackers": [t.name for t in p.tracks],
              "failed_trackers": p.failed_trackers, "octave_repaired_fraction": float(p.octave_repaired[voiced].mean())},
    )
    rep.meta["analysis_signal"] = "separated vocal" if sep_report.get("applied") else "input"
    rep.meta["profile"] = Profile(cfg.profile).value  # revision C3: outputs carry the license profile
    if recording.provenance is Provenance.USER and recording.owner_id:
        # whose voice this is, without the id in clear (PIPA): rendering checks it against the consent token
        rep.meta["owner_sha256"] = hashlib.sha256(recording.owner_id.encode()).hexdigest()
    if cfg.keep_separated_audio and sep_report.get("applied"):
        rep.meta["separated_audio"] = x
    warnings = [f"{k}: {v}" for k, v in q.flags.items()] + pr.warnings + sep_warnings
    if cfg.heads is not None:
        lr = _learned_curves(rep, xc, sr, cfg, ff, voiced)
        if lr.ok:
            for c in lr.value.values():
                rep.curves.add(c)
        else:
            warnings.append(f"learned heads skipped: {lr.reason}")
        lap("learned_heads")
    rep.meta["timings_s"] = timings
    return Result(pr.status, rep, "", warnings)


def _learned_curves(rep: Representation, x: np.ndarray, sr: int, cfg: AnalysisConfig, ff: np.ndarray, voiced: np.ndarray) -> Result[dict]:
    from ..encoders.frame import DSPFrameFeatures

    enc = cfg.feature_encoder or DSPFrameFeatures()
    if isinstance(enc, DSPFrameFeatures):
        feats = enc.from_representation(rep)
    else:
        r = enc.encode(x, sr, rep.grid)
        if not r.ok:
            return Result.failure(r.reason)
        feats = np.nan_to_num(r.value)
    try:
        return Result.success(cfg.heads.curves(feats, rep.grid, quality_factor=ff, voiced=voiced))
    except RuntimeError as exc:  # e.g. feature dimension mismatch
        return Result.failure(f"heads could not run on these features: {exc}")
