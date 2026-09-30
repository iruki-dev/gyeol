"""The analysis engine: audio → :class:`~gyeol.representation.VocalRepresentation`.

Pipeline (research §2 and §4):

1. mono, DC removal; optional singing-voice separation (+ cross-separator
   agreement as a separation-quality proxy);
2. optional device EQ (only with an explicit :class:`DeviceProfile`);
   **no enhancement / denoising by default** (research §2.3);
3. resampling to the 16 kHz analysis rate, 10 ms hop;
4. pitch consensus → voicing, f0, confidence;
5. nuisance side channel N̂ (SNR, bandwidth, codec, clipping, AGC, T60);
6. source, filter, energy groups; note-level vibrato / perturbation /
   glottal / onset descriptors;
7. Korean context tokens (lyrics or TextGrid);
8. validity masks from N̂ + frame conditions, rate reduction of slow
   dimensions, note aggregates;
9. optional residual embedding z.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import signal

from . import __version__
from ._dsp import n_frames, resample, to_mono
from .context.alignment import build_context
from .context.korean import LARYNGEAL_CLASSES, SYLLABLE_POSITIONS
from .features.cepstral import cpps
from .features.energy import frame_energy_db, onset_descriptors, relative_energy
from .features.formants import hawks_miller_bandwidth, track_formants, true_envelope_cepstrum
from .features.glottal import glottal_track
from .features.notes import NoteSpan, segment_notes, vibrato
from .features.perturbation import perturbation
from .features.pitch import PitchTracker, consensus, default_trackers
from .features.source import a1_p0, corrected_harmonic_measures, resonance_tuning
from .features.spectral import APERIODICITY_BANDS, band_aperiodicity, harmonic_peaks, spectrogram, subharmonic_ratio, tilt_measures
from .frontend import nuisance as nuisance_mod
from .frontend.equalization import DeviceProfile
from .frontend.separation import Separator, separation_agreement
from .io import load_audio
from .representation import Note, NuisanceReport, Track, VocalRepresentation
from .residual.base import ResidualEncoder
from .validity import MaskBuilder, ValidityPolicy, finite

#: dimensions aggregated per note (median / IQR of valid frames)
NOTE_AGGREGATES = (
    "f0_cents", "cpps", "h1h2c", "h2h4c", "h1a1c", "h1a3c", "naq", "qoq", "rd", "shr",
    "energy_rel_db", "f1", "f2", "f3", "spr", "alpha_ratio", "hammarberg", "r1_f0", "r1_2f0", "a1_p0",
)
#: dimensions whose aggregates exclude the post-aspirated/fortis window
CONTEXT_EXCLUDED = ("f0_cents", "cpps", "h1h2c", "h2h4c", "h1a1c", "h1a3c", "naq", "qoq", "rd")


@dataclass
class EngineConfig:
    work_sr: int = 16000
    hop_seconds: float = 0.01
    fmin: float = 60.0
    fmax: float = 1600.0
    formant_method: str = "auto"  # auto | lpc | sweep
    lpc_method: str = "wlp"  # wlp | burg
    sweep_above_hz: float = 350.0
    max_formant_hz: float = 5500.0
    compute_glottal: bool = True
    glottal_stride: int = 2
    use_g2pk: bool = False
    policy: ValidityPolicy = field(default_factory=ValidityPolicy)

    def to_dict(self) -> dict:
        return asdict(self)


class Engine:
    """Convert audio into the hybrid, validity-masked vocal representation.

    Parameters
    ----------
    config:
        analysis settings and the :class:`ValidityPolicy`.
    trackers:
        pitch trackers for the consensus (default: pYIN, YIN, SHS).
    separator, cross_separator:
        optional vocal separators.  With both, their stem agreement (SI-SDR)
        becomes the separation-quality estimate.
    device_profile:
        optional :class:`DeviceProfile` for device EQ.
    residual_encoder:
        optional :class:`ResidualEncoder` producing z.
    """

    def __init__(
        self,
        config: EngineConfig | None = None,
        *,
        trackers: Sequence[PitchTracker] | None = None,
        separator: Separator | None = None,
        cross_separator: Separator | None = None,
        device_profile: DeviceProfile | None = None,
        residual_encoder: ResidualEncoder | None = None,
        nuisance_estimators: list | None = None,
    ):
        self.config = config or EngineConfig()
        self.trackers = list(trackers) if trackers else default_trackers(self.config.fmin, self.config.fmax)
        self.separator = separator
        self.cross_separator = cross_separator
        self.device_profile = device_profile
        self.residual_encoder = residual_encoder
        self.nuisance_estimators = nuisance_estimators or []

    # ------------------------------------------------------------------
    def analyze(
        self,
        audio: np.ndarray | str | Path,
        sr: int | None = None,
        *,
        lyrics: str | None = None,
        textgrid: str | Path | None = None,
        notes: Sequence[tuple[float, float]] | None = None,
        source_path: str | Path | None = None,
    ) -> VocalRepresentation:
        """Analyse a recording.

        ``audio`` is a waveform (with ``sr``) or a file path.  ``lyrics``
        (Korean text) or ``textgrid`` (forced alignment) enable context
        tokens; ``notes`` (seconds) overrides automatic note segmentation.
        """
        cfg = self.config
        if isinstance(audio, (str, Path)):
            source_path = source_path or audio
            x, sr = load_audio(audio)
        else:
            if sr is None:
                raise ValueError("sr is required when audio is an array")
            x = to_mono(np.asarray(audio, dtype=float))
        # DC / rumble removal: 20 Hz zero-phase high-pass (−0.05 dB at 60 Hz)
        x = signal.sosfiltfilt(signal.butter(2, 20.0, "high", fs=sr, output="sos"), x - np.mean(x))
        native = x

        # 1–2: separation, EQ ---------------------------------------------
        sep_name, agreement = None, None
        if self.separator is not None:
            vocals = self.separator.separate(x, sr)
            sep_name = self.separator.name
            if self.cross_separator is not None:
                other = self.cross_separator.separate(x, sr)
                agreement = separation_agreement(vocals, other)
                sep_name = f"{sep_name}+{self.cross_separator.name}"
            x = vocals
        if self.device_profile is not None:
            x = self.device_profile.apply(x, sr)

        # 3: analysis grid ----------------------------------------------------
        wsr = cfg.work_sr
        xw = resample(x, sr, wsr)
        hop = int(round(cfg.hop_seconds * wsr))
        hop_s = hop / wsr
        n = n_frames(len(xw), hop)

        # 4: pitch ------------------------------------------------------------
        pc = consensus(xw, wsr, hop, n, self.trackers)
        voiced, f0 = pc.voiced, pc.f0

        # 5: nuisance side channel ------------------------------------------
        nuis, frame_snr = nuisance_mod.estimate(native, sr, xw, wsr, hop, n, voiced, source_path, self.nuisance_estimators)
        nuis.separation_used = sep_name
        nuis.separation_agreement_db = agreement
        nuis.device_eq_applied = self.device_profile is not None
        nuis.device_highpass_hz = self.device_profile.highpass_hz if self.device_profile else None

        # 6: features ---------------------------------------------------------
        spec = spectrogram(xw, wsr, hop, n)
        harm = harmonic_peaks(spec, f0)
        ft = track_formants(xw, wsr, hop, n, f0, harm, method=cfg.formant_method, lpc_method=cfg.lpc_method,
                            sweep_above_hz=cfg.sweep_above_hz, max_formant=cfg.max_formant_hz)
        hm = corrected_harmonic_measures(harm, f0, ft.freq, wsr)
        ap, hnr, ap_ok = band_aperiodicity(spec, f0)
        tilt = tilt_measures(spec)
        shr = subharmonic_ratio(spec, f0)
        cp = cpps(xw, wsr, hop, n, f0, fmin=cfg.fmin, fmax=cfg.fmax)
        env = true_envelope_cepstrum(xw, wsr, hop, n, f0)
        glot = glottal_track(xw, wsr, hop, f0, stride=cfg.glottal_stride) if cfg.compute_glottal else {k: np.full(n, np.nan) for k in ("naq", "qoq", "rd")}
        energy_db = frame_energy_db(xw, hop, n, int(0.04 * wsr))
        rel_energy = relative_energy(energy_db, voiced, hop_s)
        nasal, p0_idx = a1_p0(harm, f0, ft.freq[:, 0])
        tuning = resonance_tuning(f0, ft.freq[:, 0], ft.bw[:, 0])

        # notes
        if notes is not None:
            spans = [NoteSpan(int(round(a / hop_s)), min(n, int(round(b / hop_s)))) for a, b in notes]
        else:
            spans = segment_notes(pc.cents, voiced, hop_s)
        note_secs = [(s.start * hop_s, s.end * hop_s) for s in spans]

        # 7: context ----------------------------------------------------------
        context = build_context(n, 1.0 / hop_s, note_secs, lyrics=lyrics, textgrid=textgrid, use_g2pk=cfg.use_g2pk)

        # 8: validity ---------------------------------------------------------
        tracks = self._build_tracks(
            n=n, hop_s=hop_s, pc=pc, nuis=nuis, frame_snr=frame_snr, energy_rel=rel_energy, cp=cp, ap=ap, hnr=hnr,
            ap_ok=ap_ok, hm=hm, glot=glot, tilt=tilt, shr=shr, ft=ft, env=env, tuning=tuning, nasal=nasal,
            p0_idx=p0_idx, context=context,
        )
        exclusion = self._context_exclusion(context, n, hop_s)
        note_objs = self._notes(spans, hop_s, pc, tracks, exclusion, x, sr, nuis, frame_snr, energy_db, ap, hm, context)

        residual = None
        if self.residual_encoder is not None:
            residual = self.residual_encoder.encode(xw, wsr, tracks)

        return VocalRepresentation(
            sample_rate=wsr,
            hop_seconds=hop_s,
            duration=len(xw) / wsr,
            tracks=tracks,
            notes=note_objs,
            nuisance=nuis,
            context=context,
            residual=residual,
            config=cfg.to_dict(),
            version=__version__,
        )

    # ------------------------------------------------------------------
    def _build_tracks(self, *, n, hop_s, pc, nuis: NuisanceReport, frame_snr, energy_rel, cp, ap, hnr, ap_ok, hm, glot, tilt,
                      shr, ft, env, tuning, nasal, p0_idx, context) -> dict[str, Track]:
        pol = self.config.policy
        rate = 1.0 / hop_s
        voiced = pc.voiced
        f0 = pc.f0
        bw = nuis.effective_bandwidth_hz or 0.0
        lossy = nuis.codec_suspected and pol.reject_lossy_for_noise_measures
        clip_ok = nuis.clipping_fraction <= pol.max_clipping_fraction
        sep_ok = nuis.separation_agreement_db is None or nuis.separation_agreement_db >= pol.min_separation_agreement_db
        snr_ok = frame_snr >= pol.min_snr_db
        conf_ok = pc.confidence >= pol.min_f0_confidence
        eq_conf = 1.0 if nuis.device_eq_applied else pol.no_eq_confidence
        tracks: dict[str, Track] = {}

        def add(name, values, unit, mb: MaskBuilder, confidence=None, track_rate=rate):
            valid, reasons = mb.build()
            tracks[name] = Track(name, values, valid, track_rate, unit, confidence, reasons)

        def base(extra_voiced=True) -> MaskBuilder:
            mb = MaskBuilder(n)
            if extra_voiced:
                mb.require("unvoiced", voiced)
            return mb

        # pitch
        add("voicing_prob", pc.voiced_prob, "0-1", MaskBuilder(n).require("separation_quality", sep_ok))
        for name, vals, unit in (("f0_hz", f0, "Hz"), ("f0_cents", pc.cents, "cents re A4")):
            add(name, vals, unit, base().require("low_f0_confidence", conf_ok).require("separation_quality", sep_ok), pc.confidence)
        # energy
        add("energy_rel_db", energy_rel, "dB re phrase median",
            MaskBuilder(n).require("outside_phrase", np.isfinite(energy_rel)).require("agc", not nuis.agc_suspected))
        # CPPS
        add("cpps", cp, "dB", base().require("low_snr", snr_ok).require("codec", not lossy).require("f0_ceiling", ~(f0 > pol.cpps_max_f0)),
            pc.confidence)
        # aperiodicity / HNR: bands beyond the effective bandwidth become NaN
        above_bw = np.array([lo >= bw for lo, _ in APERIODICITY_BANDS])
        ap = ap.copy()
        hnr = hnr.copy()
        ap[:, above_bw] = np.nan
        hnr[:, above_bw] = np.nan
        for name, vals in (("band_aperiodicity", ap), ("band_hnr", hnr)):
            mb = (base().require("unmeasurable_f0", ap_ok).require("low_snr", snr_ok).require("codec", not lossy)
                  .require("separation_quality", sep_ok).require("clipping", clip_ok)
                  .require("bandwidth", bw >= pol.min_bandwidth_aperiodicity_hz))
            add(name, vals, "dB", mb, pc.confidence)
        # formants
        hm_bw = hawks_miller_bandwidth(ft.freq, f0[:, None])
        resolvable = [np.isfinite(ft.freq[:, i]) & (ft.freq[:, i] >= pol.formant_min_f0_ratio * f0) for i in range(4)]
        bw_ok = bw >= pol.min_bandwidth_spectral_hz
        # harmonic differences
        f1, b1 = ft.freq[:, 0], hm_bw[:, 0]
        near = (np.abs(f1 - f0) <= pol.h1h2_f1_bandwidths * b1) | (np.abs(f1 - 2 * f0) <= pol.h1h2_f1_bandwidths * b1)
        hp = nuis.device_highpass_hz
        hp_ok = np.ones(n, bool) if hp is None else ~(f0 < 2 * hp)
        for name in ("h1h2c", "h2h4c", "h1a1c", "h1a3c"):
            mb = (base().require("formants_unresolved", resolvable[0] & resolvable[1]).require("f1_near_harmonic", ~near)
                  .require("device_highpass", hp_ok).require("clipping", clip_ok).require("bandwidth", bw_ok)
                  .require("not_measurable", np.isfinite(hm[name])))
            if name == "h1a3c":
                mb.require("f3_unresolved", resolvable[2])
            add(name, hm[name], "dB", mb, pc.confidence * eq_conf)
        # glottal
        steady = np.abs(np.gradient(np.nan_to_num(pc.cents))) < 30.0
        for name in ("naq", "qoq", "rd"):
            mb = (base().require("not_measurable", np.isfinite(glot[name])).require("f0_ceiling", ~(f0 > pol.glottal_max_f0))
                  .require("codec", not nuis.codec_suspected).require("unsteady", steady).require("clipping", clip_ok))
            add(name, glot[name], "dimensionless", mb, pc.confidence)
        # tilt set and SPR (EQ-sensitive) → 25 Hz
        for name in ("alpha_ratio", "hammarberg", "lh_ratio", "spr"):
            need = pol.min_bandwidth_aperiodicity_hz if name == "lh_ratio" else pol.min_bandwidth_spectral_hz
            mb = base().require("bandwidth", bw >= need).require("clipping", clip_ok)
            add(name, tilt[name], "dB", mb, np.full(n, eq_conf))
        add("shr", shr, "ratio", base().require("low_snr", snr_ok).require("codec", not lossy).require("not_measurable", np.isfinite(shr)))
        # formant tracks → 50 Hz
        for i in range(4):
            mb = base().require("formant_unresolved", resolvable[i]).require("bandwidth", bw_ok)
            add(f"f{i + 1}", ft.freq[:, i], "Hz", mb, pc.confidence)
            mb = base().require("formant_unresolved", resolvable[i]).require("bandwidth", bw_ok)
            add(f"b{i + 1}", ft.bw[:, i], "Hz", mb, pc.confidence)
        add("envelope", env, "cepstrum (ln amplitude)", base().require("bandwidth", bw_ok).require("not_measurable", finite(env)))
        for name in ("r1_f0", "r1_2f0"):
            add(name, tuning[name], "ratio", base().require("formant_unresolved", resolvable[0]))
        mb = (base().require("formant_unresolved", resolvable[0]).require("p0_on_low_harmonic", p0_idx >= pol.nasality_min_p0_index)
              .require("not_measurable", np.isfinite(nasal)))
        if context is not None:
            mb.require("not_vowel", context.syllable_position == SYLLABLE_POSITIONS.index("nucleus"))
        add("a1_p0", nasal, "dB", mb)

        # rate reduction of slow dimensions
        for name in ("f1", "f2", "f3", "f4", "b1", "b2", "b3", "b4", "envelope"):
            tracks[name] = decimate(tracks[name], 2)
        for name in ("alpha_ratio", "hammarberg", "lh_ratio", "spr"):
            tracks[name] = decimate(tracks[name], 4)
        return tracks

    # ------------------------------------------------------------------
    def _context_exclusion(self, context, n: int, hop_s: float) -> np.ndarray:
        excl = np.zeros(n, dtype=bool)
        if context is None:
            return excl
        for cls_name, ms in self.config.policy.context_exclusion_ms.items():
            k = LARYNGEAL_CLASSES.index(cls_name)
            excl |= (context.laryngeal_class == k) & (context.time_since_onset_ms < ms)
        return excl

    def _notes(self, spans, hop_s, pc, tracks, exclusion, x, sr, nuis, frame_snr, energy_db, ap, hm, context) -> list[Note]:
        pol = self.config.policy
        out: list[Note] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ap_hi = np.nanmean(ap[:, 2:], axis=1)
            for sp in spans:
                s, e = sp.start, sp.end
                note = Note(start=s * hop_s, end=e * hop_s)
                # aggregates
                for name in NOTE_AGGREGATES:
                    tr = tracks.get(name)
                    if tr is None:
                        continue
                    ratio = rate_ratio(tr.rate, hop_s)
                    a, b = int(s / ratio), max(int(s / ratio) + 1, int(np.ceil(e / ratio)))
                    vals = tr.values[a:b]
                    ok = tr.valid[a:b] & np.isfinite(vals)
                    if name in CONTEXT_EXCLUDED and ratio == 1:
                        ok &= ~exclusion[a:b]
                    note.valid[name] = bool(ok.sum() >= 3)
                    if ok.any():
                        q1, med, q3 = np.percentile(vals[ok], [25, 50, 75])
                        note.features[f"{name}_median"] = float(med)
                        note.features[f"{name}_iqr"] = float(q3 - q1)
                    note.features[f"{name}_coverage"] = float(ok.mean()) if len(ok) else 0.0
                # vibrato
                vb = vibrato(pc.cents[s:e], hop_s, min_cycles=pol.vibrato_min_cycles)
                has_vib = vb is not None
                if has_vib:
                    note.features.update(vibrato_rate_hz=vb.rate_hz, vibrato_extent_cents=vb.extent_cents,
                                         vibrato_rate_cv=vb.rate_cv, vibrato_extent_cv=vb.extent_cv, vibrato_cycles=vb.n_cycles)
                note.valid["vibrato_rate_hz"] = has_vib
                note.valid["vibrato_extent_cents"] = has_vib and not nuis.agc_suspected
                note.valid["vibrato_rate_cv"] = note.valid["vibrato_extent_cv"] = bool(has_vib and np.isfinite(vb.rate_cv))
                # perturbation (native-rate signal)
                vib_small = (not has_vib) or vb.extent_cents < pol.perturbation_max_vibrato_cents
                pt = None
                if (e - s) * hop_s >= pol.perturbation_min_seconds:
                    f0_native = pc.f0  # frame grid identical in seconds
                    hop_native = int(round(hop_s * sr))
                    pt = perturbation(x, sr, f0_native, hop_native, s, e)
                if pt is not None:
                    note.features.update(jitter_local=pt.jitter_local, shimmer_local=pt.shimmer_local, shimmer_db=pt.shimmer_db)
                note_snr = float(np.median(frame_snr[s:e])) if e > s else -np.inf
                pert_ok = (pt is not None and vib_small and note_snr >= pol.min_snr_db and sr >= pol.perturbation_min_sr
                           and not (nuis.codec_suspected and pol.reject_lossy_for_noise_measures))
                note.valid["jitter_local"] = note.valid["shimmer_local"] = bool(pert_ok)
                # onset
                ons = onset_descriptors(s, e, energy_db, pc.cents, ap_hi, hm["h1h2c"], hop_s,
                                        vibrato_extent_cents=vb.extent_cents if has_vib else 0.0)
                note.features.update(onset_rise_ms=ons.rise_time_ms, onset_f0_settle_ms=ons.f0_settle_ms,
                                     onset_aperiodicity_db=ons.pre_voicing_aperiodicity_db, onset_h1h2c_db=ons.early_h1h2c_db)
                ap_valid = tracks["band_aperiodicity"].valid
                h_valid = tracks["h1h2c"].valid
                note.valid["onset_rise_ms"] = bool(np.isfinite(ons.rise_time_ms))
                note.valid["onset_f0_settle_ms"] = bool(np.isfinite(ons.f0_settle_ms))
                note.valid["onset_aperiodicity_db"] = bool(np.isfinite(ons.pre_voicing_aperiodicity_db) and ap_valid[s : s + 3].any())
                note.valid["onset_h1h2c_db"] = bool(np.isfinite(ons.early_h1h2c_db) and h_valid[s : s + 5].any())
                # context
                if context is not None:
                    idx = context.syllable_index[s:e]
                    idx = idx[idx >= 0]
                    if idx.size:
                        k = int(np.bincount(idx).argmax())
                        syl = context.syllables[k]
                        note.syllable = syl["text"]
                        note.features["laryngeal_class"] = float(LARYNGEAL_CLASSES.index(syl["laryngeal_class"]))
                        note.features["nasal_coda"] = float(syl["nasal_coda"])
                out.append(note)
        return out


# ---------------------------------------------------------------------------


def rate_ratio(rate: float, hop_s: float) -> int:
    return max(1, int(round(1.0 / (rate * hop_s))))


def decimate(tr: Track, factor: int) -> Track:
    """Block-median of valid frames; a block is valid if ≥ half its frames are."""
    n = tr.values.shape[0]
    m = int(np.ceil(n / factor))
    pad = m * factor - n
    vals = tr.masked()
    if vals.ndim == 1:
        blocks = np.pad(vals, (0, pad), constant_values=np.nan).reshape(m, factor)
    else:
        blocks = np.pad(vals, ((0, pad), (0, 0)), constant_values=np.nan).reshape(m, factor, -1)
    vmask = np.pad(tr.valid, (0, pad)).reshape(m, factor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmedian(blocks, axis=1)
        conf = None if tr.confidence is None else np.nanmean(np.pad(tr.confidence, (0, pad), constant_values=np.nan).reshape(m, factor), axis=1)
    valid = (vmask.sum(axis=1) * 2 >= factor) & (np.isfinite(out) if out.ndim == 1 else np.all(np.isfinite(out), axis=1))
    return Track(tr.name, out, valid, tr.rate / factor, tr.unit, conf, tr.invalid_reasons)


def analyze(audio, sr: int | None = None, **kwargs) -> VocalRepresentation:
    """Convenience wrapper: ``Engine().analyze(...)``."""
    return Engine().analyze(audio, sr, **kwargs)


__all__ = ["Engine", "EngineConfig", "analyze", "decimate"]
