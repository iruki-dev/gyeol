"""Machine-readable specification of the representation.

Mirrors the "Specification Table of the Recommended Representation" of the
research: every dimension declares its unit, rate, estimator, validity
condition, what it omits and how it is verified.  ``hypothesis`` marks
choices the research labels UNVERIFIED HYPOTHESIS.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DimensionSpec:
    name: str
    group: str
    unit: str
    rate: str
    estimator: str
    validity: str
    omits: str
    verification: str
    hypothesis: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


_S = DimensionSpec

DIMENSIONS: dict[str, DimensionSpec] = {
    d.name: d
    for d in [
        _S("voicing_prob", "pitch", "0–1", "100 Hz", "mean of pYIN / YIN / SHS voicing probabilities",
           "post-separation voice-to-accompaniment ≥ 0 dB", "voice quality of unvoiced segments",
           "frame F1 vs reference mic; error vs SDR curve"),
        _S("f0_hz", "pitch", "Hz", "100 Hz", "median of pYIN / YIN / SHS (pluggable RMVPE, SwiftF0)",
           "voiced by majority vote; confidence from inter-tracker spread", "subharmonic structure",
           "ICC / Bland–Altman across devices; gross error rate vs SNR"),
        _S("f0_cents", "pitch", "cents re A4", "100 Hz", "as f0_hz",
           "as f0_hz; post-격음/경음 window excluded from pitch aggregates", "subharmonic structure",
           "as f0_hz"),
        _S("energy_rel_db", "energy", "dB re phrase median", "100 Hz", "RMS of the separated vocal",
           "no AGC pumping flag", "absolute SPL", "correlation with reference-mic dynamics"),
        _S("cpps", "source", "dB", "100 Hz", "Hillenbrand-type smoothed cepstrum, f0-tracked quefrency search",
           "frame SNR ≥ 30 dB; no transmission codec; f0 ≤ provisional ceiling", "spectral location of noise",
           "ICC ≥ 0.9, bias < MDC across devices; Test S vs clean stem",
           "the CPPS f0 ceiling (default 700 Hz) must be calibrated (§5.iii)"),
        _S("band_aperiodicity", "source", "dB (5 bands)", "100 Hz", "harmonic-comb noise-density estimate per band",
           "frame SNR ≥ 30 dB; separation agreement above threshold; no lossy codec; bandwidth ≥ 8 kHz for top bands",
           "temporal fine structure of noise", "error vs SNR / codec / SDR curves; Test S",
           "mask-based separators may bias it (research §2.2)"),
        _S("band_hnr", "source", "dB (5 bands)", "100 Hz", "harmonic / noise power per band",
           "as band_aperiodicity", "temporal fine structure of noise", "as band_aperiodicity"),
        _S("h1h2c", "source", "dB", "100 Hz", "harmonic amplitudes + Iseli–Alwan correction (F1, F2)",
           "device high-pass ≪ f0; |F1 − f0| > B1 and |F1 − 2f0| > B1; valid F1/F2; context-tagged",
           "full glottal pulse shape", "EQ on/off ablation; EGG-referenced high-f0 test; Korean context ICC",
           "|F1 − f0| > B1 threshold to be calibrated with EGG (§5.iii)"),
        _S("h2h4c", "source", "dB", "100 Hz", "as h1h2c", "as h1h2c", "full glottal pulse shape", "as h1h2c"),
        _S("h1a1c", "source", "dB", "100 Hz", "as h1h2c (A1 corrected for F1, F2)", "as h1h2c", "full glottal pulse shape", "as h1h2c"),
        _S("h1a3c", "source", "dB", "100 Hz", "as h1h2c (A3 corrected for F1–F3)", "as h1h2c; valid F3", "full glottal pulse shape", "as h1h2c"),
        _S("naq", "source", "dimensionless", "per steady segment", "IAIF inverse filtering (QCP pluggable)",
           "steady phonation (≥ 4 periods); phase-intact channel; f0 ≤ provisional ceiling",
           "waveform detail beyond LF summary", "EGG contact-quotient correlation; codec ablation",
           "inverse filtering above ≈500 Hz f0 unvalidated"),
        _S("qoq", "source", "dimensionless", "per steady segment", "as naq", "as naq", "as naq", "as naq", "as naq"),
        _S("rd", "source", "dimensionless", "per steady segment", "NAQ / 0.11 (Fant 1995 relation)", "as naq", "as naq", "as naq", "as naq"),
        _S("alpha_ratio", "source", "dB", "25 Hz", "band energies after EQ", "bandwidth ≥ 5 kHz; EQ applied (confidence penalty otherwise)",
           "formant-specific detail", "multi-device Bland–Altman (known device-sensitive)"),
        _S("hammarberg", "source", "dB", "25 Hz", "band maxima after EQ", "as alpha_ratio", "formant-specific detail", "as alpha_ratio"),
        _S("lh_ratio", "source", "dB", "25 Hz", "energy < 4 kHz vs ≥ 4 kHz", "as alpha_ratio; bandwidth ≥ 8 kHz", "formant-specific detail", "as alpha_ratio"),
        _S("shr", "source", "ratio", "100 Hz", "subharmonic-to-harmonic ratio at tracked f0 (Sun 2002, simplified)",
           "frame SNR ≥ 30 dB; no lossy codec", "cycle-level waveform shape", "codec / SNR degradation curves; coach 수리성 probe"),
        _S("f1", "filter", "Hz", "50 Hz", "STE-WLP (f0 < 350 Hz) / harmonic-pooled vibrato-sweep all-pole fit (above)",
           "F1 ≥ 1.5·f0; bandwidth ≥ 5 kHz", "anti-resonances except nasal index", "synthetic-vowel accuracy vs f0; cross-device ICC",
           "vibrato-sweep estimate and the 1.5·f0 resolvability gate are provisional"),
        _S("f2", "filter", "Hz", "50 Hz", "as f1", "F2 ≥ 1.5·f0", "as f1", "as f1"),
        _S("f3", "filter", "Hz", "50 Hz", "as f1", "F3 ≥ 1.5·f0", "as f1", "as f1"),
        _S("f4", "filter", "Hz", "50 Hz", "as f1", "F4 ≥ 1.5·f0; bandwidth ≥ 5 kHz", "as f1", "as f1"),
        _S("b1", "filter", "Hz", "50 Hz", "LPC pole radius", "as f1", "as f1", "as f1"),
        _S("b2", "filter", "Hz", "50 Hz", "LPC pole radius", "as f2", "as f1", "as f1"),
        _S("b3", "filter", "Hz", "50 Hz", "LPC pole radius", "as f3", "as f1", "as f1"),
        _S("b4", "filter", "Hz", "50 Hz", "LPC pole radius", "as f4", "as f1", "as f1"),
        _S("envelope", "filter", "24 cepstral coeffs", "50 Hz", "true-envelope cepstrum (Röbel & Rodet 2005)",
           "voiced; bandwidth ≥ 5 kHz", "detail > 8 kHz", "resynthesis MUSHRA; resonance probe"),
        _S("spr", "filter", "dB", "25 Hz (aggregated per note)", "max dB 2–4 kHz − max dB 0–2 kHz",
           "EQ; bandwidth ≥ 5 kHz", "singer's formant fine structure", "device / codec ablation"),
        _S("r1_f0", "filter", "ratio", "100 Hz (per note)", "|F1 − f0| / B1", "F1 valid", "— (derived)", "coach resonance probe"),
        _S("r1_2f0", "filter", "ratio", "100 Hz (per note)", "|F1 − 2f0| / B1", "F1 valid", "— (derived)", "coach resonance probe"),
        _S("a1_p0", "filter", "dB", "100 Hz (per vowel)", "Chen (1997) A1 − P0", "vowel nucleus; P0 harmonic index ≥ 3; nasal-coda tagged",
           "nasal airflow", "context-controlled ICC"),
    ]
}

NOTE_DIMENSIONS: dict[str, DimensionSpec] = {
    d.name: d
    for d in [
        _S("vibrato_rate_hz", "pitch", "Hz", "per note", "autocorrelation + sinusoid fit to detrended f0",
           "≥ 2 cycles; no AGC flag", "amplitude–frequency vibrato coupling", "MDC across devices; coach probe"),
        _S("vibrato_extent_cents", "pitch", "cents (semi-extent)", "per note", "as vibrato_rate_hz", "as vibrato_rate_hz",
           "as vibrato_rate_hz", "as vibrato_rate_hz"),
        _S("vibrato_rate_cv", "pitch", "CV", "per note", "half-cycle durations", "as vibrato_rate_hz", "", "as vibrato_rate_hz"),
        _S("vibrato_extent_cv", "pitch", "CV", "per note", "half-cycle extrema", "as vibrato_rate_hz", "", "as vibrato_rate_hz"),
        _S("jitter_local", "source", "%", "per segment", "f0-guided peak picking; sustained non-vibrato only",
           "SNR ≥ 30 dB; no lossy codec; vibrato extent < 15 cents; sample rate ≥ 19 kHz", "cycle-level waveform shape",
           "codec / SNR degradation curves"),
        _S("shimmer_local", "source", "%", "per segment", "as jitter_local", "as jitter_local", "as jitter_local", "as jitter_local"),
        _S("onset_rise_ms", "energy", "ms", "per note", "10–90 % energy rise", "interpreted with preceding-consonant class",
           "laryngeal kinematics", "coach onset probe; fortis / lenis contrast control"),
        _S("onset_f0_settle_ms", "energy", "ms", "per note", "voicing onset → f0 within tolerance", "as onset_rise_ms", "", "as onset_rise_ms"),
        _S("onset_aperiodicity_db", "energy", "dB", "per note", "high-band aperiodicity, first 30 ms", "as onset_rise_ms; aperiodicity valid", "", "as onset_rise_ms"),
        _S("onset_h1h2c_db", "energy", "dB", "per note", "median H1*–H2*, first 50 ms", "as onset_rise_ms; h1h2c valid", "", "as onset_rise_ms"),
    ]
}

#: Declared omissions (research §4 / §5.iv): the representation does not
#: carry these, by design, and never fakes them.
OMISSIONS: dict[str, str] = {
    "absolute_spl": "AGC and unknown device gain make absolute SPL unrecoverable without calibration",
    "breath_intake": "breath-intake noise and timing (candidate event track if coaches flag it)",
    "room_spatial_image": "room and spatial image are nuisances, gated not represented",
    "diction_detail": "intelligibility details beyond phoneme identity",
    "visual_cues": "visual and posture cues are outside audio",
    "fine_phase": "fine phase / waveform detail relevant to extreme irregular phonation",
}
