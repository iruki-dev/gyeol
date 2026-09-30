"""Synthetic singing with exact ground truth (for tests and knob-recovery evaluation).

A source–filter singing voice:

* Rosenberg-C glottal flow with controllable open quotient, lip radiation,
  flow-modulated aspiration noise, jitter / shimmer / period doubling;
* a cascade of formant resonators, per note vowel (cross-faded), followed by
  an analytic higher-pole correction.  v0.1 realised the correction as
  digital resonators up to 300 Hz below Nyquist, i.e. near 21.8 kHz at
  44.1 kHz, which also steepened instead of flattened the top end (defect 8);
* :func:`melody` renders note sequences with vibrato, scoops, falls, 꺾기,
  glides, consonant bursts and timing offsets, returning the exact f0 track,
  note spans and ornament events.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

VOWELS: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    # (F1..F5), (B1..B5) — rough adult averages
    "a": ((750, 1250, 2600, 3400, 4200), (80, 90, 120, 150, 200)),
    "i": ((300, 2250, 3000, 3500, 4300), (60, 100, 120, 150, 200)),
    "u": ((330, 800, 2400, 3300, 4200), (60, 80, 120, 150, 200)),
    "e": ((500, 1850, 2600, 3400, 4200), (70, 90, 120, 150, 200)),
    "o": ((480, 850, 2600, 3400, 4200), (70, 80, 120, 150, 200)),
}


@dataclass
class SynthVowel:
    audio: np.ndarray
    sr: int
    truth: dict = field(default_factory=dict)


def rosenberg_flow(phase: np.ndarray, open_quotient: float, speed_quotient: float = 2.0) -> np.ndarray:
    """Rosenberg-C glottal flow as a function of cycle phase in [0, 1)."""
    tp = open_quotient * speed_quotient / (1.0 + speed_quotient)
    tn = open_quotient - tp
    g = np.zeros_like(phase)
    rise = phase < tp
    g[rise] = 0.5 * (1 - np.cos(np.pi * phase[rise] / tp))
    fall = (phase >= tp) & (phase < tp + tn)
    g[fall] = np.cos(0.5 * np.pi * (phase[fall] - tp) / tn)
    return g


def pole_set(formants, bandwidths, sr: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The *digital* resonators used by the filter: the given formants only.

    Formants within 1 kHz of Nyquist are dropped.  Resonances above F5 are
    not realised as digital poles at all (v0.1 put them up to 300 Hz below
    Nyquist — defect 8); their effect is added by :func:`higher_pole_correction`.
    """
    keep = [(f, b) for f, b in zip(formants, bandwidths) if f < sr / 2 - 1000.0]
    return tuple(f for f, _ in keep), tuple(b for _, b in keep)


def _analog_mag(f: np.ndarray, F: float, B: float) -> np.ndarray:
    return F**2 / np.sqrt((F**2 - f**2) ** 2 + (B * f) ** 2)


def higher_pole_correction(sr: int, formants, bandwidths, first: float = 5600.0, spacing: float = 1000.0,
                           last: float = 40000.0, n_taps: int = 257) -> np.ndarray:
    """Minimum-phase FIR turning the digital formant cascade into an analog one.

    Target: the continuous-time cascade of the given formants *plus* omitted
    higher resonances every ``spacing`` from ``first`` to ``last`` (well above
    any Nyquist), each unity gain at DC.  The FIR magnitude is target /
    digital-cascade response, so the rendered spectrum is that of an analog
    formant synthesiser at every sample rate, with no digital pole near
    Nyquist.
    """
    from .dsp.base import minimum_phase_fir

    nfft = 2 * (n_taps - 1) * 4
    f = np.fft.rfftfreq(nfft, 1 / sr)
    target = np.ones_like(f)
    for F, B in zip(formants, bandwidths):
        target *= _analog_mag(f, F, B)
    for Fj in np.arange(max(first, max(formants) + 1000.0), last + 1e-9, spacing):
        target *= _analog_mag(f, Fj, 300.0 + 0.05 * Fj)
    fd, bd = pole_set(formants, bandwidths, sr)
    digital = np.ones_like(f)
    w = 2 * np.pi * f / sr
    for F, B in zip(fd, bd):
        r = np.exp(-np.pi * B / sr)
        th = 2 * np.pi * F / sr
        a = np.array([1.0, -2 * r * np.cos(th), r * r])
        den = np.abs(a[0] + a[1] * np.exp(-1j * w) + a[2] * np.exp(-2j * w))
        digital *= a.sum() / den
    return minimum_phase_fir(np.clip(target / digital, 1e-7, 1e7), n_taps)


def formant_filter(x: np.ndarray, sr: int, formants, bandwidths) -> np.ndarray:
    y = x
    for f, b in zip(formants, bandwidths):
        r = np.exp(-np.pi * b / sr)
        theta = 2 * np.pi * f / sr
        a = [1.0, -2 * r * np.cos(theta), r * r]
        y = signal.lfilter([sum(a)], a, y)  # unity DC gain
    return y


def _excitation(inst_f0: np.ndarray, sr: int, open_quotient: float, speed_quotient: float, aspiration: float,
                jitter: float, shimmer: float, subharmonic: float, rng: np.random.Generator) -> np.ndarray:
    n = len(inst_f0)
    voiced = inst_f0 > 0
    phase_inc = np.where(voiced, inst_f0, 0.0) / sr
    if jitter > 0:
        n_cycles = int(np.sum(phase_inc)) + 2
        per_cycle = 1 + jitter * rng.standard_normal(n_cycles)
        cyc = np.floor(np.cumsum(phase_inc)).astype(int)
        phase_inc = phase_inc / per_cycle[np.minimum(cyc, n_cycles - 1)]
    total = np.cumsum(phase_inc)
    cycle = np.floor(total).astype(int)
    flow = rosenberg_flow(total - cycle, open_quotient, speed_quotient)
    amp = np.ones(cycle.max() + 1)
    if shimmer > 0:
        amp *= 1 + shimmer * rng.standard_normal(len(amp))
    if subharmonic > 0:
        amp *= 1 + subharmonic * np.where(np.arange(len(amp)) % 2 == 0, 1.0, -1.0)
    flow = flow * amp[cycle] * voiced
    dflow = np.diff(flow, prepend=flow[0])
    if aspiration > 0:
        noise = signal.lfilter([1, -0.9], [1], rng.standard_normal(n))
        noise *= (0.2 + flow / (flow.max() + 1e-12)) * voiced
        noise *= aspiration * np.std(dflow[voiced]) / (np.std(noise[voiced]) + 1e-12) if voiced.any() else 0.0
        dflow = dflow + noise
    return dflow


def sung_vowel(
    f0: float = 220.0,
    duration: float = 1.5,
    sr: int = 16000,
    vowel: str = "a",
    formants: tuple[float, ...] | None = None,
    bandwidths: tuple[float, ...] | None = None,
    open_quotient: float = 0.6,
    speed_quotient: float = 2.0,
    aspiration: float = 0.0,
    vibrato_rate: float = 0.0,
    vibrato_extent_cents: float = 0.0,
    jitter: float = 0.0,
    shimmer: float = 0.0,
    subharmonic: float = 0.0,
    onset_s: float = 0.05,
    release_s: float = 0.05,
    level_db: float = -12.0,
    seed: int = 0,
) -> SynthVowel:
    """A sustained sung vowel.  ``vibrato_extent_cents`` is the semi-extent."""
    rng = np.random.default_rng(seed)
    if formants is None or bandwidths is None:
        fv, bv = VOWELS[vowel]
        formants = formants or fv
        bandwidths = bandwidths or bv
    n = int(duration * sr)
    t = np.arange(n) / sr
    cents = vibrato_extent_cents * np.sin(2 * np.pi * vibrato_rate * t) if vibrato_rate > 0 else np.zeros(n)
    inst_f0 = f0 * 2 ** (cents / 1200.0)
    exc = _excitation(inst_f0, sr, open_quotient, speed_quotient, aspiration, jitter, shimmer, subharmonic, rng)
    y = formant_filter(exc, sr, *pole_set(formants, bandwidths, sr))
    y = signal.fftconvolve(y, higher_pole_correction(sr, formants, bandwidths))[:n]
    env = np.ones(n)
    a, r = int(onset_s * sr), int(release_s * sr)
    if a > 0:
        env[:a] = np.linspace(0, 1, a)
    if r > 0:
        env[-r:] = np.linspace(1, 0, r)
    y = y * env
    y = y / (np.max(np.abs(y)) + 1e-12) * 10 ** (level_db / 20)
    truth = {
        "f0": f0, "f0_track": inst_f0, "formants": tuple(formants), "bandwidths": tuple(bandwidths),
        "open_quotient": open_quotient, "vibrato_rate": vibrato_rate, "vibrato_extent_cents": vibrato_extent_cents,
        "jitter": jitter, "shimmer": shimmer, "aspiration": aspiration,
    }
    return SynthVowel(audio=y, sr=sr, truth=truth)


def phrase(notes: list[tuple[float, float]], sr: int = 16000, gap_s: float = 0.15, **kwargs) -> SynthVowel:
    """Concatenate sustained notes ``[(f0, duration), ...]`` separated by silent gaps."""
    parts, spans, t = [], [], 0.0
    for i, (f0, dur) in enumerate(notes):
        v = sung_vowel(f0=f0, duration=dur, sr=sr, seed=i, **kwargs)
        parts += [v.audio, np.zeros(int(gap_s * sr))]
        spans.append((t, t + dur, f0))
        t += dur + gap_s
    return SynthVowel(audio=np.concatenate(parts), sr=sr, truth={"notes": spans})


# ---------------------------------------------------------------------------
# melodies with ornaments
# ---------------------------------------------------------------------------


@dataclass
class SynthNote:
    f0: float
    dur: float
    gap_after: float = 0.1  # silence after the note (0 = legato into the next)
    vowel: str = "a"
    consonant: str | None = None  # "s" | "h" | "t" | "k": noise burst before the vowel
    vibrato_rate: float = 0.0
    vibrato_extent_cents: float = 0.0
    scoop_cents: float = 0.0  # start this far below and rise into the note
    scoop_s: float = 0.12
    fall_cents: float = 0.0  # drop this far at the end of the note
    fall_s: float = 0.15
    kkeokki_cents: float = 0.0  # signed excursion at the note middle
    kkeokki_s: float = 0.12
    glide_to_next: bool = False  # portamento over the last glide_s (needs gap_after = 0)
    glide_s: float = 0.12
    onset_shift_s: float = 0.0  # timing perturbation of this note's start (+ = late)


def _raised(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n)) if n > 0 else np.zeros(0)


def melody(notes: list[SynthNote], sr: int = 44100, lead_s: float = 0.2, tail_s: float = 0.3, open_quotient: float = 0.6,
           aspiration: float = 0.02, subharmonic: float = 0.0, transpose_cents: float = 0.0, level_db: float = -12.0,
           noise_db: float | None = -70.0, seed: int = 0) -> SynthVowel:
    """Render a note sequence.  Truth: f0 track (per sample), note spans, events."""
    rng = np.random.default_rng(seed)
    # layout: nominal start times, then per-note timing shifts
    starts, t = [], lead_s
    for nt in notes:
        starts.append(t + nt.onset_shift_s)
        t += nt.dur + nt.gap_after
    total = int((t + tail_s + 0.2) * sr)
    cents = np.full(total, np.nan)
    amp = np.zeros(total)
    vowel_idx = np.full(total, -1)
    cons = np.zeros(total)
    vowels = sorted({nt.vowel for nt in notes})
    spans, events = [], []
    for i, (nt, st) in enumerate(zip(notes, starts)):
        s = int(st * sr)
        nxt = int(starts[i + 1] * sr) if i + 1 < len(notes) else None
        e = s + int(nt.dur * sr)
        if nt.gap_after == 0 and nxt is not None:
            e = nxt  # legato: this note lasts until the next one starts
        n = e - s
        base = 1200 * np.log2(nt.f0 / 440.0) + transpose_cents
        c = np.full(n, base)
        tt = np.arange(n) / sr
        if nt.vibrato_rate > 0:
            fade = np.clip((tt - 0.15) / 0.2, 0, 1)
            c += fade * nt.vibrato_extent_cents * np.sin(2 * np.pi * nt.vibrato_rate * tt)
        if nt.scoop_cents:
            k = min(n, int(nt.scoop_s * sr))
            c[:k] -= nt.scoop_cents * (1 - _raised(k))
            events.append(("scoop", st, st + nt.scoop_s, nt.scoop_cents))
        if nt.fall_cents:
            k = min(n, int(nt.fall_s * sr))
            c[n - k :] -= nt.fall_cents * _raised(k)
            events.append(("fall", st + (n - k) / sr, st + n / sr, nt.fall_cents))
        if nt.kkeokki_cents:
            k = int(nt.kkeokki_s * sr)
            m = n // 2 - k // 2
            c[m : m + k] += nt.kkeokki_cents * np.sin(np.linspace(0, np.pi, k))
            events.append(("kkeokki", st + m / sr, st + (m + k) / sr, nt.kkeokki_cents))
        if nt.glide_to_next and i + 1 < len(notes):
            k = min(n, int(nt.glide_s * sr))
            target = 1200 * np.log2(notes[i + 1].f0 / 440.0) + transpose_cents
            c[n - k :] += (target - c[n - k]) * _raised(k)
            events.append(("glide", st + (n - k) / sr, st + n / sr, target - base))
        cents[s:e] = c
        a = np.ones(n)
        att = min(n, int(0.03 * sr))
        a[:att] = _raised(att)
        if nt.gap_after > 0 or i + 1 == len(notes):
            rel = min(n, int(0.04 * sr))
            a[n - rel :] = _raised(rel)[::-1]
        amp[s:e] = a
        vowel_idx[s:e] = vowels.index(nt.vowel)
        if nt.consonant:
            L = {"s": 0.08, "h": 0.06, "t": 0.015, "k": 0.02}[nt.consonant]
            k = int(L * sr)
            cs = max(0, s - k)
            burst = rng.standard_normal(s - cs)
            if nt.consonant == "s":
                burst = signal.sosfilt(signal.butter(4, 4000, "high", fs=sr, output="sos"), burst) * 0.3
            elif nt.consonant == "h":
                burst = signal.sosfilt(signal.butter(2, [500, 3000], "band", fs=sr, output="sos"), burst) * 0.15
            else:
                burst *= np.exp(-np.arange(len(burst)) / (0.004 * sr)) * 0.5
            cons[cs:s] += burst
        spans.append((st, st + n / sr, nt.f0))
    f0 = np.where(np.isfinite(cents), 440.0 * 2 ** (np.nan_to_num(cents) / 1200), 0.0)
    exc = _excitation(f0, sr, open_quotient, 2.0, aspiration, 0.0, 0.0, subharmonic, rng)
    y = np.zeros(total)
    for vi, v in enumerate(vowels):
        w = (vowel_idx == vi).astype(float)
        w = np.convolve(w, np.ones(int(0.03 * sr)) / int(0.03 * sr), mode="same")  # 30 ms cross-fade
        fv, bv = VOWELS[v]
        yv = formant_filter(exc, sr, *pole_set(fv, bv, sr))
        y += w * signal.fftconvolve(yv, higher_pole_correction(sr, fv, bv))[:total]
    y = y * amp
    peak = np.max(np.abs(y)) + 1e-12
    y = y / peak * 10 ** (level_db / 20) + cons * 10 ** (level_db / 20) * 0.5
    if noise_db is not None:
        y = y + rng.standard_normal(total) * 10 ** (noise_db / 20)
    truth = {"f0_track": np.where(amp > 0.5, f0, 0.0), "notes": spans, "events": events, "transpose_cents": transpose_cents}
    return SynthVowel(audio=y, sr=sr, truth=truth)
