"""Synthetic sung vowels with known ground truth.

The research asks for "synthetic vowels with known formants" to calibrate the
high-f0 behaviour of formant / H1*–H2* estimators (§5.iii) and for known-truth
degradation tests.  This module produces a source–filter singing voice:

* Rosenberg-type glottal flow pulses with a controllable open quotient,
* lip radiation (first difference),
* aspiration noise modulated by the glottal flow (breathiness),
* optional jitter / shimmer / vibrato,
* a cascade of second-order formant resonators.

Everything returned in :class:`SynthVowel.truth` is exact by construction.
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


def formant_filter(x: np.ndarray, sr: int, formants, bandwidths) -> np.ndarray:
    y = x
    for f, b in zip(formants, bandwidths):
        if f >= sr / 2:
            continue
        r = np.exp(-np.pi * b / sr)
        theta = 2 * np.pi * f / sr
        a = [1.0, -2 * r * np.cos(theta), r * r]
        gain = sum(a)  # unity DC gain
        y = signal.lfilter([gain], a, y)
    return y


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
    """Synthesise a sustained sung vowel.

    ``vibrato_extent_cents`` is the semi-extent (peak deviation) in cents.
    ``jitter``/``shimmer`` are relative cycle-to-cycle SDs (e.g. 0.01 = 1 %).
    ``aspiration`` is the aspiration-noise to flow-derivative RMS ratio.
    ``subharmonic`` alternates cycle amplitudes by ±subharmonic (period doubling).
    """
    rng = np.random.default_rng(seed)
    if formants is None or bandwidths is None:
        fv, bv = VOWELS[vowel]
        formants = formants or fv
        bandwidths = bandwidths or bv
    n = int(duration * sr)
    t = np.arange(n) / sr
    cents = vibrato_extent_cents * np.sin(2 * np.pi * vibrato_rate * t) if vibrato_rate > 0 else np.zeros(n)
    inst_f0 = f0 * 2 ** (cents / 1200.0)
    # cycle-wise jitter: perturb each cycle's period
    phase_inc = inst_f0 / sr
    if jitter > 0:
        n_cycles = int(np.sum(phase_inc)) + 2
        per_cycle = 1 + jitter * rng.standard_normal(n_cycles)
        cyc = np.floor(np.cumsum(phase_inc)).astype(int)
        phase_inc = phase_inc / per_cycle[np.minimum(cyc, n_cycles - 1)]
    total_phase = np.cumsum(phase_inc)
    cycle = np.floor(total_phase).astype(int)
    phase = total_phase - cycle
    flow = rosenberg_flow(phase, open_quotient, speed_quotient)
    n_cycles = cycle.max() + 1
    amp = np.ones(n_cycles)
    if shimmer > 0:
        amp *= 1 + shimmer * rng.standard_normal(n_cycles)
    if subharmonic > 0:
        amp *= 1 + subharmonic * np.where(np.arange(n_cycles) % 2 == 0, 1.0, -1.0)
    flow = flow * amp[cycle]
    dflow = np.diff(flow, prepend=flow[0])
    if aspiration > 0:
        noise = signal.lfilter([1, -0.9], [1], rng.standard_normal(n))
        noise *= 0.2 + flow / (flow.max() + 1e-12)
        noise *= aspiration * np.std(dflow) / (np.std(noise) + 1e-12)
        excitation = dflow + noise
    else:
        excitation = dflow
    # Higher-pole correction: a real vocal tract keeps resonating above F5
    # (roughly every 1 kHz); without these the cascade falls ~60 dB/oct.
    extra = np.arange(max(max(formants) + 1000.0, 5600.0), sr / 2 - 300.0, 1000.0)
    all_f = tuple(formants) + tuple(extra)
    all_b = tuple(bandwidths) + tuple(300.0 + 0.05 * extra)
    y = formant_filter(excitation, sr, all_f, all_b)
    env = np.ones(n)
    a, r = int(onset_s * sr), int(release_s * sr)
    if a > 0:
        env[:a] = np.linspace(0, 1, a)
    if r > 0:
        env[-r:] = np.linspace(1, 0, r)
    y = y * env
    y = y / (np.max(np.abs(y)) + 1e-12) * 10 ** (level_db / 20)
    truth = {
        "f0": f0,
        "f0_track": inst_f0,
        "formants": tuple(formants),
        "bandwidths": tuple(bandwidths),
        "open_quotient": open_quotient,
        "vibrato_rate": vibrato_rate,
        "vibrato_extent_cents": vibrato_extent_cents,
        "jitter": jitter,
        "shimmer": shimmer,
        "aspiration": aspiration,
    }
    return SynthVowel(audio=y, sr=sr, truth=truth)


def phrase(notes: list[tuple[float, float]], sr: int = 16000, gap_s: float = 0.15, **kwargs) -> SynthVowel:
    """Concatenate notes ``[(f0, duration), ...]`` separated by silent gaps."""
    parts = []
    spans = []
    t = 0.0
    for i, (f0, dur) in enumerate(notes):
        v = sung_vowel(f0=f0, duration=dur, sr=sr, seed=i, **kwargs)
        parts.append(v.audio)
        spans.append((t, t + dur, f0))
        t += dur
        gap = np.zeros(int(gap_s * sr))
        parts.append(gap)
        t += gap_s
    return SynthVowel(audio=np.concatenate(parts), sr=sr, truth={"notes": spans})
