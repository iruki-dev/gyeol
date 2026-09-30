"""Device equalisation and calibration (research §2.5).

* :class:`DeviceProfile` – fitted once from a *simultaneous* recording on a
  reference microphone and the consumer device (a sweep, pink noise or the
  singer's calibration phrase).  The inverse magnitude response over
  80 Hz–8 kHz becomes a minimum-phase FIR.  The device high-pass corner is
  also recorded because EQ cannot restore energy below it (H1 at low f0).
* :class:`LTASNormalizer` – blind alternative: map a recording's long-term
  average spectrum to a singer-specific reference LTAS.

EQ cannot undo nonlinear AGC / noise suppression; those conditions become
validity-mask failures instead.  *That EQ brings H1*–H2* within MDC across
phones is an UNVERIFIED HYPOTHESIS (research §5.ii); the engine therefore
only applies EQ when a profile is explicitly supplied.*
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import signal

from .._dsp import minimum_phase_fir, to_mono


def _smooth_octave(f: np.ndarray, db: np.ndarray, fraction: float = 1 / 6) -> np.ndarray:
    out = np.empty_like(db)
    for i, fc in enumerate(f):
        if fc <= 0:
            out[i] = db[i]
            continue
        lo, hi = fc * 2 ** (-fraction / 2), fc * 2 ** (fraction / 2)
        sel = (f >= lo) & (f <= hi)
        out[i] = np.mean(db[sel]) if sel.any() else db[i]
    return out


@dataclass
class DeviceProfile:
    name: str
    sample_rate: int
    freqs: list[float]
    correction_db: list[float]  # gain to apply to the device signal
    highpass_hz: float | None
    band: tuple[float, float] = (80.0, 8000.0)
    max_gain_db: float = 18.0

    @classmethod
    def fit(
        cls,
        reference: np.ndarray,
        device: np.ndarray,
        sr: int,
        name: str = "device",
        band: tuple[float, float] = (80.0, 8000.0),
        max_gain_db: float = 18.0,
        nperseg: int = 4096,
    ) -> "DeviceProfile":
        ref = to_mono(reference)
        dev = to_mono(device)
        n = min(len(ref), len(dev))
        ref, dev = ref[:n], dev[:n]
        # align (simultaneous recordings still have a small offset)
        corr = signal.correlate(ref, dev, mode="full", method="fft")
        lag = int(np.argmax(corr) - (n - 1))
        dev = np.roll(dev, lag)
        f, p_ref = signal.welch(ref, sr, nperseg=nperseg)
        _, p_dev = signal.welch(dev, sr, nperseg=nperseg)
        resp_db = 10 * np.log10((p_dev + 1e-20) / (p_ref + 1e-20))  # device / reference
        resp_db = _smooth_octave(f, resp_db)
        inb = (f >= band[0]) & (f <= band[1])
        mid = (f >= 500) & (f <= 2000)
        resp_db -= np.median(resp_db[mid])  # level is a nuisance, not a response
        corr_db = np.clip(-resp_db, -max_gain_db, max_gain_db)
        # outside the band: hold the edge values (no attempt to restore)
        lo_i, hi_i = np.flatnonzero(inb)[[0, -1]]
        corr_db[:lo_i] = corr_db[lo_i]
        corr_db[hi_i + 1 :] = corr_db[hi_i]
        # high-pass corner: lowest frequency where the device is within 3 dB of midband
        low = (f > 20) & (f < 1000)
        ok = np.flatnonzero(low & (resp_db > -3.0))
        hp = float(f[ok[0]]) if ok.size and f[ok[0]] > 25 else None
        return cls(name=name, sample_rate=sr, freqs=f.tolist(), correction_db=corr_db.tolist(), highpass_hz=hp, band=band, max_gain_db=max_gain_db)

    def fir(self, sr: int, n_taps: int = 1025) -> np.ndarray:
        nfft = 2 * (n_taps - 1)
        grid = np.fft.rfftfreq(nfft, 1 / sr)
        g = np.interp(grid, self.freqs, self.correction_db)
        return minimum_phase_fir(10 ** (g / 20), n_taps)

    def apply(self, x: np.ndarray, sr: int) -> np.ndarray:
        return signal.fftconvolve(to_mono(x), self.fir(sr), mode="full")[: len(x)]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path: str | Path) -> "DeviceProfile":
        d = json.loads(Path(path).read_text())
        d["band"] = tuple(d["band"])
        return cls(**d)


@dataclass
class LTASNormalizer:
    """Blind LTAS normalisation against a singer's reference LTAS."""

    freqs: np.ndarray
    reference_db: np.ndarray
    band: tuple[float, float] = (80.0, 8000.0)
    max_gain_db: float = 12.0

    @staticmethod
    def ltas(x: np.ndarray, sr: int, nperseg: int = 4096) -> tuple[np.ndarray, np.ndarray]:
        f, p = signal.welch(to_mono(x), sr, nperseg=nperseg)
        db = _smooth_octave(f, 10 * np.log10(p + 1e-20), 1 / 3)
        mid = (f >= 500) & (f <= 2000)
        return f, db - np.median(db[mid])

    @classmethod
    def from_reference(cls, recordings: list[np.ndarray], sr: int, **kw) -> "LTASNormalizer":
        spectra = [cls.ltas(r, sr) for r in recordings]
        f = spectra[0][0]
        return cls(freqs=f, reference_db=np.mean([s[1] for s in spectra], axis=0), **kw)

    def apply(self, x: np.ndarray, sr: int) -> np.ndarray:
        f, db = self.ltas(x, sr)
        ref = np.interp(f, self.freqs, self.reference_db)
        g = np.clip(ref - db, -self.max_gain_db, self.max_gain_db)
        inb = (f >= self.band[0]) & (f <= self.band[1])
        lo_i, hi_i = np.flatnonzero(inb)[[0, -1]]
        g[:lo_i], g[hi_i + 1 :] = g[lo_i], g[hi_i]
        nfft = 2 * (len(f) - 1)
        h = minimum_phase_fir(10 ** (g / 20), min(1025, nfft // 2 + 1))
        return signal.fftconvolve(to_mono(x), h, mode="full")[: len(x)]

