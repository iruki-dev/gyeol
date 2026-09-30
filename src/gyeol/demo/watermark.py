"""Pluggable inaudible-watermark hook for generated audio (AI Basic Act).

Every generated waveform passes through a :class:`WatermarkHook` before it
leaves :mod:`gyeol.demo`.  The hook is swappable (e.g. for a commercial or
standardised scheme); the default is :class:`SpreadSpectrumWatermark`.

Default scheme (simple, documented, *not* tamper-proof):

* a pseudo-random ±1 chip sequence derived from ``key`` (SHA-256 seeded),
  one chip sequence per block of ``block`` samples, band-limited to
  ``band`` Hz;
* added at ``strength`` × the block's in-band RMS (−26 dB by default), so
  it follows the signal level and stays below the music in that band;
* detection correlates the band-passed audio with the same sequence per
  block and reports a z-score over blocks; ``z ≥ threshold`` = detected.

It survives gain changes, mild noise and 16-bit quantisation; it does **not**
survive time shifts that break block alignment, resampling, pitch shifting or
lossy codecs, and its inaudibility has not been verified by a listening test.
Use a stronger scheme in production; this one exists so the hook is always
exercised and testable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from scipy import signal

from ..core.status import Result


@dataclass
class WatermarkDetection:
    detected: bool
    z: float
    n_blocks: int


@runtime_checkable
class WatermarkHook(Protocol):
    name: str

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray: ...

    def detect(self, audio: np.ndarray, sr: int) -> Result[WatermarkDetection]: ...

    def describe(self) -> dict: ...


@dataclass
class SpreadSpectrumWatermark:
    key: str = "gyeol-ai-generated"
    strength: float = 0.05
    band: tuple[float, float] = (2000.0, 7000.0)
    block: int = 4096
    threshold: float = 4.0
    name: str = "gyeol-spread-spectrum-v1"

    def _sos(self, sr: int):
        hi = min(self.band[1], 0.45 * sr)
        return signal.butter(4, [self.band[0], hi], btype="bandpass", fs=sr, output="sos")

    def _chips(self, n_blocks: int, sr: int) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(f"{self.key}|{sr}|{self.block}".encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        pn = rng.choice([-1.0, 1.0], size=(n_blocks, self.block))
        pn = signal.sosfiltfilt(self._sos(sr), pn, axis=1)
        return pn / (np.std(pn, axis=1, keepdims=True) + 1e-12)

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        x = np.asarray(audio, float)
        nb = len(x) // self.block
        if nb == 0:
            return x.copy()
        inband = signal.sosfiltfilt(self._sos(sr), x)
        blocks = inband[: nb * self.block].reshape(nb, self.block)
        rms = np.sqrt(np.mean(blocks**2, axis=1, keepdims=True))
        mark = (self.strength * rms * self._chips(nb, sr)).reshape(-1)
        y = x.copy()
        y[: nb * self.block] += mark
        return y

    def detect(self, audio: np.ndarray, sr: int) -> Result[WatermarkDetection]:
        x = np.asarray(audio, float)
        nb = len(x) // self.block
        if nb < 4:
            return Result.failure(f"too short for detection: {nb} blocks of {self.block} samples (need ≥ 4)")
        inband = signal.sosfiltfilt(self._sos(sr), x)[: nb * self.block].reshape(nb, self.block)
        chips = self._chips(nb, sr)
        active = np.sqrt(np.mean(inband**2, axis=1)) > 1e-7
        if active.sum() < 4:
            return Result.failure("fewer than 4 non-silent blocks")
        # normalised correlation per block: ~N(0, 1/block_eff) without a mark
        num = np.sum(inband * chips, axis=1)
        den = np.sqrt(np.sum(inband**2, axis=1) * np.sum(chips**2, axis=1)) + 1e-12
        c = (num / den)[active]
        z = float(np.mean(c) / (np.std(c) / np.sqrt(len(c)) + 1e-12))
        return Result.success(WatermarkDetection(z >= self.threshold, z, int(active.sum())))

    def describe(self) -> dict:
        return {"scheme": self.name, "band_hz": list(self.band), "block": self.block, "strength": self.strength,
                "key_id": hashlib.sha256(self.key.encode()).hexdigest()[:12]}
