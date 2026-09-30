"""Log-mel frontend aligned to the shared :class:`~gyeol.core.grid.FrameGrid`.

``torch.stft(center=True)`` yields ``1 + N // hop`` frames centred on
``i * hop`` — exactly ``FrameGrid.for_samples(N, sr, hop)`` — so mel frames,
attribute curves and the residual share one time axis.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..core.grid import DEFAULT_HOP, DEFAULT_SR


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 30.0, fmax: float | None = None) -> np.ndarray:
    fmax = fmax or sr / 2
    mel = lambda f: 2595 * np.log10(1 + np.asarray(f) / 700)  # noqa: E731
    inv = lambda m: 700 * (10 ** (np.asarray(m) / 2595) - 1)  # noqa: E731
    pts = inv(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo, c, hi = pts[i : i + 3]
        fb[i] = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)
    return fb / (fb.sum(axis=1, keepdims=True) + 1e-9)


class LogMel(nn.Module):
    def __init__(self, sr: int = DEFAULT_SR, hop: int = DEFAULT_HOP, n_fft: int = 2048, n_mels: int = 80):
        super().__init__()
        self.sr, self.hop, self.n_fft, self.n_mels = sr, hop, n_fft, n_mels
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.register_buffer("fb", torch.tensor(mel_filterbank(sr, n_fft, n_mels), dtype=torch.float32), persistent=False)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav (B, N) → log-mel (B, 1 + N // hop, n_mels)."""
        spec = torch.stft(wav, self.n_fft, self.hop, window=self.window, center=True, return_complex=True).abs() ** 2
        return torch.log(torch.einsum("mf,bft->btm", self.fb, spec) + 1e-5)
