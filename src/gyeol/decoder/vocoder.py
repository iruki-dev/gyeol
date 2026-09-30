"""Source-filter neural vocoder (NSF-HiFiGAN / SiFiGAN style), reimplemented — weights are self-trained.

Source
    A sum of harmonics driven by f0 (phase = cumulative 2π·f0/sr, harmonics
    above Nyquist removed), plus an optional **subharmonic / jitter branch**
    (a sine at f0/2 and phase jitter, gated by a per-frame ``rough`` control)
    for rough voice and fry; Gaussian noise where unvoiced.
Filter
    HiFiGAN-style transposed-conv upsampling of (mel ⊕ aperiodicity) with
    multi-receptive-field residual blocks; the source is injected at every
    upsampling stage through strided convolutions (NSF-HiFiGAN).
Noise branch
    White noise shaped per frame by the aperiodicity bands (STFT-domain
    filtering), added to the output so breath / aspiration are modelled
    explicitly rather than hallucinated.

``upsample_rates`` must multiply to the frame hop (512 by default).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.grid import DEFAULT_HOP, DEFAULT_SR


def upsample_frames(x: torch.Tensor, hop: int, n_samples: int) -> torch.Tensor:
    """Frame-rate (B, T) → sample-rate (B, N) by linear interpolation on frame centres."""
    t = torch.arange(n_samples, device=x.device, dtype=x.dtype) / hop
    i0 = t.floor().long().clamp(0, x.shape[1] - 1)
    i1 = (i0 + 1).clamp(max=x.shape[1] - 1)
    w = (t - i0.to(t.dtype)).clamp(0, 1)
    return x[:, i0] * (1 - w) + x[:, i1] * w


class HarmonicSource(nn.Module):
    def __init__(self, sr: int = DEFAULT_SR, hop: int = DEFAULT_HOP, n_harmonics: int = 16, noise_std: float = 0.003,
                 sine_amp: float = 0.1):
        super().__init__()
        self.sr, self.hop, self.n_harmonics, self.noise_std, self.sine_amp = sr, hop, n_harmonics, noise_std, sine_amp
        self.merge = nn.Linear(n_harmonics + 1, 1)

    def forward(self, f0: torch.Tensor, n_samples: int, rough: torch.Tensor | None = None, seed: int | None = None) -> torch.Tensor:
        """f0 (B, T) Hz with 0 = unvoiced → excitation (B, 1, N)."""
        gen = None
        if seed is not None:
            gen = torch.Generator(device=f0.device).manual_seed(seed)
        f0u = upsample_frames(f0, self.hop, n_samples)
        voiced = (f0u > 0).to(f0u.dtype)
        rough_u = upsample_frames(rough, self.hop, n_samples) if rough is not None else torch.zeros_like(f0u)
        jitter = torch.randn(f0u.shape, generator=gen, device=f0.device) * 0.01 * rough_u
        phase = 2 * math.pi * torch.cumsum(f0u * (1 + jitter) / self.sr, dim=1)
        k = torch.arange(1, self.n_harmonics + 1, device=f0.device, dtype=f0u.dtype)
        harm = torch.sin(phase[..., None] * k)  # (B, N, K)
        harm = harm * ((f0u[..., None] * k) < self.sr / 2).to(harm.dtype)
        sub = torch.sin(0.5 * phase)[..., None] * rough_u[..., None]  # subharmonic branch
        src = torch.cat([harm, sub], dim=-1) * self.sine_amp * voiced[..., None]
        noise = torch.randn(f0u.shape, generator=gen, device=f0.device)[..., None]
        noise = noise * (voiced[..., None] * self.noise_std + (1 - voiced[..., None]) * self.sine_amp / 3)
        return torch.tanh(self.merge(src + noise)).transpose(1, 2)


class _ResBlock(nn.Module):
    def __init__(self, ch: int, kernel: int, dilations: tuple[int, ...]):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv1d(ch, ch, kernel, dilation=d, padding=d * (kernel - 1) // 2) for d in dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c in self.convs:
            x = x + c(F.leaky_relu(x, 0.1))
        return x


class NoiseBranch(nn.Module):
    """Aperiodicity-shaped noise: band gains (dB ≤ 0) per frame → filtered white noise."""

    def __init__(self, sr: int = DEFAULT_SR, hop: int = DEFAULT_HOP, n_fft: int = 2048,
                 band_edges: tuple[float, ...] = (0.0, 1000.0, 2000.0, 4000.0, 8000.0, 22050.0)):
        super().__init__()
        self.sr, self.hop, self.n_fft = sr, hop, n_fft
        freqs = torch.fft.rfftfreq(n_fft, 1 / sr)
        idx = torch.bucketize(freqs, torch.tensor(band_edges[1:-1]))
        self.register_buffer("band_of_bin", idx, persistent=False)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.gain = nn.Parameter(torch.tensor(0.05))

    def forward(self, ap_db: torch.Tensor, n_samples: int, seed: int | None = None) -> torch.Tensor:
        gen = torch.Generator(device=ap_db.device).manual_seed(seed) if seed is not None else None
        noise = torch.randn(ap_db.shape[0], n_samples, generator=gen, device=ap_db.device)
        Z = torch.stft(noise, self.n_fft, self.hop, window=self.window, center=True, return_complex=True)  # (B, F, T)
        T = min(Z.shape[-1], ap_db.shape[1])
        g = (10 ** (ap_db[:, :T, :] / 20)).transpose(1, 2)[:, self.band_of_bin, :]  # (B, F, T)
        y = torch.istft(Z[..., :T] * g, self.n_fft, self.hop, window=self.window, center=True, length=n_samples)
        return self.gain * y


class NSFVocoder(nn.Module):
    def __init__(self, n_mels: int = 80, n_ap: int = 5, sr: int = DEFAULT_SR, hop: int = DEFAULT_HOP, channels: int = 256,
                 upsample_rates: tuple[int, ...] = (8, 8, 4, 2), resblock_kernels: tuple[int, ...] = (3, 7, 11),
                 resblock_dilations: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)), n_harmonics: int = 16):
        super().__init__()
        if math.prod(upsample_rates) != hop:
            raise ValueError(f"upsample_rates {upsample_rates} multiply to {math.prod(upsample_rates)}, not hop {hop}")
        self.sr, self.hop = sr, hop
        self.source = HarmonicSource(sr, hop, n_harmonics)
        self.noise = NoiseBranch(sr, hop)
        self.pre = nn.Conv1d(n_mels + n_ap, channels, 7, padding=3)
        self.ups, self.src_convs, self.blocks = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        ch = channels
        remaining = hop
        for r in upsample_rates:
            remaining //= r
            out = max(ch // 2, 8)
            self.ups.append(nn.ConvTranspose1d(ch, out, 2 * r, stride=r, padding=r // 2 + r % 2, output_padding=r % 2))
            self.src_convs.append(nn.Conv1d(1, out, 2 * remaining if remaining > 1 else 1, stride=remaining,
                                            padding=remaining // 2 if remaining > 1 else 0))
            self.blocks.append(nn.ModuleList(_ResBlock(out, k, d) for k, d in zip(resblock_kernels, resblock_dilations)))
            ch = out
        self.post = nn.Conv1d(ch, 1, 7, padding=3)

    def forward(self, mel: torch.Tensor, ap_db: torch.Tensor, f0_hz: torch.Tensor, rough: torch.Tensor | None = None,
                seed: int | None = None) -> torch.Tensor:
        """mel (B, T, M), ap_db (B, T, A), f0_hz (B, T) → waveform (B, (T − 1)·hop)."""
        n = (mel.shape[1] - 1) * self.hop
        src = self.source(f0_hz, n, rough, seed)  # (B, 1, n)
        x = self.pre(torch.cat([mel, ap_db], dim=-1).transpose(1, 2))
        for up, sc, blocks in zip(self.ups, self.src_convs, self.blocks):
            x = up(F.leaky_relu(x, 0.1))
            s = sc(src)
            L = min(x.shape[-1], s.shape[-1])
            x = x[..., :L] + s[..., :L]
            x = sum(b(x) for b in blocks) / len(blocks)
        y = torch.tanh(self.post(F.leaky_relu(x, 0.1)))[:, 0]
        y = F.pad(y, (0, max(0, n - y.shape[-1])))[:, :n]
        return y + self.noise(ap_db, n, seed)
