"""Reconstruction and adversarial losses for the autoencoder / vocoder (M4).

* :class:`MultiResolutionSTFTLoss` — spectral convergence + log-magnitude L1
  at several resolutions, optionally weighted per frame;
* :func:`weighted_mel_loss` — mel L1 with extra weight on consonant and
  low-energy frames (:func:`frame_weights`) so bursts and breath are not
  traded away for vowels;
* :class:`MultiPeriodDiscriminator` / :class:`MultiScaleDiscriminator` with
  LSGAN losses and feature matching (HiFiGAN).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def frame_weights(loudness_rel_db: np.ndarray, voiced: np.ndarray, consonant_weight: float = 3.0, low_energy_weight: float = 2.0,
                  low_energy_db: float = -25.0, silence_db: float = -60.0) -> np.ndarray:
    """Per-frame loss weights: consonants (unvoiced but not silent) and quiet voiced
    frames (breath, soft onsets/offsets) get more weight than sustained vowels."""
    lr = np.nan_to_num(loudness_rel_db, nan=-99.0)
    w = np.ones(len(lr))
    consonant = ~voiced & (lr > silence_db)
    quiet = voiced & (lr < low_energy_db)
    w[consonant] = consonant_weight
    w[quiet] = low_energy_weight
    return w


def weighted_mel_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    err = (pred - target).abs().mean(-1)  # (B, T)
    if weights is None:
        return err.mean()
    return (err * weights).sum() / weights.sum().clamp_min(1e-6)


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, ffts: tuple[int, ...] = (512, 1024, 2048), hops: tuple[int, ...] = (128, 256, 512)):
        super().__init__()
        self.ffts, self.hops = ffts, hops
        for n in ffts:
            self.register_buffer(f"w{n}", torch.hann_window(n), persistent=False)

    def forward(self, y: torch.Tensor, x: torch.Tensor, frame_weights: torch.Tensor | None = None, frame_hop: int = 512) -> torch.Tensor:
        total = torch.zeros((), device=y.device)
        for n, h in zip(self.ffts, self.hops):
            Y = torch.stft(y, n, h, window=getattr(self, f"w{n}"), center=True, return_complex=True).abs().clamp_min(1e-7)
            X = torch.stft(x, n, h, window=getattr(self, f"w{n}"), center=True, return_complex=True).abs().clamp_min(1e-7)
            sc = torch.linalg.norm(X - Y, dim=(1, 2)) / torch.linalg.norm(X, dim=(1, 2)).clamp_min(1e-7)
            lm = (X.log() - Y.log()).abs().mean(1)  # (B, T_n)
            if frame_weights is not None:
                idx = (torch.arange(lm.shape[1], device=y.device) * h // frame_hop).clamp(max=frame_weights.shape[1] - 1)
                w = frame_weights[:, idx]
                lm = (lm * w).sum(1) / w.sum(1).clamp_min(1e-6)
            else:
                lm = lm.mean(1)
            total = total + sc.mean() + lm.mean()
        return total / len(self.ffts)


class _PeriodDisc(nn.Module):
    def __init__(self, period: int, ch: int = 16):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([nn.Conv2d(1, ch, (5, 1), (3, 1), (2, 0)), nn.Conv2d(ch, 2 * ch, (5, 1), (3, 1), (2, 0)),
                                    nn.Conv2d(2 * ch, 2 * ch, (5, 1), 1, (2, 0))])
        self.out = nn.Conv2d(2 * ch, 1, (3, 1), 1, (1, 0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        b, n = x.shape
        pad = (self.period - n % self.period) % self.period
        x = F.pad(x, (0, pad), mode="reflect" if n > pad else "constant").view(b, 1, -1, self.period)
        feats = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            feats.append(x)
        return self.out(x).flatten(1), feats


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: tuple[int, ...] = (2, 3, 5, 7, 11), ch: int = 16):
        super().__init__()
        self.discs = nn.ModuleList(_PeriodDisc(p, ch) for p in periods)

    def forward(self, x: torch.Tensor):
        return [d(x) for d in self.discs]


class _ScaleDisc(nn.Module):
    def __init__(self, ch: int = 16):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(1, ch, 15, 1, 7), nn.Conv1d(ch, 2 * ch, 41, 4, 20, groups=4),
                                    nn.Conv1d(2 * ch, 2 * ch, 41, 4, 20, groups=8), nn.Conv1d(2 * ch, 2 * ch, 5, 1, 2)])
        self.out = nn.Conv1d(2 * ch, 1, 3, 1, 1)

    def forward(self, x: torch.Tensor):
        x = x[:, None]
        feats = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            feats.append(x)
        return self.out(x).flatten(1), feats


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, n_scales: int = 3, ch: int = 16):
        super().__init__()
        self.discs = nn.ModuleList(_ScaleDisc(ch) for _ in range(n_scales))
        self.pool = nn.AvgPool1d(4, 2, padding=2)

    def forward(self, x: torch.Tensor):
        outs = []
        for i, d in enumerate(self.discs):
            if i:
                x = self.pool(x[:, None])[:, 0]
            outs.append(d(x))
        return outs


def discriminator_loss(real: list, fake: list) -> torch.Tensor:
    return sum(((1 - r[0]) ** 2).mean() + (f[0] ** 2).mean() for r, f in zip(real, fake))


def generator_adv_loss(fake: list) -> torch.Tensor:
    return sum(((1 - f[0]) ** 2).mean() for f in fake)


def feature_matching_loss(real: list, fake: list) -> torch.Tensor:
    return sum(sum((rf.detach() - ff).abs().mean() for rf, ff in zip(r[1], f[1])) for r, f in zip(real, fake))
