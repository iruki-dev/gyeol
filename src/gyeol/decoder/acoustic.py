"""Attribute-conditioned acoustic model: (singer, env, c(t), r(t), f0, aperiodic) → mel + aperiodicity stream.

Conv front-end + Transformer encoder (a conv-transformer).  **Condition
dropout**: during training whole attribute channels are dropped at random and
replaced by a learned "missing" embedding (their presence mask is an input
too), so at inference the model tolerates attributes whose confidence is
zero.  Flow-matching is left for a later milestone.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AcousticModel(nn.Module):
    def __init__(self, c_dim: int, r_dim: int, singer_dim: int, env_dim: int, n_mels: int = 80, n_ap: int = 5,
                 hidden: int = 192, n_layers: int = 4, n_heads: int = 4, cond_dropout: float = 0.1):
        super().__init__()
        self.c_dim, self.cond_dropout = c_dim, cond_dropout
        self.missing = nn.Parameter(torch.zeros(c_dim))
        d_in = 2 * c_dim + r_dim + 3  # c, c-mask, r, (log f0, voiced, aperiodic)
        self.inp = nn.Linear(d_in, hidden)
        self.glob = nn.Linear(singer_dim + env_dim, hidden)
        self.conv = nn.Sequential(nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(), nn.Conv1d(hidden, hidden, 5, padding=2))
        layer = nn.TransformerEncoderLayer(hidden, n_heads, 4 * hidden, dropout=0.1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.mel_out = nn.Linear(hidden, n_mels)
        self.ap_out = nn.Linear(hidden, n_ap)

    def forward(self, c: torch.Tensor, c_mask: torch.Tensor, r: torch.Tensor, f0_hz: torch.Tensor, aperiodic: torch.Tensor,
                singer: torch.Tensor, env: torch.Tensor, frame_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """c, c_mask: (B, T, Dc); r: (B, T, Dr); f0_hz, aperiodic: (B, T) (f0 0 = unvoiced);
        singer: (B, Ds); env: (B, De) → (mel (B, T, n_mels), ap (B, T, n_ap) in dB ≤ 0)."""
        m = c_mask.bool()
        if self.training and self.cond_dropout > 0:
            drop = torch.rand(c.shape[0], 1, c.shape[2], device=c.device) < self.cond_dropout
            m = m & ~drop
        c_in = torch.where(m, torch.nan_to_num(c), self.missing.expand_as(c))
        voiced = (f0_hz > 0).float()
        logf0 = torch.where(f0_hz > 0, torch.log(f0_hz.clamp_min(1.0) / 440.0), torch.zeros_like(f0_hz))
        x = torch.cat([c_in, m.float(), r, logf0[..., None], voiced[..., None], torch.nan_to_num(aperiodic)[..., None]], dim=-1)
        h = self.inp(x) + self.glob(torch.cat([singer, env], dim=-1))[:, None, :]
        h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)
        pad = None if frame_mask is None else ~frame_mask.bool()
        h = self.transformer(h, src_key_padding_mask=pad)
        return self.mel_out(h), -nn.functional.softplus(self.ap_out(h))
