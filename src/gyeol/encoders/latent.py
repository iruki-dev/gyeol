"""Singer, env and residual encoders (M4).

* :class:`SingerEncoder` — utterance embedding (attentive statistics
  pooling), trained with a same-singer supervised-contrastive loss across
  songs.  :func:`singer_vector` wraps the output in a
  :class:`~gyeol.core.consent.SingerVector` carrying the source recording's
  provenance, so reference vectors can never become renderable voices.
* :class:`EnvEncoder` — utterance env vector plus supervised heads for the
  augmentation labels (noise, room, EQ, codec, compression, separation).
* :class:`ResidualEncoder` — narrow frame-level bottleneck with a
  variational information bottleneck (VIB).  Adversarial heads behind a
  gradient-reversal layer try to predict every attribute in ``c``, the
  singer and the env from ``r``; training the encoder to defeat them
  suppresses leakage.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.consent import SingerVector
from ..core.containers import Recording


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grad_reverse(x: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """Identity forward, gradient × (−lam) backward (Ganin et al. 2016)."""
    return _GradReverse.apply(x, lam)


def _conv_stack(d_in: int, hidden: int, n: int = 3, kernel: int = 5) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = d_in
    for _ in range(n):
        layers += [nn.Conv1d(d, hidden, kernel, padding=kernel // 2), nn.GroupNorm(1, hidden), nn.GELU()]
        d = hidden
    return nn.Sequential(*layers)


class AttentiveStatsPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.att = nn.Sequential(nn.Conv1d(dim, dim // 2 or 1, 1), nn.Tanh(), nn.Conv1d(dim // 2 or 1, 1, 1))

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """h (B, C, T), mask (B, T) → (B, 2C) weighted mean and std."""
        a = self.att(h)
        if mask is not None:
            a = a.masked_fill(~mask[:, None, :], -1e4)
        w = a.softmax(-1)
        mu = (h * w).sum(-1)
        sd = ((h - mu[..., None]) ** 2 * w).sum(-1).clamp_min(1e-6).sqrt()
        return torch.cat([mu, sd], dim=1)


class SingerEncoder(nn.Module):
    def __init__(self, n_mels: int = 80, hidden: int = 128, dim: int = 64):
        super().__init__()
        self.convs = _conv_stack(n_mels, hidden)
        self.pool = AttentiveStatsPool(hidden)
        self.proj = nn.Linear(2 * hidden, dim)

    def forward(self, mel: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """mel (B, T, M) → L2-normalised singer embedding (B, dim)."""
        h = self.convs(mel.transpose(1, 2))
        return F.normalize(self.proj(self.pool(h, mask)), dim=-1)


def supervised_contrastive(emb: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """SupCon (Khosla et al. 2020): pull same-singer embeddings together."""
    sim = emb @ emb.T / temperature
    n = emb.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=emb.device)
    sim = sim.masked_fill(eye, -1e9)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    has = pos.any(1)
    if not has.any():
        return emb.sum() * 0.0
    return -((logp * pos).sum(1)[has] / pos.sum(1)[has]).mean()


@torch.no_grad()
def singer_vector(encoder: SingerEncoder, mel: np.ndarray, recording: Recording) -> SingerVector:
    """Embed one recording; the vector inherits the recording's provenance and owner."""
    encoder.eval()
    v = encoder(torch.tensor(mel, dtype=torch.float32)[None])[0].numpy()
    return SingerVector(v, recording.provenance, recording.recording_id, owner_id=recording.owner_id)


ENV_HEADS = {"noise": 4, "room": 4, "eq": 2, "codec": 5, "compression": 2, "separation": 2}


class EnvEncoder(nn.Module):
    def __init__(self, n_mels: int = 80, hidden: int = 96, dim: int = 32, heads: dict[str, int] | None = None):
        super().__init__()
        self.convs = _conv_stack(n_mels, hidden)
        self.pool = AttentiveStatsPool(hidden)
        self.proj = nn.Linear(2 * hidden, dim)
        self.heads = nn.ModuleDict({k: nn.Linear(dim, n) for k, n in (heads or ENV_HEADS).items()})

    def forward(self, mel: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        e = self.proj(self.pool(self.convs(mel.transpose(1, 2)), mask))
        return e, {k: h(e) for k, h in self.heads.items()}


class ResidualEncoder(nn.Module):
    """Frame-level r(t) with a VIB bottleneck; ``dim`` should be small (4–16)."""

    def __init__(self, n_mels: int = 80, cond_dim: int = 0, hidden: int = 128, dim: int = 8):
        super().__init__()
        self.convs = _conv_stack(n_mels + cond_dim, hidden)
        self.mu = nn.Conv1d(hidden, dim, 1)
        self.logvar = nn.Conv1d(hidden, dim, 1)
        self.dim = dim

    def forward(self, mel: torch.Tensor, cond: torch.Tensor | None = None, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """→ (r (B, T, dim), KL per example (B,))."""
        x = mel if cond is None else torch.cat([mel, cond], dim=-1)
        h = self.convs(x.transpose(1, 2))
        mu, lv = self.mu(h), self.logvar(h).clamp(-8, 8)
        r = mu + torch.randn_like(mu) * (0.5 * lv).exp() if (sample and self.training) else mu
        kl = 0.5 * (mu.pow(2) + lv.exp() - 1 - lv).sum(1).mean(-1)
        return r.transpose(1, 2), kl


class LeakageHeads(nn.Module):
    """Adversaries that try to read attributes / singer / env from r (behind GRL).

    ``regress`` targets are per-frame scalars (e.g. f0 cents / 1200); ``classify``
    targets are per-frame (T,) or per-utterance (singer, env classes).
    """

    def __init__(self, r_dim: int, regress: dict[str, int], classify: dict[str, int], hidden: int = 64, grl: float = 1.0):
        super().__init__()
        self.grl = grl
        mk = lambda n: nn.Sequential(nn.Linear(r_dim, hidden), nn.GELU(), nn.Linear(hidden, n))  # noqa: E731
        self.reg = nn.ModuleDict({k: mk(n) for k, n in regress.items()})
        self.cls = nn.ModuleDict({k: mk(n) for k, n in classify.items()})

    def forward(self, r: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        z = grad_reverse(r, self.grl)
        return {k: m(z) for k, m in self.reg.items()}, {k: m(z) for k, m in self.cls.items()}
