"""The gyeol autoencoder: encoders + acoustic model + vocoder.

    encode:  wav, c(t) ─┬─ SingerEncoder → singer
                        ├─ EnvEncoder    → env (+ env class logits)
                        └─ ResidualEncoder(mel of a timbre-shifted copy, c) → r(t)
    decode:  (singer, env, c, r, f0, aperiodic) → AcousticModel → mel, ap → NSFVocoder → wav

The residual encoder sees a **timbre-shifted** copy of the input
(Seed-VC / YingMusic style augmentation, :mod:`gyeol.data.augment`) while the
decoder must reconstruct the original, so timbre has to flow through the
singer vector rather than r.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..core.grid import DEFAULT_HOP, DEFAULT_SR
from ..encoders.latent import ENV_HEADS, EnvEncoder, LeakageHeads, ResidualEncoder, SingerEncoder
from ..encoders.mel import LogMel
from .acoustic import AcousticModel
from .vocoder import NSFVocoder

#: attribute channels of c(t) fed to the decoder (signal-layer curves; learned heads may add more)
C_CHANNELS = ("pitch_center", "vibrato_rate", "vibrato_extent", "loudness_rel", "aperiodic_ratio", "subharmonic_ratio", "voicing")


@dataclass
class AutoencoderConfig:
    sr: int = DEFAULT_SR
    hop: int = DEFAULT_HOP
    n_mels: int = 80
    n_ap: int = 5
    c_dim: int = len(C_CHANNELS)
    singer_dim: int = 64
    env_dim: int = 32
    r_dim: int = 8
    hidden: int = 192
    n_layers: int = 4
    vocoder_channels: int = 256
    upsample_rates: tuple[int, ...] = (8, 8, 4, 2)
    cond_dropout: float = 0.1
    n_singers: int = 0  # >0 enables a singer-id leakage adversary on r
    leak_regress: dict[str, int] = field(default_factory=lambda: {"f0": 1, "aperiodic": 1, "loudness": 1})
    env_heads: dict[str, int] = field(default_factory=lambda: dict(ENV_HEADS))

    @classmethod
    def tiny(cls, **kw) -> "AutoencoderConfig":
        """A CPU-test-sized model."""
        base = dict(n_mels=32, singer_dim=16, env_dim=8, r_dim=4, hidden=32, n_layers=1, vocoder_channels=16)
        base.update(kw)
        return cls(**base)


class GyeolAutoencoder(nn.Module):
    def __init__(self, cfg: AutoencoderConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or AutoencoderConfig()
        self.mel = LogMel(cfg.sr, cfg.hop, n_mels=cfg.n_mels)
        h = max(cfg.hidden // 2, 16)
        self.singer_enc = SingerEncoder(cfg.n_mels, h, cfg.singer_dim)
        self.env_enc = EnvEncoder(cfg.n_mels, h, cfg.env_dim, cfg.env_heads)
        self.res_enc = ResidualEncoder(cfg.n_mels, cfg.c_dim, h, cfg.r_dim)
        cls = {"singer": cfg.n_singers} if cfg.n_singers else {}
        cls.update({f"env_{k}": n for k, n in cfg.env_heads.items()})
        self.leak = LeakageHeads(cfg.r_dim, cfg.leak_regress, cls, hidden=h)
        # env must not leak into the singer vector either
        self.singer_env_adv = LeakageHeads(cfg.singer_dim, {}, {k: n for k, n in cfg.env_heads.items()}, hidden=h)
        self.acoustic = AcousticModel(cfg.c_dim, cfg.r_dim, cfg.singer_dim, cfg.env_dim, cfg.n_mels, cfg.n_ap, cfg.hidden,
                                      cfg.n_layers, max(1, cfg.hidden // 48), cfg.cond_dropout)
        self.vocoder = NSFVocoder(cfg.n_mels, cfg.n_ap, cfg.sr, cfg.hop, cfg.vocoder_channels, cfg.upsample_rates)

    def encode(self, wav: torch.Tensor, c: torch.Tensor, c_mask: torch.Tensor, wav_for_residual: torch.Tensor | None = None) -> dict:
        mel = self.mel(wav)
        T = min(mel.shape[1], c.shape[1])
        mel, c, c_mask = mel[:, :T], c[:, :T], c_mask[:, :T]
        singer = self.singer_enc(mel)
        env, env_logits = self.env_enc(mel)
        mel_r = self.mel(wav_for_residual)[:, :T] if wav_for_residual is not None else mel
        r, kl = self.res_enc(mel_r, torch.where(c_mask.bool(), torch.nan_to_num(c), torch.zeros_like(c)))
        return {"mel": mel, "singer": singer, "env": env, "env_logits": env_logits, "r": r, "kl": kl}

    def decode(self, singer: torch.Tensor, env: torch.Tensor, c: torch.Tensor, c_mask: torch.Tensor, r: torch.Tensor,
               f0_hz: torch.Tensor, aperiodic: torch.Tensor, rough: torch.Tensor | None = None, vocode: bool = True, seed: int | None = None) -> dict:
        mel, ap = self.acoustic(c, c_mask, r, f0_hz, aperiodic, singer, env)
        out = {"mel": mel, "ap": ap}
        if vocode:
            out["wav"] = self.vocoder(mel, ap, f0_hz, rough, seed)
        return out
