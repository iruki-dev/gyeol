"""Channel-adversarial residual encoder (research §4.6) — PyTorch.

Architecture
------------
* input: log-mel of the separated vocal (100 Hz) + core conditioning matrix
* encoder: 1-D conv stack, ×4 temporal downsampling → 25 Hz
* bottleneck: variational (VIB, Alemi et al. ICLR 2017), 32-d by default
* decoder: reconstructs the log-mel from core (100 Hz) + upsampled z, so z
  only has to carry what the core misses
* nuisance heads behind a gradient-reversal layer (Ganin et al. JMLR 2016):
  device / room / codec classification from z is *penalised*
* singer head: supervised-contrastive loss on utterance-pooled z keeps
  singer identity

*The z size (32) and rate (25 Hz) are UNVERIFIED HYPOTHESES to be set by the
rate–sufficiency sweep (research §5.i).*  No pretrained weights are shipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("gyeol.residual.torch_model needs the optional 'torch' package") from exc

from ..representation import Track
from .base import CORE_INPUTS, core_matrix


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


def grad_reverse(x: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lam)


@dataclass
class ResidualConfig:
    n_mels: int = 80
    core_dim: int = 2 * len(CORE_INPUTS)
    hidden: int = 192
    z_dim: int = 32
    downsample: int = 4  # 100 Hz → 25 Hz
    nuisance_classes: dict[str, int] = field(default_factory=lambda: {"device": 8, "room": 3, "codec": 6})
    n_singers: int | None = None
    beta_kl: float = 1e-3
    lambda_adv: float = 0.5
    lambda_con: float = 0.1
    temperature: float = 0.1


def _block(c_in: int, c_out: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv1d(c_in, c_out, 5, stride=stride, padding=2), nn.GroupNorm(8, c_out), nn.GELU())


class ResidualModel(nn.Module):
    def __init__(self, cfg: ResidualConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or ResidualConfig()
        h = cfg.hidden
        layers = [_block(cfg.n_mels + cfg.core_dim, h)]
        s = cfg.downsample
        while s > 1:
            layers.append(_block(h, h, stride=2))
            s //= 2
        layers.append(_block(h, h))
        self.encoder = nn.Sequential(*layers)
        self.to_stats = nn.Conv1d(h, 2 * cfg.z_dim, 1)
        self.decoder = nn.Sequential(
            _block(cfg.core_dim + cfg.z_dim, h), _block(h, h), _block(h, h), nn.Conv1d(h, cfg.n_mels, 1)
        )
        self.nuisance_heads = nn.ModuleDict({k: nn.Sequential(nn.Linear(cfg.z_dim, h), nn.GELU(), nn.Linear(h, n)) for k, n in cfg.nuisance_classes.items()})
        self.projector = nn.Sequential(nn.Linear(cfg.z_dim, h), nn.GELU(), nn.Linear(h, 64))

    def encode(self, mel: torch.Tensor, core: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """mel (B, n_mels, T), core (B, core_dim, T) → (mu, logvar) at T / downsample."""
        hdn = self.encoder(torch.cat([mel, core], dim=1))
        mu, logvar = self.to_stats(hdn).chunk(2, dim=1)
        return mu, logvar.clamp(-10, 10)

    def forward(self, mel: torch.Tensor, core: torch.Tensor, sample: bool = True) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(mel, core)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        z_up = F.interpolate(z, size=mel.shape[-1], mode="nearest")
        recon = self.decoder(torch.cat([core, z_up], dim=1))
        pooled = z.mean(dim=-1)
        nuis = {k: head(grad_reverse(pooled, self.cfg.lambda_adv)) for k, head in self.nuisance_heads.items()}
        return {"z": z, "mu": mu, "logvar": logvar, "recon": recon, "nuisance_logits": nuis, "embedding": F.normalize(self.projector(pooled), dim=-1)}


def supervised_contrastive(emb: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """SupCon loss (Khosla et al. 2020) over a batch of normalised embeddings."""
    sim = emb @ emb.T / temperature
    n = emb.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=emb.device)
    sim = sim.masked_fill(eye, -1e9)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    has_pos = pos.any(dim=1)
    if not has_pos.any():
        return emb.sum() * 0.0
    return -(log_prob * pos).sum(1)[has_pos].div(pos.sum(1)[has_pos]).mean()


def residual_loss(model: ResidualModel, out: dict, mel: torch.Tensor, nuisance_labels: dict[str, torch.Tensor] | None = None,
                  singer_labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    cfg = model.cfg
    recon = F.l1_loss(out["recon"], mel)
    kl = -0.5 * torch.mean(1 + out["logvar"] - out["mu"] ** 2 - out["logvar"].exp())
    total = recon + cfg.beta_kl * kl
    parts = {"recon": recon, "kl": kl}
    if nuisance_labels:
        # heads are trained to classify; the reversed gradient makes z uninformative
        adv = sum(F.cross_entropy(out["nuisance_logits"][k], y) for k, y in nuisance_labels.items())
        total = total + adv
        parts["nuisance_ce"] = adv
    if singer_labels is not None:
        con = supervised_contrastive(out["embedding"], singer_labels, cfg.temperature)
        total = total + cfg.lambda_con * con
        parts["singer_con"] = con
    parts["total"] = total
    return parts


def log_mel(audio: np.ndarray, sr: int, n_mels: int = 80, hop_seconds: float = 0.01, n_frames: int | None = None) -> np.ndarray:
    """(n_mels, T) log-mel on the core frame grid."""
    hop = int(round(hop_seconds * sr))
    win = 4 * hop
    x = torch.tensor(np.asarray(audio, dtype=np.float32))
    spec = torch.stft(x, n_fft=1024, hop_length=hop, win_length=win, window=torch.hann_window(win), center=True, return_complex=True).abs() ** 2
    fb = torch.tensor(_mel_filterbank(sr, 1024, n_mels), dtype=torch.float32)
    mel = torch.log(fb @ spec + 1e-6).numpy()
    if n_frames is not None:
        mel = mel[:, :n_frames] if mel.shape[1] >= n_frames else np.pad(mel, ((0, 0), (0, n_frames - mel.shape[1])), mode="edge")
    return mel


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 40.0, fmax: float | None = None) -> np.ndarray:
    fmax = fmax or sr / 2
    mel = lambda f: 2595 * np.log10(1 + f / 700)  # noqa: E731
    inv = lambda m: 700 * (10 ** (m / 2595) - 1)  # noqa: E731
    pts = inv(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo, c, hi = pts[i : i + 3]
        fb[i] = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)
    return fb


class TorchResidualEncoder:
    """:class:`~gyeol.residual.base.ResidualEncoder` backed by a trained model."""

    def __init__(self, model: ResidualModel, hop_seconds: float = 0.01, device: str = "cpu"):
        self.model = model.eval().to(device)
        self.hop_seconds = hop_seconds
        self.device = device

    @classmethod
    def from_checkpoint(cls, path: str, cfg: ResidualConfig | None = None, **kw) -> "TorchResidualEncoder":
        model = ResidualModel(cfg)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return cls(model, **kw)

    def encode(self, audio: np.ndarray, sr: int, core: dict[str, Track]) -> Track:
        cm = core_matrix(core).T  # (C, T)
        mel = log_mel(audio, sr, self.model.cfg.n_mels, self.hop_seconds, cm.shape[1])
        with torch.no_grad():
            mu, _ = self.model.encode(torch.tensor(mel)[None].to(self.device), torch.tensor(cm, dtype=torch.float32)[None].to(self.device))
        z = mu[0].T.cpu().numpy()
        ds = self.model.cfg.downsample
        rate = 1.0 / (self.hop_seconds * ds)
        # z is only meaningful where the core is (research: "only with core validity")
        voiced = core["f0_cents"].valid if "f0_cents" in core else np.ones(cm.shape[1], bool)
        m = len(z)
        blocks = np.pad(voiced, (0, max(0, m * ds - len(voiced))))[: m * ds].reshape(m, ds)
        return Track("residual", z, blocks.any(axis=1), rate, f"latent ({self.model.cfg.z_dim}-d)")
