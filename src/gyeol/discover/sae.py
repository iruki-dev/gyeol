"""TopK sparse autoencoder for discovery on the residual r(t) and phonation features.

Architecture (Gao et al. 2024, "Scaling and evaluating sparse autoencoders")::

    z = TopK(W_enc (x − b_pre) + b_enc)        k active latents per frame, ReLU
    x̂ = W_dec z + b_pre                         decoder rows kept at unit norm

Loss: normalised MSE (reconstruction error / input variance) plus the AuxK
term: the ``k_aux`` largest pre-activations among *dead* latents (not fired
for ``dead_after`` steps) reconstruct the residual ``x − x̂``, which revives
them.  Inputs are standardised per dimension (statistics stored with the
fit).  Training is deterministic for a given seed.

The SAE is an analysis tool, not a coaching signal: features only become
attributes through :mod:`gyeol.discover.transfer` (promotion rule).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    def __init__(self, d: int, m: int, k: int):
        super().__init__()
        if not 0 < k <= m:
            raise ValueError("need 0 < k ≤ m")
        self.d, self.m, self.k = d, m, k
        self.b_pre = nn.Parameter(torch.zeros(d))
        self.W_dec = nn.Parameter(torch.randn(m, d))
        self.enc = nn.Linear(d, m)
        with torch.no_grad():
            self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True))
            self.enc.weight.copy_(self.W_dec)  # tied initialisation: encoder = decoderᵀ
            self.enc.bias.zero_()

    def pre(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x - self.b_pre)

    def topk(self, pre: torch.Tensor, k: int | None = None) -> torch.Tensor:
        k = k or self.k
        v, i = pre.topk(k, dim=-1)
        return torch.zeros_like(pre).scatter(-1, i, F.relu(v))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.topk(self.pre(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_pre

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pre = self.pre(x)
        z = self.topk(pre)
        return self.decode(z), z, pre

    @torch.no_grad()
    def normalise_decoder(self) -> None:
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))


@dataclass
class SAEConfig:
    n_latents: int = 64
    k: int = 4
    steps: int = 2000
    batch: int = 256
    lr: float = 2e-3
    aux_k: int = 16
    aux_coef: float = 1 / 32
    dead_after: int = 200  # steps without firing → dead
    seed: int = 0


@dataclass
class SAEFit:
    model: TopKSAE
    mean: np.ndarray
    std: np.ndarray
    config: SAEConfig
    history: list[float] = field(default_factory=list)  # normalised MSE per 100 steps

    def _x(self, X: np.ndarray) -> torch.Tensor:
        return torch.tensor((np.asarray(X, float) - self.mean) / self.std, dtype=torch.float32)

    @torch.no_grad()
    def codes(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        return self.model.encode(self._x(X)).numpy().astype(float)

    @torch.no_grad()
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        xh, _, _ = self.model(self._x(X))
        return xh.numpy() * self.std + self.mean

    def fvu(self, X: np.ndarray) -> float:
        """Fraction of variance unexplained (in standardised units)."""
        xs = (np.asarray(X, float) - self.mean) / self.std
        err = xs - (self.reconstruct(X) - self.mean) / self.std
        return float(np.sum(err**2) / (np.sum((xs - xs.mean(0)) ** 2) + 1e-12))

    @property
    def directions(self) -> np.ndarray:
        """Decoder directions (m, d) in standardised input space."""
        return self.model.W_dec.detach().numpy().astype(float)


def train_sae(X: np.ndarray, config: SAEConfig | None = None) -> SAEFit:
    cfg = config or SAEConfig()
    X = np.asarray(X, float)
    if X.ndim != 2 or len(X) < cfg.batch:
        raise ValueError(f"need a (N, D) matrix with N ≥ batch ({cfg.batch}); got {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError("SAE input contains NaN / inf (mask unknown frames first)")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    mean, std = X.mean(0), X.std(0) + 1e-8
    Xs = torch.tensor((X - mean) / std, dtype=torch.float32)
    model = TopKSAE(X.shape[1], cfg.n_latents, cfg.k)
    with torch.no_grad():
        model.b_pre.copy_(Xs.median(dim=0).values)  # cheap stand-in for the geometric median
    opt = torch.optim.Adam(model.parameters(), cfg.lr)
    last_fired = torch.zeros(cfg.n_latents, dtype=torch.long)
    fit = SAEFit(model, mean, std, cfg)
    var = Xs.var(0).sum().clamp_min(1e-8)
    acc = 0.0
    for step in range(1, cfg.steps + 1):
        xb = Xs[torch.as_tensor(rng.integers(0, len(Xs), cfg.batch))]
        xh, z, pre = model(xb)
        loss = ((xh - xb) ** 2).sum(-1).mean() / var
        fired = (z > 0).any(0)
        last_fired = torch.where(fired, torch.zeros_like(last_fired), last_fired + 1)
        dead = last_fired > cfg.dead_after
        if dead.any() and cfg.aux_coef > 0:
            e = (xb - xh).detach()
            k_aux = min(cfg.aux_k, int(dead.sum()))
            z_aux = model.topk(pre.masked_fill(~dead, -torch.inf), k_aux)
            e_hat = z_aux @ model.W_dec
            loss = loss + cfg.aux_coef * ((e_hat - e) ** 2).sum(-1).mean() / (e.var(0).sum().clamp_min(1e-8))
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.normalise_decoder()
        acc += float(loss.detach())
        if step % 100 == 0:
            fit.history.append(acc / 100)
            acc = 0.0
    return fit


@dataclass
class FeatureStats:
    frequency: np.ndarray  # fraction of frames where the latent is active
    mean_activation: np.ndarray  # mean activation when active
    energy_share: np.ndarray  # share of the reconstruction energy (standardised units)


def feature_stats(fit: SAEFit, X: np.ndarray) -> FeatureStats:
    z = fit.codes(X)
    active = z > 0
    freq = active.mean(0)
    mean_act = np.where(active.any(0), z.sum(0) / np.maximum(active.sum(0), 1), 0.0)
    energy = (z**2).sum(0) * (fit.directions**2).sum(1)
    return FeatureStats(freq, mean_act, energy / max(float(energy.sum()), 1e-12))
