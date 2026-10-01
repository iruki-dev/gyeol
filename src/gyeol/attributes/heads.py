"""Learned attribute heads with calibration and out-of-distribution "unknown".

Tasks (each a head on shared frame features):

* ``register`` — softmax over chest / mixed / falsetto,
* ``phonation`` — independent sigmoids over breathy / pressed_belt /
  pharyngeal_twang / fry / rough (qualities can co-occur),
* ``phones`` — softmax over a Korean phone set (diction),
* ``laryngeal`` — softmax over lenis / aspirated / fortis (ㄷ/ㄸ/ㅌ-type contrasts).

Outputs are **calibrated** (per-task temperature fitted on held-out singers)
and **abstain**: Mahalanobis detectors on the penultimate embedding and on
the input features mark out-of-distribution frames "unknown" (confidence 0),
and confidences are multiplied by the frontend's per-frame quality factor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.containers import AttributeCurve
from ..core.grid import FrameGrid

REGISTER = ("chest", "mixed", "falsetto")
PHONATION_QUALITIES = ("breathy", "pressed_belt", "pharyngeal_twang", "fry", "rough")
LARYNGEAL = ("lenis", "aspirated", "fortis")


@dataclass
class TaskSpec:
    n: int
    kind: str  # "softmax" | "sigmoid" | "regression" (heteroscedastic Gaussian: mean and log-variance per target)
    labels: tuple[str, ...] = ()
    unit: str = ""  # regression targets only

    def __post_init__(self) -> None:
        if self.kind not in ("softmax", "sigmoid", "regression"):
            raise ValueError(f"unknown task kind {self.kind!r}")

    @property
    def out_dim(self) -> int:
        return 2 * self.n if self.kind == "regression" else self.n


def default_tasks(n_phones: int = 0) -> dict[str, TaskSpec]:
    t = {"register": TaskSpec(3, "softmax", REGISTER), "phonation": TaskSpec(5, "sigmoid", PHONATION_QUALITIES),
         "laryngeal": TaskSpec(3, "softmax", LARYNGEAL)}
    if n_phones:
        t["phones"] = TaskSpec(n_phones, "softmax")
    return t


class FrameHeads(nn.Module):
    """Temporal-conv trunk + one linear head per task."""

    def __init__(self, in_dim: int, tasks: dict[str, TaskSpec], hidden: int = 128, kernel: int = 5, dropout: float = 0.1):
        super().__init__()
        self.tasks = tasks
        self.norm = nn.LayerNorm(in_dim)
        self.trunk = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel, padding=kernel // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2), nn.GELU(),
        )
        self.heads = nn.ModuleDict({k: nn.Linear(hidden, s.out_dim) for k, s in tasks.items()})

    def forward(self, x: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """x: (B, T, D) → ({task: logits (B, T, n)}, embedding (B, T, H))."""
        h = self.trunk(self.norm(x).transpose(1, 2)).transpose(1, 2)
        return {k: head(h) for k, head in self.heads.items()}, h


# ---------------------------------------------------------------------------
# calibration and OOD
# ---------------------------------------------------------------------------


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor, kind: str, max_iter: int = 200) -> float:
    """Temperature minimising NLL on held-out data (Guo et al. 2017)."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        z = logits / log_t.exp()
        loss = F.cross_entropy(z, targets) if kind == "softmax" else F.binary_cross_entropy_with_logits(z, targets.float())
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20.0))


def expected_calibration_error(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """ECE of the top-class confidence (softmax tasks)."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == targets).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


@dataclass
class MahalanobisOOD:
    """Class-conditional Gaussians with a shared covariance on the embedding.

    Score = distance to the nearest class mean; the threshold is the
    ``quantile`` of in-distribution held-out scores, so ~(1 − quantile) of
    in-distribution frames are (conservatively) marked unknown.
    """

    means: np.ndarray | None = None
    precision: np.ndarray | None = None
    threshold: float = float("inf")

    def fit(self, emb: np.ndarray, labels: np.ndarray, shrinkage: float = 1e-3) -> "MahalanobisOOD":
        classes = np.unique(labels)
        self.means = np.stack([emb[labels == c].mean(0) for c in classes])
        centred = emb - self.means[np.searchsorted(classes, labels)]
        cov = np.cov(centred, rowvar=False) + shrinkage * np.eye(emb.shape[1])
        self.precision = np.linalg.inv(cov)
        return self

    def score(self, emb: np.ndarray) -> np.ndarray:
        d = emb[:, None, :] - self.means[None, :, :]
        m = np.einsum("nkd,de,nke->nk", d, self.precision, d)
        return np.sqrt(np.maximum(m.min(axis=1), 0.0))

    def calibrate(self, emb_in: np.ndarray, quantile: float = 0.99) -> "MahalanobisOOD":
        self.threshold = float(np.quantile(self.score(emb_in), quantile))
        return self

    def is_unknown(self, emb: np.ndarray) -> np.ndarray:
        return self.score(emb) > self.threshold


@dataclass
class CalibratedHeads:
    """Trained heads + per-task temperatures + OOD detector → attribute curves."""

    model: FrameHeads
    temperatures: dict[str, float] = field(default_factory=dict)
    #: regression tasks: factor on the predicted variance fitted on held-out singers (NLL-optimal scaling)
    variance_scale: dict[str, float] = field(default_factory=dict)
    #: regression tasks: median calibrated σ on held-out singers (σ at which confidence = 0.5)
    sigma_ref: dict[str, float] = field(default_factory=dict)
    ood: MahalanobisOOD | None = None  # on the penultimate embedding
    input_ood: MahalanobisOOD | None = None  # on the raw input features
    feature_name: str = "dsp"

    @torch.no_grad()
    def predict(self, feats: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """(outputs per task, unknown mask, embedding) for (T, D) features.

        Classification tasks give probabilities (T, n); regression tasks give
        ``[mean (T, n) | calibrated σ (T, n)]`` concatenated to (T, 2n).
        """
        self.model.eval()
        logits, emb = self.model(torch.tensor(feats, dtype=torch.float32)[None])
        probs = {}
        for k, z in logits.items():
            spec = self.model.tasks[k]
            if spec.kind == "regression":
                mu, logvar = z[0][:, : spec.n], z[0][:, spec.n :]
                sigma = (logvar.exp() * self.variance_scale.get(k, 1.0)).sqrt()
                probs[k] = torch.cat([mu, sigma], -1).numpy()
                continue
            z = z[0] / self.temperatures.get(k, 1.0)
            probs[k] = (z.softmax(-1) if spec.kind == "softmax" else z.sigmoid()).numpy()
        e = emb[0].numpy()
        unknown = np.zeros(len(e), bool)
        if self.ood is not None:
            unknown |= self.ood.is_unknown(e)
        if self.input_ood is not None:
            unknown |= self.input_ood.is_unknown(np.asarray(feats, float))
        return probs, unknown, e

    def curves(self, feats: np.ndarray, grid: FrameGrid, quality_factor: np.ndarray | None = None,
               voiced: np.ndarray | None = None) -> dict[str, AttributeCurve]:
        probs, unknown, _ = self.predict(feats)
        q = np.ones(grid.n_frames) if quality_factor is None else quality_factor
        v = np.ones(grid.n_frames, bool) if voiced is None else voiced
        out = {}
        for k, p in probs.items():
            spec = self.model.tasks[k]
            if spec.kind == "regression":
                mu, sigma = p[:, : spec.n], p[:, spec.n :]
                ref = self.sigma_ref.get(k, float(np.median(sigma)))
                c = (ref / (ref + sigma.mean(axis=1))) * q * ~unknown
                vals = (mu[:, 0] if spec.n == 1 else mu).copy()
                vals[unknown] = np.nan
                out[k] = AttributeCurve(k, vals, c, grid, spec.unit, labels=spec.labels if spec.n > 1 else (),
                                        meta={"calibrated": k in self.variance_scale, "features": self.feature_name,
                                              "sigma": sigma, "unknown_fraction": float(unknown.mean())})
                continue
            peak = p.max(axis=1) if spec.kind == "softmax" else np.abs(p - 0.5).max(axis=1) * 2
            conf = peak * q * ~unknown * (v if k in ("register", "phonation") else 1.0)
            vals = p.copy()
            vals[unknown] = np.nan
            out[k] = AttributeCurve(k, vals, conf, grid, "probability", labels=spec.labels,
                                    meta={"calibrated": k in self.temperatures, "features": self.feature_name,
                                          "unknown_fraction": float(unknown.mean())})
        return out
