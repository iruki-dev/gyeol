"""Training loop for the attribute heads (plain PyTorch, deterministic).

A :class:`FrameExample` holds (T, D) features and, per task, frame targets
with a mask (weak utterance-level labels are simply broadcast over the
voiced frames and masked elsewhere).  Training uses singer-disjoint
train / calibration splits: temperatures and the OOD threshold are fitted on
the calibration singers only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from ..attributes.heads import CalibratedHeads, FrameHeads, MahalanobisOOD, TaskSpec, fit_temperature


@dataclass
class FrameExample:
    features: np.ndarray  # (T, D)
    targets: dict[str, np.ndarray]  # softmax: (T,) int (−1 = ignore); sigmoid / regression: (T, K) float, NaN = ignore
    singer: str = ""


@dataclass
class HeadTrainConfig:
    hidden: int = 128
    epochs: int = 30
    lr: float = 2e-3
    weight_decay: float = 1e-4
    seed: int = 0
    ood_quantile: float = 0.99
    task_weights: dict[str, float] = field(default_factory=dict)


def _loss(logits: dict[str, torch.Tensor], ex: FrameExample, tasks: dict[str, TaskSpec], weights: dict[str, float]) -> torch.Tensor:
    total = torch.zeros(())
    for k, spec in tasks.items():
        if k not in ex.targets:
            continue
        z = logits[k][0]
        tgt = ex.targets[k]
        w = weights.get(k, 1.0)
        if spec.kind == "softmax":
            t = torch.tensor(tgt, dtype=torch.long)
            m = t >= 0
            if m.any():
                total = total + w * F.cross_entropy(z[m], t[m])
        elif spec.kind == "regression":
            t = torch.tensor(np.asarray(tgt, float).reshape(len(tgt), spec.n), dtype=torch.float32)
            m = ~torch.isnan(t)
            if m.any():
                mu, logvar = z[:, : spec.n], z[:, spec.n :].clamp(-12, 12)
                total = total + w * F.gaussian_nll_loss(mu[m], t[m], logvar.exp()[m])
        else:
            t = torch.tensor(tgt, dtype=torch.float32)
            m = ~torch.isnan(t)
            if m.any():
                total = total + w * F.binary_cross_entropy_with_logits(z[m], t[m])
    return total


def train_heads(train: list[FrameExample], calib: list[FrameExample], tasks: dict[str, TaskSpec],
                config: HeadTrainConfig | None = None, feature_name: str = "dsp") -> CalibratedHeads:
    cfg = config or HeadTrainConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = FrameHeads(train[0].features.shape[1], tasks, hidden=cfg.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for _ in range(cfg.epochs):
        model.train()
        for i in rng.permutation(len(train)):
            ex = train[i]
            logits, _ = model(torch.tensor(ex.features, dtype=torch.float32)[None])
            loss = _loss(logits, ex, tasks, cfg.task_weights)
            if loss.requires_grad:
                opt.zero_grad()
                loss.backward()
                opt.step()
    return calibrate_heads(model, train, calib, tasks, cfg.ood_quantile, feature_name)


def calibrate_heads(model: FrameHeads, train: list[FrameExample], calib: list[FrameExample], tasks: dict[str, TaskSpec],
                    ood_quantile: float = 0.99, feature_name: str = "dsp") -> CalibratedHeads:
    """Temperatures / variance scales on held-out (calibration) singers and the OOD detectors."""
    heads = CalibratedHeads(model, feature_name=feature_name)
    # temperatures on held-out (calibration) singers
    model.eval()
    with torch.no_grad():
        outs = [model(torch.tensor(ex.features, dtype=torch.float32)[None]) for ex in calib]
    for k, spec in tasks.items():
        if spec.kind == "regression":  # variance scaling on held-out singers (Levi et al. 2019)
            r2, sig = [], []
            for (logits, _), ex in zip(outs, calib):
                if k not in ex.targets:
                    continue
                t = np.asarray(ex.targets[k], float).reshape(len(ex.targets[k]), spec.n)
                z = logits[k][0].numpy()
                mu, var = z[:, : spec.n], np.exp(np.clip(z[:, spec.n :], -12, 12))
                m = np.isfinite(t)
                r2.append(((t - mu) ** 2 / var)[m])
                sig.append(np.sqrt(var)[m])
            if r2 and sum(len(v) for v in r2):
                scale = float(np.mean(np.concatenate(r2)))
                heads.variance_scale[k] = scale
                heads.sigma_ref[k] = float(np.median(np.concatenate(sig)) * np.sqrt(scale))
            continue
        zs, ts = [], []
        for (logits, _), ex in zip(outs, calib):
            if k not in ex.targets:
                continue
            tgt = ex.targets[k]
            if spec.kind == "softmax":
                m = tgt >= 0
                zs.append(logits[k][0][torch.tensor(m)])
                ts.append(torch.tensor(tgt[m], dtype=torch.long))
            else:  # element-wise mask: some qualities may be unlabelled in a corpus
                m = ~np.isnan(tgt)
                zs.append(logits[k][0][torch.tensor(m)])
                ts.append(torch.tensor(tgt[m], dtype=torch.float32))
        if zs and sum(len(t) for t in ts):
            heads.temperatures[k] = fit_temperature(torch.cat(zs), torch.cat(ts), spec.kind)
    # OOD detector: fit on training embeddings, threshold on calibration embeddings
    lab_task = next((k for k, s in tasks.items() if s.kind == "softmax"), None)
    with torch.no_grad():
        tr_emb, tr_lab = [], []
        for ex in train:
            _, e = model(torch.tensor(ex.features, dtype=torch.float32)[None])
            lab = ex.targets.get(lab_task, np.zeros(len(ex.features), int)) if lab_task else np.zeros(len(ex.features), int)
            keep = lab >= 0
            tr_emb.append(e[0].numpy()[keep])
            tr_lab.append(lab[keep])
        cal_emb = np.concatenate([o[1][0].numpy() for o in outs])
    heads.ood = MahalanobisOOD().fit(np.concatenate(tr_emb), np.concatenate(tr_lab)).calibrate(cal_emb, ood_quantile)
    # input-space detector: the trunk's LayerNorm hides input scale from the embedding
    tr_in = np.concatenate([ex.features for ex in train])
    cal_in = np.concatenate([ex.features for ex in calib])
    heads.input_ood = MahalanobisOOD().fit(tr_in, np.zeros(len(tr_in), int), shrinkage=1e-2).calibrate(cal_in, ood_quantile)
    return heads
