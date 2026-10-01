"""Autoencoder training step (M4).

Generator-side objective per batch::

    L = λ_mel · weighted mel L1          (consonant / low-energy frames up-weighted)
      + λ_stft · multi-resolution STFT  (waveform, if the vocoder runs)
      + λ_kl · KL(r)                    (VIB bottleneck)
      + λ_env · CE(env heads, augmentation labels)
      + λ_spk · SupCon(singer vectors, singer ids)
      + λ_leak · adversarial leakage    (heads on r and on the singer vector behind GRL:
                                          they learn to predict, the encoders learn to defeat them)
      + λ_re · re-encoding consistency  (r(decode(x)) ≈ r(x))
      + λ_adv · (LSGAN generator + feature matching)   if discriminators are given

The DSP attribute curves are not differentiable, so ``c`` re-encoding
consistency is an *evaluation* (:func:`gyeol.eval.reconstruction.reencoding_consistency`);
the differentiable r consistency is part of training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from ..core.containers import Representation
from ..decoder.model import C_CHANNELS, GyeolAutoencoder
from ..encoders.latent import supervised_contrastive
from .losses import (
    MultiResolutionSTFTLoss,
    discriminator_loss,
    feature_matching_loss,
    frame_weights,
    generator_adv_loss,
    weighted_mel_loss,
)

_SCALE = {"pitch_center": 1 / 1200.0, "loudness_rel": 0.1, "aperiodic_ratio": 0.1, "vibrato_extent": 0.02, "vibrato_rate": 0.1}


@dataclass
class AEBatch:
    wav: torch.Tensor  # (B, N) target audio
    c: torch.Tensor  # (B, T, Dc) scaled attribute values
    c_mask: torch.Tensor  # (B, T, Dc) bool: attribute known
    f0_hz: torch.Tensor  # (B, T), 0 = unvoiced
    aperiodic: torch.Tensor  # (B, T) aperiodic ratio / 10
    weights: torch.Tensor  # (B, T) frame loss weights
    wav_for_residual: torch.Tensor | None = None  # timbre-shifted copy
    rough: torch.Tensor | None = None  # (B, T) 0..1
    singer_id: torch.Tensor | None = None  # (B,)
    env_classes: dict[str, torch.Tensor] = field(default_factory=dict)  # name -> (B,)


def batch_from_representations(reps: list[Representation], wavs: list[np.ndarray], singer_ids: list[int] | None = None,
                               env_classes: list[dict[str, int]] | None = None, wavs_for_residual: list[np.ndarray] | None = None) -> AEBatch:
    """Pad a list of analysed recordings into an :class:`AEBatch` (shared grid required)."""
    grid = reps[0].grid
    for r in reps:
        if not r.grid.same_axis(grid):
            raise ValueError("all representations in a batch must share sample rate and hop")
    T = max(r.grid.n_frames for r in reps)
    N = (T - 1) * grid.hop
    B = len(reps)
    c = np.zeros((B, T, len(C_CHANNELS)))
    m = np.zeros((B, T, len(C_CHANNELS)), bool)
    f0 = np.zeros((B, T))
    ap = np.zeros((B, T))
    w = np.zeros((B, T))
    rough = np.zeros((B, T))
    wav = np.zeros((B, N))
    for i, rep in enumerate(reps):
        t = rep.grid.n_frames
        for j, name in enumerate(C_CHANNELS):
            cur = rep.curves[name]
            c[i, :t, j] = np.nan_to_num(cur.values) * _SCALE.get(name, 1.0)
            m[i, :t, j] = cur.confidence > 0
        fc = rep.curves["f0_cents"]
        f0[i, :t] = np.where(fc.confidence > 0, 440.0 * 2 ** (np.nan_to_num(fc.values) / 1200), 0.0)
        ap[i, :t] = np.nan_to_num(rep.curves["aperiodic_ratio"].values) * 0.1
        voiced = f0[i, :t] > 0
        w[i, :t] = frame_weights(rep.curves["loudness_rel"].values, voiced)
        rough[i, :t] = np.clip(np.nan_to_num(rep.curves["subharmonic_ratio"].values) / 0.5, 0, 1)
        n = min(N, len(wavs[i]))
        wav[i, :n] = wavs[i][:n]
    tt = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt)  # noqa: E731
    wr = None
    if wavs_for_residual is not None:
        wr = np.zeros((B, N))
        for i, a in enumerate(wavs_for_residual):
            wr[i, : min(N, len(a))] = a[:N]
        wr = tt(wr)
    return AEBatch(tt(wav), tt(c), tt(m, torch.bool), tt(f0), tt(ap), tt(w), wr, tt(rough),
                   None if singer_ids is None else tt(singer_ids, torch.long),
                   {} if env_classes is None else {k: tt([e[k] for e in env_classes], torch.long) for k in env_classes[0]})


@dataclass
class LossWeights:
    mel: float = 45.0
    stft: float = 1.0
    kl: float = 0.01
    env: float = 1.0
    singer: float = 0.5
    leak: float = 1.0
    reencode: float = 1.0
    adv: float = 1.0
    fm: float = 2.0


def generator_step(model: GyeolAutoencoder, batch: AEBatch, w: LossWeights | None = None, vocode: bool = True,
                   discriminators: list | None = None, mrstft: MultiResolutionSTFTLoss | None = None,
                   outputs: dict | None = None) -> dict[str, torch.Tensor]:
    """Generator-side losses; if ``outputs`` is a dict, the decoded waveform and its target are stored in it
    (``"wav"``, ``"target"``) so the caller can run the discriminator step without decoding again."""
    w = w or LossWeights()
    enc = model.encode(batch.wav, batch.c, batch.c_mask, batch.wav_for_residual)
    T = enc["mel"].shape[1]
    c, cm, f0, ap = batch.c[:, :T], batch.c_mask[:, :T], batch.f0_hz[:, :T], batch.aperiodic[:, :T]
    fw = batch.weights[:, :T]
    rough = None if batch.rough is None else batch.rough[:, :T]
    dec = model.decode(enc["singer"], enc["env"], c, cm, enc["r"], f0, ap, rough, vocode=vocode)
    losses: dict[str, torch.Tensor] = {}
    losses["mel"] = w.mel * weighted_mel_loss(dec["mel"], enc["mel"].detach(), fw)
    losses["kl"] = w.kl * enc["kl"].mean()
    # env supervision
    if batch.env_classes:
        losses["env"] = w.env * sum(F.cross_entropy(enc["env_logits"][k], y) for k, y in batch.env_classes.items())
    # singer contrastive
    if batch.singer_id is not None and len(torch.unique(batch.singer_id)) < len(batch.singer_id):
        losses["singer"] = w.singer * supervised_contrastive(enc["singer"], batch.singer_id)
    # leakage adversaries (GRL inside the heads): they learn to predict, encoders learn to fool them
    reg, _ = model.leak(enc["r"])
    voiced = f0 > 0
    leak = torch.zeros(())
    targets = {"f0": torch.where(voiced, torch.log(f0.clamp_min(1.0) / 440.0), torch.zeros_like(f0)), "aperiodic": ap,
               "loudness": c[..., list(C_CHANNELS).index("loudness_rel")]}
    for k, pred in reg.items():
        if k in targets:
            mask = voiced if k != "loudness" else torch.ones_like(voiced)
            if mask.any():
                leak = leak + F.mse_loss(pred[..., 0][mask], targets[k][mask])
    # utterance-level adversaries on the time-pooled residual and on the singer vector
    _, cls_pooled = model.leak(enc["r"].mean(1))
    if batch.singer_id is not None and "singer" in cls_pooled:
        leak = leak + F.cross_entropy(cls_pooled["singer"], batch.singer_id)
    if batch.env_classes:
        _, s_cls = model.singer_env_adv(enc["singer"])
        for k, y in batch.env_classes.items():
            if f"env_{k}" in cls_pooled:
                leak = leak + F.cross_entropy(cls_pooled[f"env_{k}"], y)
            if k in s_cls:
                leak = leak + F.cross_entropy(s_cls[k], y)
    losses["leak"] = w.leak * leak
    if vocode:
        y = dec["wav"]
        x = batch.wav[:, : y.shape[1]]
        if outputs is not None:
            outputs["wav"], outputs["target"] = y, x
        mr = mrstft or MultiResolutionSTFTLoss().to(y.device)
        losses["stft"] = w.stft * mr(y, x, fw, model.cfg.hop)
        # re-encoding consistency on r
        re = model.encode(y, c, cm)
        Tm = min(re["r"].shape[1], enc["r"].shape[1])
        losses["reencode"] = w.reencode * (re["r"][:, :Tm] - enc["r"][:, :Tm].detach()).abs().mean()
        if discriminators:
            real = [o for d in discriminators for o in d(x)]
            fake = [o for d in discriminators for o in d(y)]
            losses["adv"] = w.adv * generator_adv_loss(fake)
            losses["fm"] = w.fm * feature_matching_loss(real, fake)
    losses["total"] = sum(losses.values())
    return losses


def discriminator_step(discriminators: list, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    r = [o for d in discriminators for o in d(real)]
    f = [o for d in discriminators for o in d(fake.detach())]
    return discriminator_loss(r, f)
