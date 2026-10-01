"""BS-RoFormer (Lu et al., ICASSP 2024 — band-split rotary transformer) for vocal separation.

A self-contained reimplementation of the architecture whose module layout
matches the MIT reference (lucidrains/BS-RoFormer 0.3.x, the layout used by the
community vocal checkpoints, e.g. viperx's ``model_bs_roformer_ep_317``):

    complex STFT → band split (RMSNorm + Linear per band) → depth × [time
    transformer, frequency transformer] (pre-norm attention with rotary
    embeddings and per-head sigmoid gates, GELU feed-forward) → RMSNorm →
    per-band MLP + GLU mask estimator → complex mask × STFT → iSTFT

so a fetched checkpoint loads key by key (:meth:`BSRoFormer.load_reference_weights`,
strict, with a report of missing / unexpected keys).  Configs use the
reference's keyword names (``dim``, ``depth``, ``freqs_per_bands``, ``stft_hop_length`` …),
so a ZFTurbo-style YAML ``model:`` section can be passed as is.

:class:`RoFormerSeparator` runs long inputs in overlapping chunks with
cross-faded overlap-add and returns the vocal stem.  gyeol ships **no weights**:
``gyeol fetch bs_roformer_viperx_ep317`` shows the license and asks first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.license import Profile, lookup, require_allowed
from ..core.status import Result
from ..dsp.base import resample, to_mono

DEFAULT_FREQS_PER_BANDS = (2,) * 24 + (4,) * 12 + (12,) * 8 + (24,) * 8 + (48,) * 8 + (128, 129)

#: hyper-parameters of viperx's BS-RoFormer vocal model (ZFTurbo config ``model_bs_roformer_ep_317_sdr_12.9755.yaml``)
VIPERX_EP317 = dict(dim=512, depth=12, stereo=True, num_stems=1, time_transformer_depth=1, freq_transformer_depth=1,
                    freqs_per_bands=DEFAULT_FREQS_PER_BANDS, dim_head=64, heads=8, attn_dropout=0.1, ff_dropout=0.1,
                    stft_n_fft=2048, stft_hop_length=441, stft_win_length=2048, stft_normalized=False, mask_estimator_depth=2)
VIPERX_SR = 44100
VIPERX_CHUNK = 352800  # samples per inference chunk (8 s)


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class RotaryEmbedding(nn.Module):
    """Interleaved-pair rotary embedding (positions 0..n−1, θ = 10000); ``freqs`` is a (frozen) parameter."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.freqs = nn.Parameter(1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim)), requires_grad=False)

    def rotate(self, t: torch.Tensor) -> torch.Tensor:
        """t: (..., n, d) → rotated along the sequence axis −2."""
        n = t.shape[-2]
        ang = torch.arange(n, device=t.device, dtype=self.freqs.dtype)[:, None] * self.freqs[None, :]
        ang = ang.repeat_interleave(2, dim=-1).to(t.dtype)
        x = t.unflatten(-1, (-1, 2))
        rot = torch.stack((-x[..., 1], x[..., 0]), dim=-1).flatten(-2)
        return t * ang.cos() + rot * ang.sin()


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(RMSNorm(dim), nn.Linear(dim, dim * mult), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * mult, dim),
                                 nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0, rotary_embed: RotaryEmbedding | None = None):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        inner = heads * dim_head
        self.rotary_embed = rotary_embed
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(nn.Linear(inner, dim, bias=False), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        b, n, _ = x.shape
        q, k, v = self.to_qkv(x).view(b, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4)  # (3, b, h, n, d)
        if self.rotary_embed is not None:
            q, k = self.rotary_embed.rotate(q), self.rotary_embed.rotate(k)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        out = out * self.to_gates(x).transpose(1, 2).unsqueeze(-1).sigmoid()
        return self.to_out(out.transpose(1, 2).reshape(b, n, -1))


class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, attn_dropout: float, ff_dropout: float, rotary_embed):
        super().__init__()
        self.layers = nn.ModuleList(nn.ModuleList([Attention(dim, heads, dim_head, attn_dropout, rotary_embed), FeedForward(dim, 4, ff_dropout)])
                                    for _ in range(depth))
        self.norm = nn.Identity()  # norm_output=False in the reference BS-RoFormer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class BandSplit(nn.Module):
    def __init__(self, dim: int, dim_inputs: tuple[int, ...]):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = nn.ModuleList(nn.Sequential(RMSNorm(d), nn.Linear(d, dim)) for d in dim_inputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([f(p) for p, f in zip(x.split(self.dim_inputs, dim=-1), self.to_features)], dim=-2)


def _mlp(dim_in: int, dim_out: int, dim_hidden: int, depth: int) -> nn.Sequential:
    dims = (dim_in, *((dim_hidden,) * (depth - 1)), dim_out)
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class MaskEstimator(nn.Module):
    def __init__(self, dim: int, dim_inputs: tuple[int, ...], depth: int, mlp_expansion_factor: int = 4):
        super().__init__()
        self.to_freqs = nn.ModuleList(nn.Sequential(_mlp(dim, d * 2, dim * mlp_expansion_factor, depth), nn.GLU(dim=-1)) for d in dim_inputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([m(b) for b, m in zip(x.unbind(dim=-2), self.to_freqs)], dim=-1)


class BSRoFormer(nn.Module):
    def __init__(self, dim: int, *, depth: int, stereo: bool = False, num_stems: int = 1, time_transformer_depth: int = 2,
                 freq_transformer_depth: int = 2, freqs_per_bands: tuple[int, ...] = DEFAULT_FREQS_PER_BANDS, dim_head: int = 64,
                 heads: int = 8, attn_dropout: float = 0.0, ff_dropout: float = 0.0, stft_n_fft: int = 2048, stft_hop_length: int = 512,
                 stft_win_length: int = 2048, stft_normalized: bool = False, mask_estimator_depth: int = 2, mlp_expansion_factor: int = 4,
                 linear_transformer_depth: int = 0, **_ignored):
        super().__init__()
        if linear_transformer_depth:
            raise NotImplementedError("linear-attention layers (linear_transformer_depth > 0) are not reimplemented")
        self.stereo, self.audio_channels, self.num_stems = stereo, 2 if stereo else 1, num_stems
        n_freqs = stft_n_fft // 2 + 1
        if sum(freqs_per_bands) != n_freqs:
            raise ValueError(f"freqs_per_bands must sum to {n_freqs} for n_fft={stft_n_fft}, got {sum(freqs_per_bands)}")
        tf = dict(heads=heads, dim_head=dim_head, attn_dropout=attn_dropout, ff_dropout=ff_dropout)
        time_rot, freq_rot = RotaryEmbedding(dim_head), RotaryEmbedding(dim_head)
        self.layers = nn.ModuleList(nn.ModuleList([Transformer(dim, time_transformer_depth, rotary_embed=time_rot, **tf),
                                                   Transformer(dim, freq_transformer_depth, rotary_embed=freq_rot, **tf)]) for _ in range(depth))
        self.final_norm = RMSNorm(dim)
        self.stft_kwargs = dict(n_fft=stft_n_fft, hop_length=stft_hop_length, win_length=stft_win_length, normalized=stft_normalized)
        self.register_buffer("window", torch.hann_window(stft_win_length), persistent=False)
        dims = tuple(2 * f * self.audio_channels for f in freqs_per_bands)
        self.band_split = BandSplit(dim, dims)
        self.mask_estimators = nn.ModuleList(MaskEstimator(dim, dims, mask_estimator_depth, mlp_expansion_factor) for _ in range(num_stems))
        self.checkpointing = False  # gradient checkpointing of the transformer stack (training on CPU, revision B4)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio (B, T) mono or (B, C, T) → stems (B, C, T) (num_stems == 1) or (B, N, C, T)."""
        if audio.ndim == 2:
            audio = audio[:, None]
        b, c, n = audio.shape
        if c != self.audio_channels:
            raise ValueError(f"model expects {self.audio_channels} channel(s), got {c}")
        spec = torch.stft(audio.reshape(b * c, n), window=self.window, return_complex=True, **self.stft_kwargs)  # (b c, f, t)
        f, t = spec.shape[-2:]
        ri = torch.view_as_real(spec).reshape(b, c, f, t, 2).permute(0, 2, 1, 3, 4).reshape(b, f * c, t, 2)  # (b, (f c), t, 2)
        x = ri.permute(0, 2, 1, 3).reshape(b, t, f * c * 2)  # b t (f c ri)
        x = self.band_split(x)  # (b, t, bands, d)
        for time_tr, freq_tr in self.layers:
            x = self._block(time_tr, freq_tr, x)
        x = self.final_norm(x)
        mask = torch.stack([m(x) for m in self.mask_estimators], dim=1)  # (b, s, t, (f c ri))
        mask = mask.reshape(b, self.num_stems, t, f * c, 2).permute(0, 1, 3, 2, 4)  # (b, s, (f c), t, 2)
        out = torch.view_as_complex(ri.unsqueeze(1).contiguous()) * torch.view_as_complex(mask.contiguous())
        out = out.reshape(b, self.num_stems, f, c, t).permute(0, 1, 3, 2, 4).reshape(b * self.num_stems * c, f, t)
        y = torch.istft(out, window=self.window, length=n, **self.stft_kwargs).reshape(b, self.num_stems, c, n)
        return y[:, 0] if self.num_stems == 1 else y

    def _block(self, time_tr, freq_tr, x):
        def run(x):
            b, t, k, d = x.shape
            x = time_tr(x.transpose(1, 2).reshape(b * k, t, d)).reshape(b, k, t, d).transpose(1, 2)
            return freq_tr(x.reshape(b * t, k, d)).reshape(b, t, k, d)

        if self.checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint

            return checkpoint(run, x, use_reentrant=False)
        return run(x)

    def load_reference_weights(self, path: str | Path) -> None:
        """Strict load of a reference checkpoint (plain state dict, or under ``state_dict`` / ``model``; a ``model.`` prefix is stripped)."""
        sd = torch.load(str(path), map_location="cpu", weights_only=True)
        for key in ("state_dict", "model"):
            if isinstance(sd, dict) and isinstance(sd.get(key), dict):
                sd = sd[key]
        sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
        res = self.load_state_dict(sd, strict=False)
        bad = [k for k in res.missing_keys]
        if bad or res.unexpected_keys:
            raise ValueError(f"checkpoint does not match this BS-RoFormer config: missing {bad[:8]}{'…' if len(bad) > 8 else ''}, "
                             f"unexpected {res.unexpected_keys[:8]}{'…' if len(res.unexpected_keys) > 8 else ''}")


def file_sha256(path: str | Path, chunk: int = 1 << 22) -> str:
    """SHA-256 of a weight file (checked against the registry's pinned value on every load)."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class RoFormerSeparator:
    """Vocal stem from a BS-RoFormer, in cross-faded overlapping chunks (the separator protocol: ``separate(audio, sr)``)."""

    def __init__(self, model: BSRoFormer, sr: int = VIPERX_SR, chunk: int = VIPERX_CHUNK, overlap: float = 0.25, asset: str = "bs_roformer_viperx_ep317",
                 profile: Profile = Profile.COMMERCIAL, name: str = "bs-roformer", device: str = "cpu"):
        require_allowed(lookup(asset), profile, announce=False)
        self.model, self.sr, self.chunk, self.overlap, self.asset, self.name, self.device = model.eval(), sr, chunk, overlap, asset, name, device
        self.weights_sha256: str | None = None
        self.model.to(device)

    @classmethod
    def from_checkpoint(cls, path: str | Path, config: dict | None = None, *, asset: str = "bs_roformer_viperx_ep317",
                        profile: Profile = Profile.COMMERCIAL, **kw) -> "RoFormerSeparator":
        require_allowed(lookup(asset), profile, announce=False)  # gate before reading the weights
        if not Path(path).is_file():
            raise FileNotFoundError(f"no separator weights at {path}; fetch them with `gyeol fetch {asset}` (shows the license first)")
        digest = file_sha256(path)
        pinned = lookup(asset).sha256
        if pinned and digest != pinned:
            raise ValueError(f"checksum mismatch for {path}: sha256 {digest} is not the pinned {pinned} of {asset!r}; refusing to load")
        model = BSRoFormer(**(config or VIPERX_EP317))
        model.load_reference_weights(path)
        sep = cls(model, asset=asset, profile=profile, **kw)
        sep.weights_sha256 = digest
        return sep

    @classmethod
    def from_cache(cls, asset: str = "bs_roformer_viperx_ep317", profile: Profile = Profile.COMMERCIAL, **kw) -> Result["RoFormerSeparator"]:
        """The fetched weights of ``asset`` in gyeol's cache, if present (never downloads)."""
        from ..cli import CACHE

        files = sorted((CACHE / asset).glob("*.ckpt")) + sorted((CACHE / asset).glob("*.pt")) + sorted((CACHE / asset).glob("*.pth"))
        if not files:
            return Result.unavailable(f"no fetched weights for {asset!r} in {CACHE / asset} (run `gyeol fetch {asset}`)")
        try:
            return Result.success(cls.from_checkpoint(files[0], asset=asset, profile=profile, **kw))
        except (ValueError, RuntimeError) as exc:
            return Result.failure(f"cannot load {files[0]}: {exc}")

    @torch.no_grad()
    def separate(self, audio: np.ndarray, sr: int) -> Result[np.ndarray]:
        x = to_mono(np.asarray(audio, float))
        if not np.all(np.isfinite(x)) or len(x) < 2048:
            return Result.failure("separator needs finite audio of at least 2048 samples")
        xs = resample(x, sr, self.sr).astype(np.float32)
        n = len(xs)
        chunk = min(self.chunk, n)
        step = max(1, int(chunk * (1 - self.overlap)))
        fade = chunk - step
        win = np.ones(chunk, np.float32)
        if fade > 0:
            ramp = np.linspace(0, 1, fade + 2, dtype=np.float32)[1:-1]
            win[:fade], win[-fade:] = ramp, ramp[::-1]
        out, norm = np.zeros(n, np.float32), np.zeros(n, np.float32)
        starts = list(range(0, max(n - chunk, 0) + 1, step))
        if starts[-1] + chunk < n:
            starts.append(n - chunk)
        ch = self.model.audio_channels
        for s in starts:
            seg = torch.tensor(xs[s : s + chunk], device=self.device)[None, None].expand(1, ch, -1)
            y = self.model(seg)[0].mean(0).cpu().numpy()
            w = win.copy()
            if s == 0:
                w[:fade] = 1.0
            if s + chunk >= n:
                w[-fade or len(w):] = 1.0
            out[s : s + chunk] += y * w
            norm[s : s + chunk] += w
        voc = out / np.maximum(norm, 1e-6)
        voc = resample(voc.astype(float), self.sr, sr)
        return Result.success(np.pad(voc, (0, max(0, len(x) - len(voc))))[: len(x)])


def tiny_config(**kw) -> dict:
    """A CPU-test-sized BS-RoFormer config (n_fft 256 → 129 bins in 8 bands)."""
    base = dict(dim=16, depth=1, stereo=False, time_transformer_depth=1, freq_transformer_depth=1, freqs_per_bands=(4, 4, 8, 8, 16, 16, 32, 41),
                dim_head=8, heads=2, stft_n_fft=256, stft_hop_length=64, stft_win_length=256, mask_estimator_depth=2)
    base.update(kw)
    return base


__all__ = ["BSRoFormer", "DEFAULT_FREQS_PER_BANDS", "RoFormerSeparator", "VIPERX_EP317", "tiny_config"]
