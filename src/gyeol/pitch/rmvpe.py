"""RMVPE (Wei et al., Interspeech 2023) reimplemented from the paper and the
Apache-2.0 reference implementation's published architecture.

Pipeline: 16 kHz audio → log-mel (128 bands, 30–8000 Hz, window 1024, hop
160 = 10 ms, HTK mel) → deep U-Net (5 encoder / 4 intermediate / 5 decoder
stages of residual conv blocks) → 3-channel conv → BiGRU → 360 sigmoid bins
(20 cents each, from 1997.38 cents re 10 Hz ≈ 31.7 Hz) → local weighted
average of the 9 bins around the peak.

Module names and shapes follow the reference layout (``unet.encoder…``,
``cnn``, ``fc.0.gru``, ``fc.1``), so a checkpoint fetched with
``gyeol fetch rmvpe`` loads with ``strict=True``; a mismatch is reported
key by key.  **gyeol ships no RMVPE weights and never downloads them.**
Numerical agreement with the reference model can only be checked once
weights are fetched (``tests/test_m8_*`` runs that check when
``GYEOL_RMVPE_WEIGHTS`` points at them).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.license import Profile, lookup, require_allowed
from ..core.status import Result
from ..dsp.base import resample
from .base import PitchTrack

SR = 16000
HOP = 160
N_MELS = 128
N_BINS = 360
CENTS0 = 1997.3794084376191  # cents re 10 Hz of bin 0
THRESHOLD = 0.03


# ---------------------------------------------------------------- mel front end


def _hz_to_mel_htk(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, float) / 700.0)


def _mel_to_hz_htk(m):
    return 700.0 * (10.0 ** (np.asarray(m, float) / 2595.0) - 1.0)


@lru_cache(maxsize=2)
def mel_filterbank(sr: int = SR, n_fft: int = 1024, n_mels: int = N_MELS, fmin: float = 30.0, fmax: float = 8000.0) -> np.ndarray:
    """Slaney-normalised triangular filters on the HTK mel scale (librosa ``htk=True``)."""
    fft_f = np.linspace(0, sr / 2, n_fft // 2 + 1)
    pts = _mel_to_hz_htk(np.linspace(_hz_to_mel_htk(fmin), _hz_to_mel_htk(fmax), n_mels + 2))
    fdiff = np.diff(pts)
    ramps = pts[:, None] - fft_f[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    w = np.maximum(0, np.minimum(lower, upper))
    return (w * (2.0 / (pts[2 : n_mels + 2] - pts[:n_mels]))[:, None]).astype(np.float32)


class MelSpectrogram(nn.Module):
    def __init__(self, n_mels: int = N_MELS, sr: int = SR, win: int = 1024, hop: int = HOP, fmin: float = 30.0, fmax: float = 8000.0,
                 clamp: float = 1e-5):
        super().__init__()
        self.win, self.hop, self.clamp = win, hop, clamp
        self.register_buffer("window", torch.hann_window(win), persistent=False)
        self.register_buffer("fb", torch.tensor(mel_filterbank(sr, win, n_mels, fmin, fmax)), persistent=False)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """(B, N) → (B, n_mels, T) log-mel, T = N // hop + 1."""
        spec = torch.stft(audio, self.win, self.hop, self.win, self.window, center=True, return_complex=True)
        mag = spec.abs()
        return torch.log(torch.clamp(self.fb @ mag, min=self.clamp))


# ---------------------------------------------------------------- network (reference layout)


class BiGRU(nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_layers: int):
        super().__init__()
        self.gru = nn.GRU(n_in, n_hidden, num_layers=n_layers, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    def __init__(self, cin: int, cout: int, momentum: float = 0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout, momentum=momentum), nn.ReLU(),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout, momentum=momentum), nn.ReLU())
        self.is_shortcut = cin != cout
        if self.is_shortcut:
            self.shortcut = nn.Conv2d(cin, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + (self.shortcut(x) if self.is_shortcut else x)


class ResEncoderBlock(nn.Module):
    def __init__(self, cin: int, cout: int, kernel_size, n_blocks: int = 1, momentum: float = 0.01):
        super().__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList([ConvBlockRes(cin, cout, momentum)] + [ConvBlockRes(cout, cout, momentum) for _ in range(n_blocks - 1)])
        self.kernel_size = kernel_size
        if kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor):
        for c in self.conv:
            x = c(x)
        return (x, self.pool(x)) if self.kernel_size is not None else x


class Encoder(nn.Module):
    def __init__(self, cin: int, in_size: int, n_encoders: int, kernel_size, n_blocks: int, cout: int = 16, momentum: float = 0.01):
        super().__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(cin, momentum=momentum)
        self.layers = nn.ModuleList()
        for _ in range(n_encoders):
            self.layers.append(ResEncoderBlock(cin, cout, kernel_size, n_blocks, momentum))
            cin, cout, in_size = cout, cout * 2, in_size // 2
        self.out_size, self.out_channel = in_size, cout

    def forward(self, x: torch.Tensor):
        skips = []
        x = self.bn(x)
        for layer in self.layers:
            s, x = layer(x)
            skips.append(s)
        return x, skips


class Intermediate(nn.Module):
    def __init__(self, cin: int, cout: int, n_inters: int, n_blocks: int, momentum: float = 0.01):
        super().__init__()
        self.layers = nn.ModuleList([ResEncoderBlock(cin, cout, None, n_blocks, momentum)] +
                                    [ResEncoderBlock(cout, cout, None, n_blocks, momentum) for _ in range(n_inters - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride, n_blocks: int = 1, momentum: float = 0.01):
        super().__init__()
        out_pad = (0, 1) if tuple(stride) == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(cin, cout, 3, stride, 1, out_pad, bias=False),
                                   nn.BatchNorm2d(cout, momentum=momentum), nn.ReLU())
        self.conv2 = nn.ModuleList([ConvBlockRes(cout * 2, cout, momentum)] + [ConvBlockRes(cout, cout, momentum) for _ in range(n_blocks - 1)])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat((self.conv1(x), skip), dim=1)
        for c in self.conv2:
            x = c(x)
        return x


class Decoder(nn.Module):
    def __init__(self, cin: int, n_decoders: int, stride, n_blocks: int, momentum: float = 0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_decoders):
            self.layers.append(ResDecoderBlock(cin, cin // 2, stride, n_blocks, momentum))
            cin //= 2

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, skips[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(self, kernel_size, n_blocks: int, en_de_layers: int = 5, inter_layers: int = 4, cin: int = 1, en_out: int = 16):
        super().__init__()
        self.encoder = Encoder(cin, N_MELS, en_de_layers, kernel_size, n_blocks, en_out)
        self.intermediate = Intermediate(self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks)
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, skips = self.encoder(x)
        return self.decoder(self.intermediate(x), skips)


class E2E(nn.Module):
    """Log-mel (B, 128, T) → bin salience (B, T, 360); T must be a multiple of 32."""

    def __init__(self, n_blocks: int = 4, n_gru: int = 1, kernel_size=(2, 2), en_de_layers: int = 5, inter_layers: int = 4,
                 en_out: int = 16):
        super().__init__()
        self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, 1, en_out)
        self.cnn = nn.Conv2d(en_out, 3, 3, padding=1)
        self.fc = nn.Sequential(BiGRU(3 * N_MELS, 256, n_gru), nn.Linear(512, N_BINS), nn.Dropout(0.25), nn.Sigmoid())

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(x)).transpose(1, 2).flatten(-2)
        return self.fc(x)


# ---------------------------------------------------------------- decoding


def salience_to_cents(salience: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    """Local weighted average of the 9 bins around each frame's peak (cents re 10 Hz; 0 = unvoiced)."""
    s = np.asarray(salience, float)
    mapping = np.pad(CENTS0 + 20.0 * np.arange(N_BINS), (4, 4))
    centre = np.argmax(s, axis=1)
    sp = np.pad(s, ((0, 0), (4, 4)))
    idx = centre[:, None] + np.arange(9)[None, :]
    w = np.take_along_axis(sp, idx, axis=1)
    c = (w * mapping[idx]).sum(1) / np.maximum(w.sum(1), 1e-12)
    return np.where(s.max(1) > threshold, c, 0.0)


def cents_to_hz(cents: np.ndarray) -> np.ndarray:
    return np.where(cents > 0, 10.0 * 2 ** (np.asarray(cents, float) / 1200.0), 0.0)


class RMVPE(nn.Module):
    """Mel front end + E2E; ``forward(audio16k)`` → salience (B, T, 360), T = N // 160 + 1."""

    def __init__(self, **e2e_kw):
        super().__init__()
        self.mel = MelSpectrogram()
        self.model = E2E(**e2e_kw)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        mel = self.mel(audio)
        T = mel.shape[-1]
        pad = 32 * ((T - 1) // 32 + 1) - T
        mel = F.pad(mel, (0, pad), mode="constant")
        return self.model(mel)[:, :T]

    def load_reference_weights(self, path: str | Path) -> None:
        """Load a reference ``rmvpe.pt`` state dict into the E2E part (strict, with a key-by-key report)."""
        sd = torch.load(str(path), map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        res = self.model.load_state_dict(sd, strict=False)
        if res.missing_keys or res.unexpected_keys:
            raise ValueError(f"RMVPE weights do not match the reimplementation: missing {res.missing_keys[:10]}"
                             f"{'…' if len(res.missing_keys) > 10 else ''}, unexpected {res.unexpected_keys[:10]}")


class RMVPETracker:
    """:class:`~gyeol.pitch.base.PitchTracker` around :class:`RMVPE`.

    ``weights_path`` must point at weights the user fetched (``gyeol fetch
    rmvpe`` shows the license first).  ``model=`` injects an already built
    network (tests use tiny random ones).
    """

    name = "rmvpe"
    asset = "rmvpe"

    def __init__(self, weights_path: str | Path | None = None, profile: Profile = Profile.COMMERCIAL, model: RMVPE | None = None,
                 threshold: float = THRESHOLD, max_chunk_s: float = 30.0):
        require_allowed(lookup(self.asset), profile, announce=False)
        if model is None:
            if weights_path is None:
                raise ValueError("RMVPETracker needs weights_path (fetched with `gyeol fetch rmvpe`) or a model")
            if not Path(weights_path).is_file():
                raise FileNotFoundError(f"no RMVPE weights at {weights_path}; fetch them with `gyeol fetch rmvpe` (shows the license first)")
            model = RMVPE()
            model.load_reference_weights(weights_path)
        self.model = model.eval()
        self.threshold = threshold
        self.max_chunk = int(max_chunk_s * SR)

    def track(self, audio: np.ndarray, sr: int) -> Result[PitchTrack]:
        x = np.asarray(audio, float)
        if x.ndim != 1 or len(x) < HOP * 4 or not np.all(np.isfinite(x)):
            return Result.failure("RMVPE needs finite mono audio of at least 40 ms")
        x16 = resample(x, sr, SR).astype(np.float32)
        sal = []
        with torch.no_grad():
            for s in range(0, len(x16), self.max_chunk):  # chunk on frame boundaries
                seg = x16[s : s + self.max_chunk]
                out = self.model(torch.tensor(seg)[None])[0].numpy()
                sal.append(out[: len(seg) // HOP + (1 if s + self.max_chunk >= len(x16) else 0)])
        sal = np.concatenate(sal)
        cents = salience_to_cents(sal, self.threshold)
        f0 = cents_to_hz(cents)
        times = np.arange(len(f0)) * HOP / SR
        return Result.success(PitchTrack(self.name, times, np.where(f0 > 0, f0, np.nan), np.clip(sal.max(1), 0, 1)))
