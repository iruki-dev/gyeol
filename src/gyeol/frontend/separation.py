"""Singing-voice separation adapters.

gyeol ships no separator weights.  Adapters wrap separators the user has
installed and fetched, and every adapter names its license-registry asset so
the active profile is enforced:

* :class:`CallableSeparator` – any ``f(audio, sr) -> vocals`` (e.g. a Mel- or
  BS-RoFormer inference function); pass the asset name for its weights
  (``"roformer_community"`` by default, whose training-data provenance is
  flagged as unclear).
* :class:`DemucsSeparator` – HTDemucs through the optional ``demucs`` package.
* :class:`BackingTrackCanceller` – the sing-along case: the backing track is
  known, so its leakage is cancelled by least squares per frequency.

Two separators can be compared with :func:`separation_agreement`
(symmetric SI-SDR), a cheap proxy for local separation reliability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import signal

from ..core.license import Profile, lookup, require_allowed
from ..core.status import Result
from ..dsp.base import resample, si_sdr, to_mono

EPS = 1e-12


class CallableSeparator:
    def __init__(self, fn: Callable[[np.ndarray, int], np.ndarray], name: str = "roformer", asset: str = "roformer_community",
                 profile: Profile = Profile.COMMERCIAL):
        require_allowed(lookup(asset), profile, announce=False)
        self.fn, self.name, self.asset = fn, name, asset

    def separate(self, audio: np.ndarray, sr: int) -> Result[np.ndarray]:
        try:
            out = to_mono(np.asarray(self.fn(audio, sr), dtype=float))
        except Exception as exc:  # noqa: BLE001 - surface separator failures as status
            return Result.failure(f"{self.name} failed: {exc}")
        if out.shape[0] != len(audio):
            return Result.failure(f"{self.name} returned {out.shape[0]} samples for {len(audio)} input samples")
        return Result.success(out)


class DemucsSeparator:
    name = "htdemucs"

    def __init__(self, model: str = "htdemucs", device: str = "cpu", asset: str | None = None, profile: Profile = Profile.COMMERCIAL):
        # HTDemucs is not yet in the registry; the caller must register its weights' license
        if asset is None:
            raise ValueError("register the Demucs weights in the license registry and pass asset=<name>")
        require_allowed(lookup(asset), profile, announce=False)
        self.model, self.device = model, device

    def separate(self, audio: np.ndarray, sr: int) -> Result[np.ndarray]:
        try:
            import torch
            from demucs.apply import apply_model
            from demucs.pretrained import get_model
        except ImportError:
            return Result.unavailable("demucs is not installed")
        m = get_model(self.model)
        m.eval()
        x = resample(to_mono(audio), sr, m.samplerate)
        wav = torch.tensor(np.stack([x, x]), dtype=torch.float32)[None]
        with torch.no_grad():
            out = apply_model(m, wav, device=self.device)[0]
        voc = out[m.sources.index("vocals")].mean(0).cpu().numpy()
        return Result.success(resample(voc, m.samplerate, sr)[: len(audio)])


@dataclass
class BackingTrackCanceller:
    backing: np.ndarray
    backing_sr: int
    nperseg: int = 2048
    max_lag_s: float = 1.0
    name: str = "backing-cancel"

    def separate(self, audio: np.ndarray, sr: int) -> Result[np.ndarray]:
        mix = to_mono(np.asarray(audio, float))
        b = resample(to_mono(self.backing), self.backing_sr, sr)
        n = min(len(mix), len(b))
        if n < 4 * self.nperseg:
            return Result.failure("signals too short for cancellation")
        max_lag = int(self.max_lag_s * sr)
        cc = signal.correlate(mix[:n], b[:n], mode="full", method="fft")
        lags = np.arange(-n + 1, n)
        sel = np.abs(lags) <= max_lag
        lag = int(lags[sel][np.argmax(np.abs(cc[sel]))])
        al = np.zeros(len(mix))
        if lag >= 0:
            seg = b[: len(mix) - lag]
            al[lag : lag + len(seg)] = seg
        else:
            seg = b[-lag : -lag + len(mix)]
            al[: len(seg)] = seg
        _, _, M = signal.stft(mix, sr, nperseg=self.nperseg)
        _, _, B = signal.stft(al, sr, nperseg=self.nperseg)
        H = np.sum(M * np.conj(B), axis=1) / (np.sum(np.abs(B) ** 2, axis=1) + EPS)
        _, v = signal.istft(M - H[:, None] * B, sr, nperseg=self.nperseg)
        return Result.success(v[: len(mix)])


def separation_agreement(stem_a: np.ndarray, stem_b: np.ndarray) -> float:
    """Symmetric SI-SDR (dB) between two separators' vocal stems."""
    return 0.5 * (si_sdr(stem_a, stem_b) + si_sdr(stem_b, stem_a))


# ---------------------------------------------------------------- accompaniment detection (revision A1)


@dataclass
class AccompanimentPolicy:
    """Decision thresholds for "may contain accompaniment" (``separation="auto"``).

    Defaults come from synthetic voice / voice+backing mixtures (see
    ``docs/revisions/A.md``) and are meant to be re-fitted on the real-recording
    set (``gyeol eval realset``).  They err towards separating.
    """

    residual_db: float = -28.0  # energy outside the voice's own harmonic series, re its harmonics (75th pct.)
    residual_flatness: float = 0.02  # … and that residual is tonal (instrument partials), not flat (noise)
    tonal_gaps: float = 0.3  # share of loud frames that are tonal while no stable voice is found
    low_freq_fraction: float = 5e-4  # median energy share below 90 Hz in loud frames
    min_frames: int = 8


@dataclass
class AccompanimentEstimate:
    may_contain: bool
    residual_db: float | None  # ≲ −30 dB for a single voice (vibrato, breath, low or high range); ≳ −27 dB with backing
    residual_flatness: float | None  # ≥ 0.03 for voice leakage, ≥ 0.5 for noise, ≤ 0.01 for instrument partials
    tonal_gaps: float
    low_freq_fraction: float
    reasons: list[str]


def estimate_accompaniment(audio: np.ndarray, sr: int, policy: AccompanimentPolicy | None = None) -> Result[AccompanimentEstimate]:
    """Does the input look like more than one voice? (cheap; runs before separation)

    Pitch-informed: a fast YIN track gives the dominant f0.  In loud frames
    where it is stable (±50 cents over the 128 ms window), the energy outside
    that one harmonic series (harmonics 1–16 up to 4 kHz, ±1.5 bins + 3 %) is
    measured relative to the series itself, together with the spectral
    flatness of that residual: a lone voice leaves little, and what it leaves
    (breath, room noise) is flat; chords and bass leave tonal partials.  Loud
    frames with no stable voice that are still tonal (backing playing through
    the singer's rests) and energy below 90 Hz are two more cues.
    """
    from ..pitch.adapters import YinTracker

    pol = policy or AccompanimentPolicy()
    x = to_mono(np.asarray(audio, float))
    if len(x) < sr // 4 or not np.all(np.isfinite(x)):
        return Result.failure("need at least 250 ms of finite audio to estimate accompaniment")
    fs = 16000
    xs = resample(x - np.mean(x), sr, fs)
    tr = YinTracker().track(xs, fs)
    if not tr.ok:
        return Result.failure(f"pitch pre-pass failed: {tr.reason}")
    f, _, Z = signal.stft(xs, fs, nperseg=2048, noverlap=1536)
    P = np.abs(Z) ** 2
    tc = np.arange(P.shape[1]) * 512 / fs
    f0_at = lambda dt: np.interp(tc + dt, tr.value.times, np.nan_to_num(tr.value.f0_hz, nan=0.0))  # noqa: E731
    f0, lo, hi = f0_at(0.0), f0_at(-0.064), f0_at(0.064)
    stable = (f0 > 70) & (lo > 70) & (hi > 70) & (np.abs(1200 * np.log2(np.maximum(lo, 1) / np.maximum(hi, 1))) < 50)
    L = 10 * np.log10(P.sum(0) + EPS)
    loud = L > np.percentile(L, 95) - 20
    if loud.sum() < pol.min_frames:
        return Result.failure("too few loud frames")
    df = fs / 2048
    rdb, rflat = [], []
    for j in np.flatnonzero(loud & stable):
        c = f0[j]
        nh = min(int(4000 / c), 16)
        h = np.arange(1, nh + 1) * c
        rng = (f >= 0.5 * c) & (f <= (nh + 0.5) * c)
        comb = np.any(np.abs(f[None, :] - h[:, None]) <= (1.5 * df + 0.03 * h)[:, None], axis=0) & rng
        res = P[rng & ~comb, j] + EPS
        rdb.append(10 * np.log10(res.sum() / (P[comb, j].sum() + EPS)))
        rflat.append(float(np.exp(np.mean(np.log(res))) / np.mean(res)))
    gaps = loud & ~stable & (f0 <= 70)
    band = (f >= 100) & (f <= 4000)
    gp = P[band][:, gaps] + EPS
    gflat = np.exp(np.mean(np.log(gp), axis=0)) / np.mean(gp, axis=0) if gaps.any() else np.zeros(0)
    tonal_gaps = float(np.sum(gflat < 0.05) / loud.sum())
    lf = float(np.median(P[f < 90][:, loud].sum(0) / (P[:, loud].sum(0) + EPS)))
    r_db = float(np.percentile(rdb, 75)) if len(rdb) >= pol.min_frames else None
    r_flat = float(np.median(rflat)) if len(rflat) >= pol.min_frames else None
    reasons = []
    if r_db is not None and r_db >= pol.residual_db and r_flat is not None and r_flat <= pol.residual_flatness:
        reasons.append(f"tonal energy outside the voice's harmonics ({r_db:.1f} dB, flatness {r_flat:.3f})")
    if tonal_gaps >= pol.tonal_gaps:
        reasons.append(f"tonal sound where no voice is found ({tonal_gaps * 100:.0f}% of loud frames)")
    if lf >= pol.low_freq_fraction:
        reasons.append(f"energy below 90 Hz ({lf:.4f})")
    return Result.success(AccompanimentEstimate(bool(reasons), r_db, r_flat, tonal_gaps, lf, reasons))


# ---------------------------------------------------------------- separation quality (revision A1)


@dataclass
class SeparationPolicy:
    sir_low_db: float = -10.0  # frame stem-to-accompaniment ratio at which confidence reaches 0
    sir_high_db: float = 5.0  # … and 1
    max_residual_db: float = -20.0  # leakage of the accompaniment into the stem above which it is flagged
    residual_penalty: float = 0.5  # factor on all frames when flagged
    min_lag_prominence: float = 16.0  # residual counts only with a sharp delay peak (see TakePolicy)
    unseparated_factor: float = 0.5  # factor when accompaniment is suspected but no separator is available …
    unseparated_full_db: float = -15.0  # … applied in full when the estimated residual is at or above this level,
    unseparated_none_db: float = -30.0  # … and fading to 1 as it falls to this level (a faint residual barely matters)


def unseparated_factor(residual_db: float | None, policy: SeparationPolicy | None = None) -> float:
    """Confidence factor for audio that may contain accompaniment and was not separated.

    Graded by the pitch-informed residual level of :func:`estimate_accompaniment`:
    a loud accompaniment gets ``unseparated_factor``, a faint residual (an
    already-separated stem) almost nothing.  Unknown level → the full penalty.
    """
    pol = policy or SeparationPolicy()
    if residual_db is None or not np.isfinite(residual_db):
        return pol.unseparated_factor
    w = np.clip((residual_db - pol.unseparated_none_db) / (pol.unseparated_full_db - pol.unseparated_none_db), 0.0, 1.0)
    return float(1.0 - (1.0 - pol.unseparated_factor) * w)


@dataclass
class SeparationQuality:
    separator: str
    residual_db: float  # accompaniment leakage into the vocal stem (coherence-based, ≈ −30 dB floor)
    residual_reference: str  # "backing" (known track) or "estimate" (mixture − stem)
    frame_sir_db: np.ndarray  # stem vs estimated accompaniment per grid frame
    frame_factor: np.ndarray  # 0..1 multiplier for per-frame confidences
    flags: dict[str, str]


def separation_quality(mixture: np.ndarray, stem: np.ndarray, sr: int, grid, *, backing: np.ndarray | None = None,
                       separator: str = "", policy: SeparationPolicy | None = None) -> SeparationQuality:
    """How much accompaniment is left, globally and per frame.

    * residual leakage uses the coherence bleed check of :mod:`gyeol.frontend.quality`
      — against the known backing track when given, else against the
      accompaniment estimate (mixture − stem);
    * per-frame SIR compares the stem with that estimate: where the backing
      dominates the voice, separation errors dominate too.
    """
    from .quality import detect_bleed

    pol = policy or SeparationPolicy()
    mix, voc = np.asarray(mixture, float), np.asarray(stem, float)
    n = min(len(mix), len(voc))
    acc = mix[:n] - voc[:n]
    ref, ref_name = (np.asarray(backing, float), "backing") if backing is not None else (acc, "estimate")
    b = detect_bleed(voc[:n], ref, sr)
    residual = b.value.bleed_db if b.usable else float("nan")
    if b.usable and not b.value.lag_prominence >= pol.min_lag_prominence:
        # no sharp delay peak: the coherence comes from pitches the voice shares with the accompaniment,
        # not from leakage — report the resolution floor
        residual = min(residual, -30.0)
    win = int(0.046 * sr)
    centres = np.round(grid.times() * sr).astype(int)

    def frame_db(x):
        xp = np.pad(x, (win // 2, win))
        fr = xp[centres[:, None] + np.arange(win)[None, :]]
        return 10 * np.log10(np.mean(fr**2, axis=1) + EPS)

    sir = frame_db(voc[:n]) - frame_db(acc)
    factor = np.clip((sir - pol.sir_low_db) / (pol.sir_high_db - pol.sir_low_db), 0.0, 1.0)
    flags = {}
    if np.isfinite(residual) and residual > pol.max_residual_db:
        flags["residual_accompaniment"] = f"accompaniment leakage {residual:.1f} dB in the vocal stem"
        factor = factor * pol.residual_penalty
    return SeparationQuality(separator, float(residual), ref_name, sir, factor, flags)


_SEPARATOR_CACHE: dict = {}


def default_separator(profile: Profile = Profile.COMMERCIAL, backing: np.ndarray | None = None, backing_sr: int | None = None) -> Result:
    """The separator ``analyze`` uses when none is given (never downloads):

    1. the known backing track → :class:`BackingTrackCanceller`;
    2. fetched BS-RoFormer weights in gyeol's cache → :class:`~gyeol.frontend.roformer.RoFormerSeparator`;
    3. otherwise unavailable.
    """
    if backing is not None:
        return Result.success(BackingTrackCanceller(np.asarray(backing, float), backing_sr or 44100))
    key = ("roformer", Profile(profile))
    if key in _SEPARATOR_CACHE:
        return _SEPARATOR_CACHE[key]
    from .roformer import RoFormerSeparator

    r = RoFormerSeparator.from_cache(profile=profile)
    if r.ok:  # only successes are cached: a fetch later in the process is picked up
        _SEPARATOR_CACHE[key] = r
    return r


# ---------------------------------------------------------------- separator choice and the target cache (revision D1)
#
# The target song is separated once — asynchronously, at upload — with the heavy BS-RoFormer, and the stems are
# cached by content hash.  User takes are assumed to be recorded on headphones: ``analyze(separation="auto")``
# separates a take only when accompaniment bleed into the microphone is detected against the cached
# accompaniment, and then with a selectable *light* separator (by default the backing canceller, which subtracts
# the known accompaniment).

#: separator names → "heavy" (target songs) or "light" (user takes)
SEPARATORS = {"bs_roformer": "heavy", "roformer_light": "light", "htdemucs": "light", "backing": "light"}


def make_separator(name, profile: Profile = Profile.COMMERCIAL, *, accompaniment: np.ndarray | None = None,
                   accompaniment_sr: int | None = None, checkpoint: str | None = None, config: dict | None = None,
                   asset: str | None = None) -> Result:
    """A separator by name (or pass an object with ``separate(audio, sr)`` through).

    * ``bs_roformer`` — fetched viperx BS-RoFormer weights (heavy; for target songs);
    * ``roformer_light`` — a smaller BS-RoFormer from a local ``checkpoint`` with its ``config`` and registry ``asset``;
    * ``htdemucs`` — HTDemucs (needs the ``demucs`` package and its weights registered as ``asset``);
    * ``backing`` — subtracts a known ``accompaniment`` (e.g. the cached accompaniment of the target song).

    Never downloads; every weight file passes the license gate for ``profile``.
    """
    if hasattr(name, "separate"):
        return Result.success(name)
    if name not in SEPARATORS:
        raise ValueError(f"unknown separator {name!r}; one of {', '.join(SEPARATORS)}")
    if name == "backing":
        if accompaniment is None:
            return Result.unavailable("the backing canceller needs the accompaniment (separate the target song first)")
        return Result.success(BackingTrackCanceller(np.asarray(accompaniment, float), accompaniment_sr or 44100))
    if name == "bs_roformer":
        return default_separator(profile)
    if name == "roformer_light":
        if not checkpoint:
            return Result.unavailable("roformer_light needs a local checkpoint, its config and its registry asset")
        from .roformer import RoFormerSeparator

        try:
            return Result.success(RoFormerSeparator.from_checkpoint(checkpoint, config, asset=asset or "roformer_community", profile=profile,
                                                                    name="roformer-light"))
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            return Result.failure(f"roformer_light: {exc}")
    try:  # htdemucs
        import demucs  # noqa: F401
    except ImportError:
        return Result.unavailable("htdemucs needs the demucs package")
    if asset is None:
        return Result.unavailable("register the HTDemucs weights in the license registry and pass asset=<name>")
    return Result.success(DemucsSeparator(asset=asset, profile=profile))


@dataclass
class TargetSeparation:
    """Stems of one target song: the vocal and the accompaniment (mixture − vocal), at the mixture's rate."""

    key: str
    vocal: np.ndarray
    accompaniment: np.ndarray
    sr: int
    separator: str
    meta: dict


def content_key(audio: np.ndarray, sr: int, separator_id: str) -> str:
    """SHA-256 over the mono float32 samples, the rate and the separator identity."""
    import hashlib

    h = hashlib.sha256()
    h.update(np.ascontiguousarray(to_mono(np.asarray(audio, float)).astype("<f4")).tobytes())
    h.update(f"|{int(sr)}|{separator_id}".encode())
    return h.hexdigest()


class SeparationCache:
    """File cache of target separations keyed by content hash: ``<root>/<k[:2]>/<k>.npz`` + ``.json`` (atomic writes)."""

    def __init__(self, root):
        from pathlib import Path

        self.root = Path(root)

    def _paths(self, key: str):
        d = self.root / key[:2]
        return d / f"{key}.npz", d / f"{key}.json"

    def get(self, key: str) -> TargetSeparation | None:
        import json

        npz, meta = self._paths(key)
        if not (npz.exists() and meta.exists()):
            return None
        m = json.loads(meta.read_text(encoding="utf-8"))
        with np.load(npz) as z:
            return TargetSeparation(key, z["vocal"].astype(float), z["accompaniment"].astype(float), int(m["sr"]), m["separator"], m)

    def put(self, ts: TargetSeparation) -> None:
        import json
        import os
        import tempfile

        npz, meta = self._paths(ts.key)
        npz.parent.mkdir(parents=True, exist_ok=True)
        for path, write in ((npz, lambda fh: np.savez(fh, vocal=ts.vocal.astype(np.float32), accompaniment=ts.accompaniment.astype(np.float32))),
                            (meta, lambda fh: fh.write(json.dumps({**ts.meta, "sr": ts.sr, "separator": ts.separator, "key": ts.key},
                                                                  default=str).encode()))):
            fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
            try:
                with os.fdopen(fd, "wb") as fh:
                    write(fh)
                os.replace(tmp, path)
            except BaseException:
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise


def separate_target(audio: np.ndarray, sr: int, *, cache: SeparationCache | None = None, separator="bs_roformer",
                    profile: Profile = Profile.COMMERCIAL, separator_id: str | None = None, **kw) -> Result[TargetSeparation]:
    """Separate a target song once; with a ``cache``, the result is stored and later calls return it without separating."""
    import time

    x = to_mono(np.asarray(audio, float))
    sid = separator_id or (separator if isinstance(separator, str) else getattr(separator, "name", type(separator).__name__))
    key = content_key(x, sr, sid)
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            hit.meta["cache"] = "hit"
            return Result.success(hit)
    sep = make_separator(separator, profile, **kw)
    if not sep.ok:
        return Result(sep.status, None, sep.reason)
    t0 = time.perf_counter()
    out = sep.value.separate(x, sr)
    if not out.ok:
        return Result(out.status, None, out.reason)
    vocal = np.asarray(out.value, float)[: len(x)]
    vocal = np.pad(vocal, (0, len(x) - len(vocal)))
    ts = TargetSeparation(key, vocal, x - vocal, int(sr), sid, {"separator_name": getattr(sep.value, "name", sid), "seconds": time.perf_counter() - t0,
                                                                "asset": getattr(sep.value, "asset", None), "profile": Profile(profile).value,
                                                                "weights_sha256": getattr(sep.value, "weights_sha256", None), "cache": "miss"})
    if cache is not None:
        cache.put(ts)
    return Result.success(ts)


class SeparationQueue:
    """Background separation of uploaded target songs (one worker by default; duplicate submissions share a future)."""

    def __init__(self, cache: SeparationCache | None = None, max_workers: int = 1):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Lock

        self.cache = cache
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="gyeol-separation")
        self._pending: dict = {}
        self._lock = Lock()

    def submit(self, audio: np.ndarray, sr: int, *, separator="bs_roformer", profile: Profile = Profile.COMMERCIAL,
               separator_id: str | None = None, **kw):
        sid = separator_id or (separator if isinstance(separator, str) else getattr(separator, "name", type(separator).__name__))
        key = content_key(audio, sr, sid)
        with self._lock:
            fut = self._pending.get(key)
            if fut is None or (fut.done() and not fut.result().ok):
                fut = self._pool.submit(separate_target, audio, sr, cache=self.cache, separator=separator, profile=profile,
                                        separator_id=sid, **kw)
                self._pending[key] = fut
        return fut

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)


class PrecomputedSeparator:
    """Hands back an already separated vocal (e.g. from :class:`SeparationCache`) as if separating."""

    def __init__(self, ts: TargetSeparation):
        self.ts, self.name = ts, f"{ts.separator} (cached)"

    def separate(self, audio: np.ndarray, sr: int) -> Result[np.ndarray]:
        if sr != self.ts.sr or abs(len(audio) - len(self.ts.vocal)) > sr // 100:
            return Result.failure("cached separation does not belong to this recording (rate or length differs)")
        v = self.ts.vocal[: len(audio)]
        return Result.success(np.pad(v, (0, len(audio) - len(v))))


@dataclass
class TakePolicy:
    """When a user take (headphones assumed) is separated: only if accompaniment bleed is detected."""

    bleed_threshold_db: float = -25.0  # detect_bleed resolves leakage down to ≈ −30 dB
    #: …and the leakage must have a sharp delay peak (GCC-PHAT robust z).  On synthetic takes a clean voice
    #: scores 9–12, −30 dB bleed 13–18, −25 dB 21–29, −20 dB 33–45
    min_lag_prominence: float = 16.0
    separator: str = "backing"  # light separator for takes; see SEPARATORS
