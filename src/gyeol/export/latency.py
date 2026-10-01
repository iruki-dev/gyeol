"""Latency profiles for server inference.

* :func:`profile` — median / p90 / p99 wall time of any callable over several
  input sizes, with the real-time factor (compute seconds per audio second);
* :func:`profile_onnx` — the same for an ONNX Runtime session;
* :func:`pipeline_profile` — per-stage time of the signal-layer analysis
  (``rep.meta["timings_s"]``, recorded by :func:`gyeol.attributes.extract.analyze`)
  and of the explanation, for one recording.

Numbers depend on the machine; the report records CPU count, thread count
and library versions so profiles can be compared.
"""

from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np


@dataclass
class LatencyRow:
    label: str
    audio_seconds: float
    median_ms: float
    p90_ms: float
    p99_ms: float
    rtf: float  # median compute time / audio duration


@dataclass
class LatencyProfile:
    name: str
    rows: list[LatencyRow]
    environment: dict = field(default_factory=dict)

    def table(self) -> str:
        head = f"{'case':28s} {'audio s':>8s} {'median ms':>10s} {'p90 ms':>9s} {'p99 ms':>9s} {'RTF':>7s}"
        lines = [f"# {self.name}", head]
        for r in self.rows:
            lines.append(f"{r.label:28s} {r.audio_seconds:8.2f} {r.median_ms:10.2f} {r.p90_ms:9.2f} {r.p99_ms:9.2f} {r.rtf:7.3f}")
        return "\n".join(lines)


def environment() -> dict:
    import torch

    env = {"python": platform.python_version(), "machine": platform.machine(), "cpus": os.cpu_count(),
           "torch": torch.__version__, "torch_threads": torch.get_num_threads()}
    try:
        import onnxruntime as ort

        env["onnxruntime"] = ort.__version__
    except ImportError:
        pass
    return env


def _time(fn: Callable[[], Any], n_warmup: int, n_runs: int) -> np.ndarray:
    for _ in range(n_warmup):
        fn()
    ts = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return np.array(ts) * 1000.0


def profile(name: str, make_call: Callable[[float], Callable[[], Any]], durations_s: Sequence[float] = (1.0, 5.0, 10.0),
            n_warmup: int = 1, n_runs: int = 5) -> LatencyProfile:
    """``make_call(seconds)`` returns a zero-argument callable processing that much audio."""
    rows = []
    for d in durations_s:
        ms = _time(make_call(d), n_warmup, n_runs)
        med = float(np.median(ms))
        rows.append(LatencyRow(f"{name} {d:g}s", d, med, float(np.percentile(ms, 90)), float(np.percentile(ms, 99)), med / 1000.0 / d))
    return LatencyProfile(name, rows, environment())


def profile_onnx(path: str, make_feeds: Callable[[float], dict], durations_s: Sequence[float] = (1.0, 5.0, 10.0), n_warmup: int = 1,
                 n_runs: int = 5, threads: int | None = None) -> LatencyProfile:
    import onnxruntime as ort

    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    prof = profile(f"onnx:{os.path.basename(str(path))}", lambda d: (lambda feeds=make_feeds(d): sess.run(None, feeds)), durations_s,
                   n_warmup, n_runs)
    prof.environment["ort_threads"] = threads or "default"
    return prof


def pipeline_profile(audio: np.ndarray, sr: int, target=None, trackers=None, n_runs: int = 3) -> dict[str, float]:
    """Median seconds per analysis stage (+ explanation) for one recording."""
    from ..attributes.extract import analyze
    from ..core.containers import Recording
    from ..explain import explain

    stages: dict[str, list[float]] = {}
    for _ in range(n_runs):
        t0 = time.perf_counter()
        rep = analyze(Recording(np.asarray(audio, float), sr), trackers=trackers)
        total = time.perf_counter() - t0
        if not rep.usable:
            raise ValueError(f"analysis failed: {rep.reason}")
        for k, v in rep.value.meta.get("timings_s", {}).items():
            stages.setdefault(k, []).append(v)
        stages.setdefault("analyze_total", []).append(total)
        if target is not None:
            t1 = time.perf_counter()
            explain([rep.value], target)
            stages.setdefault("explain", []).append(time.perf_counter() - t1)
    out = {k: float(np.median(v)) for k, v in stages.items()}
    out["audio_seconds"] = len(audio) / sr
    return out
