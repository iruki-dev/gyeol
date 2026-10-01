"""Device, precision and thread handling for training (revision B1).

``device="auto"`` resolves to CUDA when it is available, otherwise CPU.  The
CPU path always runs in float32 (no autocast, no half precision); CUDA may use
bfloat16 autocast when asked.  The intra-op thread count is configurable
(``threads=0`` keeps PyTorch's default).
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceSpec:
    device: torch.device
    dtype: torch.dtype
    threads: int
    autocast: bool

    @property
    def is_cpu(self) -> bool:
        return self.device.type == "cpu"

    def describe(self) -> dict:
        return {"device": str(self.device), "dtype": str(self.dtype).replace("torch.", ""), "threads": self.threads, "autocast": self.autocast}


def resolve_device(device: str = "auto", threads: int = 0, mixed_precision: bool = False) -> DeviceSpec:
    """``"auto" | "cpu" | "cuda" | "cuda:N"`` → a :class:`DeviceSpec` (and set the thread count)."""
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device 'cuda' requested but CUDA is not available; use device='auto' or 'cpu'")
    if threads and threads > 0:
        torch.set_num_threads(int(threads))
        os.environ.setdefault("OMP_NUM_THREADS", str(int(threads)))
    autocast = bool(mixed_precision) and dev.type == "cuda"
    return DeviceSpec(dev, torch.float32, torch.get_num_threads(), autocast)


def autocast_context(spec: DeviceSpec):
    """bfloat16 autocast on CUDA when enabled; a no-op on CPU (float32 throughout)."""
    if spec.autocast:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def to_device(obj, spec: DeviceSpec):
    """Move tensors (possibly nested in dicts / lists / dataclass-like objects) to the device as float32."""
    if isinstance(obj, torch.Tensor):
        obj = obj.to(spec.device)
        return obj.float() if obj.is_floating_point() else obj
    if isinstance(obj, dict):
        return {k: to_device(v, spec) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, spec) for v in obj)
    if hasattr(obj, "__dataclass_fields__"):
        for k in obj.__dataclass_fields__:
            setattr(obj, k, to_device(getattr(obj, k), spec))
        return obj
    return obj


__all__ = ["DeviceSpec", "autocast_context", "resolve_device", "to_device"]
