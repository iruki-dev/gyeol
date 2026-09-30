"""Frame encoders: frozen SSL adapters (explicit local checkpoints) and the DSP baseline (M3).
Singer / env / residual encoders arrive in M4."""

from .frame import DSP_FEATURE_CURVES, DSPFrameFeatures, FrameEncoder, TorchSSLEncoder, ssl_from_checkpoint

__all__ = ["DSP_FEATURE_CURVES", "DSPFrameFeatures", "FrameEncoder", "TorchSSLEncoder", "ssl_from_checkpoint"]
