"""Residual embedding z (see :mod:`gyeol.residual.base`; the PyTorch model
lives in :mod:`gyeol.residual.torch_model` and needs ``torch``)."""

from .base import CORE_INPUTS, ResidualEncoder, core_matrix

__all__ = ["CORE_INPUTS", "ResidualEncoder", "core_matrix"]
