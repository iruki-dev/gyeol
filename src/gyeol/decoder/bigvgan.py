"""BigVGAN-v2 fallback vocoder adapter (MIT weights, fetched by the user).

BigVGAN has **no f0 control**: it renders whatever pitch the mel implies, so
it is a fallback for reconstruction, not for pitch-edited demos.  It needs
its own mel configuration (``h.num_mels`` etc.); gyeol's mel must match the
checkpoint's config or the output is wrong.
"""

from __future__ import annotations

import numpy as np

from ..core.license import Profile, lookup, require_allowed
from ..core.status import Result


class BigVGANAdapter:
    name = "bigvgan_v2"
    f0_control = False

    def __init__(self, checkpoint_dir: str, profile: Profile = Profile.COMMERCIAL, device: str = "cpu"):
        require_allowed(lookup("bigvgan_v2"), profile, announce=False)
        self.checkpoint_dir, self.device = checkpoint_dir, device
        self._model = None

    def vocode(self, mel: np.ndarray) -> Result[np.ndarray]:
        """mel (T, M) in the checkpoint's own log-mel convention → waveform."""
        try:
            import bigvgan  # type: ignore
            import torch
        except ImportError:
            return Result.unavailable("the 'bigvgan' package is not installed")
        if self._model is None:
            self._model = bigvgan.BigVGAN.from_pretrained(self.checkpoint_dir, use_cuda_kernel=False).eval().to(self.device)
            self._model.remove_weight_norm()
        with torch.no_grad():
            y = self._model(torch.tensor(mel.T[None], dtype=torch.float32, device=self.device))[0, 0].cpu().numpy()
        return Result.success(y)
