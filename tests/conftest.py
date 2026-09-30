import numpy as np
import pytest

from gyeol._dsp import n_frames
from gyeol.features.pitch import consensus

SR = 16000
HOP = 160


@pytest.fixture
def analyse_pitch():
    def run(x, sr=SR, hop=HOP):
        n = n_frames(len(x), hop)
        return n, consensus(x, sr, hop, n)

    return run


def add_white_noise(x, snr_db, seed=0):
    rng = np.random.default_rng(seed)
    return x + rng.standard_normal(len(x)) * np.std(x) * 10 ** (-snr_db / 20)
