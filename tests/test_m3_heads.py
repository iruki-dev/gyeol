
import numpy as np
import pytest
import torch

from gyeol.attributes.extract import AnalysisConfig
from gyeol.attributes.heads import CalibratedHeads, MahalanobisOOD, default_tasks, expected_calibration_error
from gyeol.core import FrameGrid, Recording, Status
from gyeol.encoders import DSPFrameFeatures, TorchSSLEncoder
from gyeol.eval import leakage, probe_battery, probe_classify, probe_regress
from gyeol.train import load_checkpoint, save_checkpoint
from gyeol.train.heads import FrameExample, HeadTrainConfig, train_heads

from .helpers import dsp_trackers
from .synthetic_corpus import SR, corpus, examples

TASKS = {k: v for k, v in default_tasks().items() if k in ("register", "phonation")}


@pytest.fixture(scope="module")
def data():
    exs, reps = examples(corpus(8, seed=0))
    split = {"train": ("s0", "s2", "s4", "s6"), "calib": ("s1", "s5"), "test": ("s3", "s7")}
    pick = lambda names: [e for e in exs if e.singer in names]  # noqa: E731
    return pick(split["train"]), pick(split["calib"]), pick(split["test"]), exs, reps


@pytest.fixture(scope="module")
def heads(data):
    tr, cal, _, _, _ = data
    return train_heads(tr, cal, TASKS, HeadTrainConfig(hidden=32, epochs=25, seed=0))


def _collect(heads, exs, calibrated=True):
    h = heads if calibrated else CalibratedHeads(heads.model)
    P, Y, B, BY = [], [], [], []
    for e in exs:
        probs, _, _ = h.predict(e.features)
        m = e.targets["register"] >= 0
        P.append(probs["register"][m])
        Y.append(e.targets["register"][m])
        mm = ~np.isnan(e.targets["phonation"][:, 0])
        B.append(probs["phonation"][mm, :2])
        BY.append(e.targets["phonation"][mm, :2])
    return np.concatenate(P), np.concatenate(Y), np.concatenate(B), np.concatenate(BY)


def test_heads_generalise_to_held_out_singers(heads, data):
    _, _, te, _, _ = data
    P, Y, B, BY = _collect(heads, te)
    assert (P.argmax(1) == Y).mean() > 0.6  # chance ≈ 0.33
    assert ((B > 0.5) == (BY > 0.5)).mean() > 0.65


def test_temperature_scaling_improves_calibration(heads, data):
    _, _, te, _, _ = data
    assert set(heads.temperatures) == {"register", "phonation"}
    P, Y, _, _ = _collect(heads, te, calibrated=True)
    R, _, _, _ = _collect(heads, te, calibrated=False)
    assert expected_calibration_error(P, Y) < expected_calibration_error(R, Y)


def test_ood_inputs_are_unknown_with_zero_confidence(heads, data):
    _, _, te, _, _ = data
    feats = te[0].features
    _, unk_in, _ = heads.predict(feats)
    garbage = np.random.default_rng(0).standard_normal(feats.shape) * 3
    _, unk_out, _ = heads.predict(garbage)
    assert unk_in.mean() < 0.1 and unk_out.mean() > 0.9
    g = FrameGrid(44100, 512, len(garbage))
    c = heads.curves(garbage, g)["register"]
    assert c.confidence.max() == 0.0 and np.isnan(c.values).all()


def test_confidence_drops_with_frontend_quality(heads, data):
    _, _, te, _, _ = data
    f = te[0].features
    g = FrameGrid(SR, 512, len(f))
    full = heads.curves(f, g, quality_factor=np.ones(len(f)))["register"].confidence
    half = heads.curves(f, g, quality_factor=np.full(len(f), 0.5))["register"].confidence
    assert np.allclose(half, 0.5 * full)


def test_learned_curves_in_analysis(heads, data):
    it = corpus(1, seed=3)[0]
    cfg = AnalysisConfig(heads=heads, feature_encoder=DSPFrameFeatures())
    from gyeol.attributes.extract import analyze

    rep = analyze(Recording(it["audio"], SR), trackers=dsp_trackers(), config=cfg).unwrap()
    reg = rep.curves["register"]
    assert reg.values.shape[1] == 3 and reg.labels == ("chest", "mixed", "falsetto")
    assert reg.meta["calibrated"] and reg.confidence.max() > 0
    # frames outside the voice carry no register confidence
    voiced = np.isfinite(rep.curves["f0_cents"].values)
    assert reg.confidence[~voiced].max() == 0.0


def test_heads_checkpoint_roundtrip_with_provenance(heads, tmp_path):
    save_checkpoint(tmp_path / "heads.pt", heads.model.state_dict(), name="heads", sources=["vocalset", "aihub_465_multi_singer"],
                    config={"tasks": list(TASKS)})
    state, info = load_checkpoint(tmp_path / "heads.pt")
    assert info.source_names == ["vocalset", "aihub_465_multi_singer"] and info.sources[0]["license"] == "CC-BY-4.0"
    from gyeol.attributes.heads import FrameHeads

    m = FrameHeads(heads.model.norm.normalized_shape[0], TASKS, hidden=32)
    m.load_state_dict(state)
    m.eval()
    x = torch.randn(1, 10, heads.model.norm.normalized_shape[0])
    assert torch.allclose(m(x)[0]["register"], heads.model.eval()(x)[0]["register"])


def test_all_tasks_train_including_laryngeal_and_phones():
    rng = np.random.default_rng(0)
    tasks = default_tasks(n_phones=7)
    exs = []
    for s in range(4):
        y = np.repeat(rng.integers(0, 3, 5), 8)  # labels persist over segments, like real consonant/vowel spans
        f = np.eye(3)[y] + 0.1 * rng.standard_normal((40, 3))
        f = np.c_[f, rng.standard_normal((40, 5))]
        exs.append(FrameExample(f, {"laryngeal": y, "phones": rng.integers(-1, 7, 40), "register": y,
                                    "phonation": np.full((40, 5), np.nan)}, f"s{s}"))
    h = train_heads(exs[:3], exs[3:], tasks, HeadTrainConfig(hidden=16, epochs=20))
    probs, _, _ = h.predict(exs[3].features)
    assert (probs["laryngeal"].argmax(1) == exs[3].targets["laryngeal"]).mean() > 0.8
    assert probs["phones"].shape == (40, 7)


# --- encoders ------------------------------------------------------------------

class _TinySSL(torch.nn.Module):
    """Stand-in for a torchaudio SSL model: 20 ms frames, 3 'layers'."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv1d(1, 8, 320, stride=320)

    def extract_features(self, wav, num_layers=None):
        h = self.conv(wav[:, None]).transpose(1, 2)
        return [h, torch.tanh(h), h * 2][: num_layers or 3], None


def test_torch_ssl_encoder_projects_to_grid():
    enc = TorchSSLEncoder(_TinySSL(), dim=8, layers=(0, 1), frame_rate=50.0)
    x = np.random.default_rng(0).standard_normal(SR)
    g = FrameGrid.for_samples(len(x), SR)
    f = enc.encode(x, SR, g).unwrap()
    assert f.shape == (g.n_frames, 8) and np.isfinite(f[2:-2]).all()
    assert enc.encode(np.zeros(100), SR, g).status is Status.FAILED
    assert not any(p.requires_grad for p in enc.module.parameters())


# --- probes --------------------------------------------------------------------

def test_probes_and_leakage():
    rng = np.random.default_rng(0)
    n, groups = 600, np.repeat(np.arange(12), 50)
    y = rng.integers(0, 3, n)
    informative = np.c_[np.eye(3)[y] * 2 + rng.standard_normal((n, 3)), rng.standard_normal((n, 5))]
    noise = rng.standard_normal((n, 8))
    good = probe_classify(informative, y, groups, "register")
    assert good.score > 0.8 and good.p_value < 1e-6
    leak = leakage(noise, y, groups, "register")
    assert leak.near_chance and abs(leak.probe.score - leak.probe.chance) < 0.1
    assert not leakage(informative, y, groups).near_chance
    t = rng.standard_normal(n)
    r = probe_regress(np.c_[t + 0.1 * rng.standard_normal(n), noise], t, groups, "time_offset")
    assert r.score > 0.9
    table = probe_battery({"c": informative, "r": noise}, {"register": (y, "classify"), "offset": (t, "regress")}, groups)
    assert table[("c", "register")].score > table[("r", "register")].score


def test_mahalanobis_detector_basics():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((500, 4))
    det = MahalanobisOOD().fit(X, np.zeros(500, int)).calibrate(X, 0.95)
    assert det.is_unknown(X).mean() == pytest.approx(0.05, abs=0.02)
    assert det.is_unknown(X + 10).all()
