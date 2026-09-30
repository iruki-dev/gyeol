import numpy as np
import pytest
import torch
import torch.nn.functional as F

from gyeol.core import FrameGrid, Provenance, Recording, Status
from gyeol.decoder import AutoencoderConfig, BigVGANAdapter, GyeolAutoencoder, HarmonicSource, NSFVocoder
from gyeol.encoders.latent import LeakageHeads, ResidualEncoder, SingerEncoder, grad_reverse, singer_vector, supervised_contrastive
from gyeol.encoders.mel import LogMel
from gyeol.eval import BenchmarkItem, probe_classify, reencoding_consistency, vocoder_benchmark
from gyeol.train.autoencoder import batch_from_representations, discriminator_step, generator_step
from gyeol.train.losses import MultiPeriodDiscriminator, MultiResolutionSTFTLoss, MultiScaleDiscriminator, frame_weights
from gyeol.synth import sung_vowel

from .helpers import SR, make_melody, rep_of


def test_logmel_matches_frame_grid():
    x = torch.randn(2, 44100)
    mel = LogMel()(x)
    assert mel.shape == (2, FrameGrid.for_samples(44100).n_frames, 80)


def test_harmonic_source_follows_f0():
    torch.manual_seed(0)
    src = HarmonicSource()
    for f in (110.0, 220.0, 440.0):
        e = src(torch.full((1, 50), f), 49 * 512, seed=0)[0, 0].detach().numpy()
        e = e - e.mean()
        sp = np.abs(np.fft.rfft(e * np.hanning(len(e))))
        fr = np.fft.rfftfreq(len(e), 1 / SR)
        sel = (fr > 50) & (fr < 2000)
        peaks = fr[sel][np.argsort(sp[sel])[-3:]]
        assert np.all(np.abs(peaks / f - np.round(peaks / f)) < 0.02)  # every strong peak is a harmonic
    unvoiced = src(torch.zeros(1, 20), 19 * 512, seed=0)
    assert torch.isfinite(unvoiced).all()


def test_vocoder_shapes_and_hop_check():
    voc = NSFVocoder(n_mels=16, channels=16)
    y = voc(torch.randn(2, 11, 16), -torch.rand(2, 11, 5) * 20, torch.full((2, 11), 200.0), torch.rand(2, 11), seed=0)
    assert y.shape == (2, 10 * 512)
    with pytest.raises(ValueError, match="multiply"):
        NSFVocoder(upsample_rates=(8, 8, 4))


def test_acoustic_model_condition_dropout_and_missing_attributes():
    cfg = AutoencoderConfig.tiny()
    m = GyeolAutoencoder(cfg).acoustic
    B, T = 2, 12
    args = (torch.randn(B, T, cfg.c_dim), torch.ones(B, T, cfg.c_dim, dtype=torch.bool), torch.randn(B, T, cfg.r_dim),
            torch.full((B, T), 200.0), torch.zeros(B, T), torch.randn(B, cfg.singer_dim), torch.randn(B, cfg.env_dim))
    m.eval()
    full, _ = m(*args)
    # an attribute marked missing is replaced by the learned null embedding: its value no longer matters
    masked = list(args)
    masked[1] = args[1].clone()
    masked[1][..., 0] = False
    a, _ = m(*masked)
    masked[0] = args[0].clone()
    masked[0][..., 0] = 999.0
    b, _ = m(*masked)
    assert torch.allclose(a, b) and not torch.allclose(a, full)


@pytest.fixture(scope="module")
def batch():
    m1, m2 = make_melody(dur=0.3, gap=0.1), make_melody(dur=0.3, gap=0.1, transpose=-500, seed=2)
    r1, r2 = rep_of(m1.audio).unwrap(), rep_of(m2.audio[: len(m1.audio)]).unwrap()
    env = [{"noise": 0, "room": 0, "eq": 0, "codec": 0, "compression": 0, "separation": 0},
           {"noise": 1, "room": 2, "eq": 1, "codec": 0, "compression": 0, "separation": 0}]
    return batch_from_representations([r1, r2], [m1.audio, m2.audio], singer_ids=[0, 1], env_classes=env), (r1, m1)


def test_training_step_reduces_loss(batch):
    b, _ = batch
    torch.manual_seed(0)
    model = GyeolAutoencoder(AutoencoderConfig.tiny(n_singers=2))
    opt = torch.optim.Adam(model.parameters(), 2e-3)
    discs = [MultiPeriodDiscriminator(ch=4), MultiScaleDiscriminator(2, ch=4)]
    dopt = torch.optim.Adam([p for d in discs for p in d.parameters()], 2e-4)
    first = None
    for step in range(12):
        vocode = step % 4 == 0
        L = generator_step(model, b, vocode=vocode, discriminators=discs if vocode else None)
        opt.zero_grad()
        L["total"].backward()
        opt.step()
        if vocode:
            assert {"stft", "reencode", "adv", "fm"} <= set(L)
            with torch.no_grad():
                enc = model.encode(b.wav, b.c, b.c_mask)
                T = enc["mel"].shape[1]
                y = model.decode(enc["singer"], enc["env"], b.c[:, :T], b.c_mask[:, :T], enc["r"], b.f0_hz[:, :T], b.aperiodic[:, :T])["wav"]
            d = discriminator_step(discs, b.wav[:, : y.shape[1]], y)
            dopt.zero_grad()
            d.backward()
            dopt.step()
        first = first if first is not None else float(L["mel"].detach())
    assert float(L["mel"].detach()) < 0.8 * first
    assert {"env", "leak", "kl"} <= set(L)
    assert "singer" not in L  # SupCon needs a same-singer pair; this batch has two different singers


def test_gradient_reversal():
    x = torch.ones(3, requires_grad=True)
    grad_reverse(x, 0.5).sum().backward()
    assert torch.allclose(x.grad, torch.full((3,), -0.5))


def _train_residual(adversarial: bool, steps: int = 400):
    """The target depends on an attribute that the decoder gets explicitly only half
    of the time (condition dropout), so without an adversary r learns to carry it as
    a backup channel; with the GRL adversary it should not."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    N, T, D = 64, 20, 12
    attr = rng.integers(0, 3, (N, T))
    content = rng.standard_normal((N, T, 4))
    x = np.concatenate([np.eye(3)[attr] * 2.0, content, rng.standard_normal((N, T, D - 7)) * 0.1], axis=-1)
    target = content + np.eye(3)[attr] @ rng.standard_normal((3, 4))
    X = torch.tensor(x, dtype=torch.float32)
    A = torch.tensor(np.eye(3)[attr], dtype=torch.float32)
    Y = torch.tensor(target, dtype=torch.float32)
    enc = ResidualEncoder(D, 0, hidden=32, dim=4)
    dec = torch.nn.Sequential(torch.nn.Linear(4 + 3, 32), torch.nn.GELU(), torch.nn.Linear(32, 4))
    adv = LeakageHeads(4, {}, {"attr": 3}, hidden=32, grl=1.0)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()) + list(adv.parameters()), 3e-3)
    for _ in range(steps):
        r, kl = enc(X)
        keep = (torch.rand(N, 1, 1) > 0.5).float()
        loss = F.mse_loss(dec(torch.cat([r, A * keep], -1)), Y) + 1e-3 * kl.mean()
        if adversarial:
            _, cls = adv(r)
            loss = loss + F.cross_entropy(cls["attr"].reshape(-1, 3), torch.tensor(attr.reshape(-1)))
        opt.zero_grad()
        loss.backward()
        opt.step()
    enc.eval()
    with torch.no_grad():
        r, _ = enc(X)
    groups = np.repeat(np.arange(N), T)
    return probe_classify(r.reshape(-1, 4).numpy(), attr.reshape(-1), groups, "attr").score


def test_adversarial_heads_reduce_attribute_leakage():
    plain = _train_residual(False)
    adv = _train_residual(True)
    assert adv < plain - 0.1
    assert adv < 0.6  # chance = 1/3


def test_singer_encoder_contrastive_and_provenance():
    torch.manual_seed(0)
    enc = SingerEncoder(16, 32, 8)
    opt = torch.optim.Adam(enc.parameters(), 3e-3)
    rng = np.random.default_rng(0)
    base = rng.standard_normal((4, 16)) * 2  # four "singers": different spectral colour
    def sample():
        ids = np.repeat(np.arange(4), 4)
        mel = base[ids][:, None, :] + rng.standard_normal((16, 30, 16))
        return torch.tensor(mel, dtype=torch.float32), torch.tensor(ids)
    for _ in range(60):
        mel, ids = sample()
        loss = supervised_contrastive(enc(mel), ids)
        opt.zero_grad()
        loss.backward()
        opt.step()
    mel, ids = sample()
    with torch.no_grad():
        e = enc(mel)
    sim = (e @ e.T).numpy()
    same = sim[ids[:, None] == ids[None, :]].mean()
    diff = sim[ids[:, None] != ids[None, :]].mean()
    assert same > diff + 0.3
    rec = Recording(np.zeros(100), SR, Provenance.REFERENCE)
    sv = singer_vector(enc, mel[0].numpy(), rec)
    assert sv.provenance is Provenance.REFERENCE and sv.source_recording_id == rec.recording_id


def test_frame_weights_emphasise_consonants_and_quiet_frames():
    loud = np.array([0.0, -10, -30, -50, -99])
    voiced = np.array([True, True, True, False, False])
    w = frame_weights(loud, voiced)
    assert w[0] == 1 and w[2] > 1 and w[3] > 1 and w[4] == 1


def test_mrstft_loss_zero_for_identical_signals():
    x = torch.randn(1, 8192)
    assert float(MultiResolutionSTFTLoss()(x, x)) < 1e-5
    assert float(MultiResolutionSTFTLoss()(x * 0.5, x)) > 0.1


def test_vocoder_benchmark_and_reencoding_consistency():
    items = [BenchmarkItem(sung_vowel(f0=220, duration=0.8, sr=SR, aspiration=a).audio, SR, {"technique": t})
             for a, t in ((0.0, "modal"), (0.5, "breathy"))] + \
            [BenchmarkItem(sung_vowel(f0=220, duration=0.8, sr=SR, subharmonic=0.3).audio, SR, {"technique": "rough"})]
    from .helpers import dsp_trackers
    from gyeol.attributes.extract import analyze

    an = lambda rec: analyze(rec, trackers=dsp_trackers())  # noqa: E731
    ident = vocoder_benchmark(lambda x, sr: x, items, analyzer=an)
    assert set(ident) == {"modal", "breathy", "rough"}
    assert all(r.lsd_voiced_db < 1e-3 and r.f0_rmse_cents < 1e-6 and r.voicing_error == 0 for r in ident.values())
    detuned = vocoder_benchmark(lambda x, sr: np.interp(np.arange(len(x)) * 1.03, np.arange(len(x)), x), items, analyzer=an)
    assert all(r.f0_rmse_cents > 30 for r in detuned.values())
    a = an(Recording(items[0].audio, SR, Provenance.SYNTHETIC)).unwrap()
    assert all(v == 0 for v in reencoding_consistency(a, a).values() if np.isfinite(v))


def test_bigvgan_adapter_is_optional():
    ad = BigVGANAdapter("/nonexistent")
    r = ad.vocode(np.zeros((10, 100)))
    assert r.status is Status.UNAVAILABLE and not ad.f0_control
