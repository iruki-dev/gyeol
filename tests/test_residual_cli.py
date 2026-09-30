import json

import numpy as np
import pytest

import gyeol
from gyeol.cli import main
from gyeol.io import save_audio
from gyeol.residual import core_matrix
from gyeol.synth import phrase


def test_core_matrix_masks_invalid_values():
    rep = gyeol.analyze(phrase([(220, 0.5)], gap_s=0.2).audio, 16000)
    m = core_matrix(rep.tracks)
    assert m.shape[1] == 2 * len(gyeol.residual.CORE_INPUTS)
    assert np.isfinite(m).all()
    # value columns are zero wherever the validity column is zero
    vals, flags = m[:, 0::2], m[:, 1::2]
    assert np.all(vals[flags == 0] == 0)


def test_residual_model_training_step():
    torch = pytest.importorskip("torch")
    from gyeol.residual.torch_model import ResidualConfig, ResidualModel, TorchResidualEncoder, grad_reverse, residual_loss

    torch.manual_seed(0)
    cfg = ResidualConfig(hidden=32, z_dim=8)
    model = ResidualModel(cfg)
    mel = torch.randn(4, cfg.n_mels, 100)
    core = torch.randn(4, cfg.core_dim, 100)
    out = model(mel, core)
    assert out["z"].shape == (4, 8, 25)  # 100 Hz → 25 Hz
    assert out["recon"].shape == mel.shape
    labels = {k: torch.randint(0, n, (4,)) for k, n in cfg.nuisance_classes.items()}
    loss = residual_loss(model, out, mel, labels, torch.tensor([0, 0, 1, 1]))
    loss["total"].backward()
    assert all(p.grad is not None for p in model.encoder.parameters())

    # gradient reversal flips the sign of the gradient reaching the encoder
    x = torch.ones(3, requires_grad=True)
    grad_reverse(x, 0.5).sum().backward()
    assert torch.allclose(x.grad, torch.full((3,), -0.5))

    rep = gyeol.analyze(phrase([(220, 0.5), (262, 0.5)], gap_s=0.2).audio, 16000)
    enc = TorchResidualEncoder(model)
    z = enc.encode(np.zeros(16000), 16000, rep.tracks)
    assert z.values.shape[1] == 8 and z.rate == pytest.approx(25.0)
    engine = gyeol.Engine(residual_encoder=enc)
    rep2 = engine.analyze(phrase([(220, 0.5)], gap_s=0.2).audio, 16000)
    assert rep2.residual is not None and 0 < rep2.residual.valid.mean() < 1


def test_cli_analyze_and_spec(tmp_path, capsys):
    wav = tmp_path / "t.wav"
    save_audio(wav, phrase([(262, 0.5), (330, 0.5)], gap_s=0.1).audio, 16000)
    out = tmp_path / "t.npz"
    assert main(["analyze", str(wav), "-o", str(out), "--lyrics", "도레", "--summary"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["n_notes"] == 2 and out.exists()
    assert main(["spec", "--json"]) == 0
    spec = json.loads(capsys.readouterr().out)
    assert "absolute_spl" in spec["omissions"] and "h1h2c" in spec["frame"]
