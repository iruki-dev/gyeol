import hashlib

import numpy as np
import pytest

from gyeol.core import (
    ASSETS,
    Asset,
    AttributeCurve,
    AttributeCurves,
    FrameGrid,
    GridMismatchError,
    Result,
    ResultError,
    Status,
    add_asset,
    asset,
    describe,
)


# --- FrameGrid -------------------------------------------------------------

def test_frame_grid_defaults_and_geometry():
    g = FrameGrid.for_samples(44100)
    assert (g.sr, g.hop) == (44100, 512)
    assert g.n_frames == 1 + 44100 // 512
    assert g.rate == pytest.approx(86.13, abs=0.01)
    assert g.frame_of(1.0) == round(44100 / 512)
    np.testing.assert_allclose(g.times()[:3], [0, 512 / 44100, 1024 / 44100])


def test_frame_grid_mismatch_is_loud():
    a = FrameGrid(44100, 512, 10)
    with pytest.raises(GridMismatchError):
        a.require_same(FrameGrid(16000, 160, 10))
    with pytest.raises(GridMismatchError):
        a.require_same(FrameGrid(44100, 512, 11))
    a.require_same(FrameGrid(44100, 512, 11), frames=False)


def test_curves_share_one_grid():
    g = FrameGrid(44100, 512, 5)
    c = AttributeCurve("f0", np.array([1, 2, np.nan, 4, 5.0]), np.ones(5), g)
    assert c.confidence[2] == 0.0  # unknown values carry zero confidence
    curves = AttributeCurves(g, {"f0": c})
    with pytest.raises(GridMismatchError):
        curves.add(AttributeCurve("x", np.zeros(6), np.ones(6), g.with_frames(6)))
    with pytest.raises(GridMismatchError):
        AttributeCurve("bad", np.zeros(4), np.ones(4), g)


def test_result_status():
    r = Result.failure("clipped")
    assert not r.ok and r.status is Status.FAILED
    with pytest.raises(ResultError, match="clipped"):
        r.unwrap()
    u = Result.unreliable(3.0, "low snr")
    assert u.usable and not u.ok and u.unwrap() == 3.0


# --- asset list ------------------------------------------------------------

def test_asset_list_documents_every_third_party_model_and_dataset():
    for name in ["vocalset", "gtsinger", "csd", "opencpop", "popbutfy", "vocadito", "mir1k", "rmvpe", "hubert_fairseq",
                 "bs_roformer_viperx_ep317", "openvpi_nsf_hifigan", "so_vits_svc", "pesto"]:
        a = asset(name)
        assert a.license and a.source and a.kind in ("dataset", "weights", "code")
    assert asset("gtsinger").license == "CC-BY-NC-SA-4.0"
    assert all(len(a.sha256) == 64 for a in ASSETS.values() if a.sha256)
    with pytest.raises(KeyError, match="not in gyeol's asset list"):
        asset("some_scraped_dataset")


def test_add_asset_and_describe(monkeypatch):
    monkeypatch.setattr("gyeol.core.assets.ASSETS", dict(ASSETS))
    from gyeol.core import assets

    add_asset(Asset("my_ckpt", "weights", "proprietary", "in-house"))
    with pytest.raises(ValueError, match="already listed"):
        add_asset(Asset("my_ckpt", "weights", "x", "y"))
    assert assets.describe("my_ckpt")["license"] == "proprietary"
    assert describe("never_heard_of")["kind"] == "unlisted"


def test_checkpoint_records_its_sources(tmp_path):
    import torch

    from gyeol.train import load_checkpoint, save_checkpoint

    sd = {"w": torch.zeros(2)}
    parent = save_checkpoint(tmp_path / "a.pt", sd, name="a", sources=["vocalset"], config={"lr": 1})
    save_checkpoint(tmp_path / "b.pt", sd, name="b", sources=["vocalset", "gtsinger"], config={}, parents=[parent])
    state, info = load_checkpoint(tmp_path / "b.pt")
    assert torch.equal(state["w"], sd["w"])
    assert info.source_names == ["vocalset", "gtsinger"]
    assert {s["name"]: s["license"] for s in info.sources}["gtsinger"] == "CC-BY-NC-SA-4.0"
    assert info.parents[0]["name"] == "a" and info.parents[0]["sources"][0]["name"] == "vocalset"
    # a plain state dict loads too, with empty provenance
    torch.save(sd, tmp_path / "bare.pt")
    state, info = load_checkpoint(tmp_path / "bare.pt")
    assert torch.equal(state["w"], sd["w"]) and info.sources == []


def _fake_download(payload: bytes):
    def retrieve(url, dest):
        open(dest, "wb").write(payload)

    return retrieve


def test_fetch_downloads_and_verifies_the_checksum(tmp_path, monkeypatch, capsys):
    import gyeol.cli as cli
    from gyeol.core import assets

    payload = b"weights"
    monkeypatch.setattr(assets, "ASSETS", dict(ASSETS))
    add_asset(Asset("tiny", "weights", "MIT", "test", url="https://example.invalid/t.pt", sha256=hashlib.sha256(payload).hexdigest()))
    monkeypatch.setattr(cli.urllib.request, "urlretrieve", _fake_download(payload))
    dest = tmp_path / "t.pt"
    assert cli.main(["fetch", "tiny", "--dest", str(dest)]) == 0
    assert dest.read_bytes() == payload
    out = capsys.readouterr().out
    assert "MIT" in out and "sha256 verified" in out
    assert cli.main(["fetch", "tiny", "--dest", str(dest)]) == 0 and "already present" in capsys.readouterr().out
    # a corrupted download is deleted and reported
    monkeypatch.setattr(cli.urllib.request, "urlretrieve", _fake_download(b"tampered"))
    bad = tmp_path / "bad.pt"
    assert cli.main(["fetch", "tiny", "--dest", str(bad)]) == 4
    assert not bad.exists() and not (tmp_path / "bad.pt.part").exists()
    # assets without a URL are obtained manually; unknown names are reported
    assert cli.main(["fetch", "gtsinger"]) == 3
    assert cli.main(["fetch", "nope"]) == 2


def test_licenses_command_lists_assets_and_the_responsibility_line(capsys):
    from gyeol.cli import main

    assert main(["licenses"]) == 0
    out = capsys.readouterr().out
    assert "gtsinger" in out and "CC-BY-NC-SA-4.0" in out and "responsible for complying" in out


# --- store -----------------------------------------------------------------

def test_consent_store_features_and_deletion(tmp_path):
    from gyeol_service.store import ConsentError, ConsentStore, FeatureStore, Purpose, delete_user

    cs = ConsentStore(tmp_path)
    fs = FeatureStore(tmp_path, cs)
    with pytest.raises(ConsentError):
        fs.put("u1", "singer", {"v": np.ones(4)})
    tok = cs.grant("u1", {Purpose.ANALYSIS, Purpose.STORAGE})
    assert cs.verify(tok)
    fs.put("u1", "singer", {"v": np.ones(4)}, {"model": "tiny"})
    arrays, meta = fs.get("u1", "singer")
    assert meta["model"] == "tiny" and np.all(arrays["v"] == 1)
    cs.revoke("u1", Purpose.STORAGE)
    assert not cs.verify(tok)
    with pytest.raises(ConsentError):
        fs.put("u1", "other", {"v": np.ones(1)})
    out = delete_user(tmp_path, "u1")
    assert out["files"] >= 1
    assert fs.keys("u1") == [] and not cs.allows("u1", Purpose.ANALYSIS)
    assert "u1" in (tmp_path / "deletion_log.jsonl").read_text()
    with pytest.raises(ValueError):
        fs.put("../etc", "x", {})


def test_raw_audio_retention(tmp_path):
    from gyeol_service.store import ConsentStore, Purpose, RawAudioRetention, RawAudioStore, RetentionPolicy

    cs = ConsentStore(tmp_path)
    cs.grant("u1", {Purpose.STORAGE})
    default = RawAudioStore(tmp_path, cs)
    assert default.policy.raw_audio is RawAudioRetention.DELETE_AFTER_EXTRACTION
    assert default.put("u1", "rec1", np.zeros(10), 44100) is None  # never written
    keep = RawAudioStore(tmp_path, cs, RetentionPolicy(RawAudioRetention.KEEP_DAYS, keep_days=1))
    assert keep.put("u1", "rec2", np.zeros(10), 44100).exists()
    assert keep.sweep(now=0.0) == 0
    import time

    assert keep.sweep(now=time.time() + 2 * 86400) == 1


def test_readme_lists_every_third_party_asset_with_its_license():
    from pathlib import Path

    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    section = readme[readme.index("## Licenses"):]
    for a in ASSETS.values():
        if a.name in ("gyeol_synthetic", "own_recordings"):
            continue
        assert f"`{a.name}`" in section and a.license in section, a.name
    assert "responsible for complying with these licenses and with applicable law" in section
