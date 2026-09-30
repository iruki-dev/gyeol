import io
import warnings

import numpy as np
import pytest

from gyeol.core import (
    AttributeCurve,
    AttributeCurves,
    ConsentedVoice,
    ConsentError,
    ConsentToken,
    FrameGrid,
    GridMismatchError,
    LicenseError,
    LicenseTag,
    Profile,
    Provenance,
    Purpose,
    Recording,
    Representation,
    Result,
    ResultError,
    SingerVector,
    Status,
    lookup,
    require_allowed,
    require_consented_voice,
)
from gyeol.core.license import AssetKind, LicensedAsset


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


# --- licensing -------------------------------------------------------------

@pytest.mark.parametrize("name", ["gtsinger", "csd", "opencpop", "popbutfy", "openvpi_nsf_hifigan", "so_vits_svc", "pesto"])
def test_commercial_profile_refuses_non_commercial(name):
    with pytest.raises(LicenseError):
        require_allowed(lookup(name), Profile.COMMERCIAL)


def test_commercial_profile_allows_commercial_and_prints_aihub_conditions(capsys):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        require_allowed(lookup("vocalset"), Profile.COMMERCIAL)
        d = require_allowed(lookup("aihub_465_multi_singer"), Profile.COMMERCIAL)
    err = capsys.readouterr().err
    assert d.allowed
    assert "never be redistributed" in err and "domestically" in err and "individually" in err


def test_unknown_license_refused_commercially_and_unregistered_assets_rejected():
    with pytest.raises(LicenseError):
        require_allowed(LicensedAsset("x", AssetKind.DATASET, LicenseTag.UNKNOWN, "?"), Profile.COMMERCIAL)
    with pytest.raises(LicenseError, match="not in the license registry"):
        lookup("some_scraped_dataset")


def test_research_profile_allows_with_notices():
    with pytest.warns(UserWarning, match="copyleft"):
        assert require_allowed(lookup("pesto"), Profile.RESEARCH).allowed
    assert not lookup("pesto").vendor_allowed


def test_data_loader_license_gate(tmp_path):
    from gyeol.data import Manifest, ManifestItem, open_manifest

    m = Manifest("gtsinger", str(tmp_path), [ManifestItem("a.wav", "s1")])
    with pytest.raises(LicenseError):
        open_manifest(m, "commercial")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = open_manifest(m, "research")
        assert len(ds) == 1
        path = tmp_path / "m.json"
        Manifest("vocalset", str(tmp_path), [ManifestItem("b.wav", "f1", {"technique": "belt"})]).write(path)
        ds2 = open_manifest(path, Profile.COMMERCIAL)
    assert ds2.resolve(next(iter(ds2))) == tmp_path / "b.wav"


def test_checkpoint_license_gate(tmp_path):
    import torch

    from gyeol.train import load_checkpoint, most_restrictive, save_checkpoint

    sd = {"w": torch.zeros(2)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        save_checkpoint(tmp_path / "ok.pt", sd, name="ok", sources=["vocalset"], config={"lr": 1}, profile=Profile.COMMERCIAL)
        state, info = load_checkpoint(tmp_path / "ok.pt", Profile.COMMERCIAL)
        assert info.license is LicenseTag.COMMERCIAL_OK and torch.equal(state["w"], sd["w"])
        # research checkpoint trained on NC data inherits the NC tag ...
        save_checkpoint(tmp_path / "nc.pt", sd, name="nc", sources=["vocalset", "gtsinger"], config={}, profile=Profile.RESEARCH)
        # ... and a commercial run refuses to load it
        with pytest.raises(LicenseError):
            load_checkpoint(tmp_path / "nc.pt", Profile.COMMERCIAL)
        load_checkpoint(tmp_path / "nc.pt", Profile.RESEARCH)
        # commercial training may not consume NC sources at all
        with pytest.raises(LicenseError):
            save_checkpoint(tmp_path / "x.pt", sd, name="x", sources=["csd"], config={}, profile=Profile.COMMERCIAL)
        # a bare torch checkpoint without gyeol metadata is refused
        torch.save(sd, tmp_path / "bare.pt")
        with pytest.raises(LicenseError, match="no embedded license"):
            load_checkpoint(tmp_path / "bare.pt", Profile.RESEARCH)
    assert most_restrictive([LicenseTag.COMMERCIAL_OK, LicenseTag.COMMERCIAL_OK_CONDITIONAL]) is LicenseTag.COMMERCIAL_OK_CONDITIONAL
    assert most_restrictive([LicenseTag.COMMERCIAL_OK, LicenseTag.NONCOMMERCIAL]) is LicenseTag.NONCOMMERCIAL


def test_fetch_shows_license_and_requires_confirmation(capsys, monkeypatch):
    import gyeol.cli as cli

    called = []
    monkeypatch.setattr(cli.urllib.request, "urlretrieve", lambda *a: called.append(a))
    monkeypatch.setitem(cli.REGISTRY, "tiny", LicensedAsset("tiny", AssetKind.CHECKPOINT, LicenseTag.COMMERCIAL_OK, "MIT", url="https://example.invalid/t.pt"))
    monkeypatch.setattr(cli, "lookup", lambda n: cli.REGISTRY[n])
    args = cli.argparse.Namespace(name="tiny", profile="commercial", yes=False)
    assert cli._fetch(args, stdin=io.StringIO("no\n")) == 1
    assert called == []
    out = capsys.readouterr().out
    assert "MIT" in out and "commercial_ok" in out
    assert cli.main(["fetch", "gtsinger"]) == 2  # refused under commercial profile, nothing downloaded
    assert called == []


# --- consent ---------------------------------------------------------------

def _user_voice(user="u1"):
    rec = Recording(np.zeros(100), 44100, Provenance.USER, owner_id=user)
    sv = SingerVector(np.ones(8), Provenance.USER, rec.recording_id, owner_id=user)
    tok = ConsentToken(user, frozenset({Purpose.ANALYSIS, Purpose.VOICE_SYNTHESIS}))
    return rec, sv, tok


def test_consented_voice_happy_path():
    rec, sv, tok = _user_voice()
    v = ConsentedVoice.create(sv, rec, tok)
    assert require_consented_voice(v) is v
    with pytest.raises(AttributeError):
        v.user_id = "other"


def test_reference_vectors_can_never_be_rendered():
    ref = Recording(np.zeros(100), 44100, Provenance.REFERENCE)
    sv = SingerVector(np.ones(8), Provenance.REFERENCE, ref.recording_id)
    tok = ConsentToken("u1", frozenset(Purpose))
    with pytest.raises(ConsentError):
        ConsentedVoice.create(sv, ref, tok)
    with pytest.raises(TypeError):
        ConsentedVoice(sv, "u1", "t", ref.recording_id)
    with pytest.raises(ConsentError):
        require_consented_voice(sv)


@pytest.mark.parametrize("case", ["other_user", "no_purpose", "revoked", "wrong_recording"])
def test_consent_violations(case):
    rec, sv, tok = _user_voice()
    if case == "other_user":
        tok = ConsentToken("u2", frozenset({Purpose.VOICE_SYNTHESIS}))
    elif case == "no_purpose":
        tok = ConsentToken("u1", frozenset({Purpose.ANALYSIS}))
    elif case == "revoked":
        tok = ConsentToken("u1", frozenset({Purpose.VOICE_SYNTHESIS}), revoked=True)
    elif case == "wrong_recording":
        rec = Recording(np.zeros(100), 44100, Provenance.USER, owner_id="u1")
    with pytest.raises(ConsentError):
        ConsentedVoice.create(sv, rec, tok)


def test_representation_rejects_provenance_mismatch():
    g = FrameGrid(44100, 512, 3)
    sv = SingerVector(np.ones(4), Provenance.REFERENCE, "r")
    with pytest.raises(ValueError):
        Representation(g, AttributeCurves(g), "r", Provenance.USER, singer=sv)


# --- store -----------------------------------------------------------------

def test_consent_store_features_and_deletion(tmp_path):
    from gyeol.store import ConsentStore, FeatureStore, delete_user

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
    from gyeol.store import ConsentStore, RawAudioRetention, RawAudioStore, RetentionPolicy

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
