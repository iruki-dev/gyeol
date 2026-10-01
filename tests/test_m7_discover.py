"""M7: discovery — TopK SAE, feature matching, conditional directions, transfer tests,
promotion rule and registry, residual-energy monitoring."""

import numpy as np
import pytest

from gyeol.discover import (
    DirectionConfig,
    PairedFrames,
    PromotionError,
    PromotionRegistry,
    ResidualMonitor,
    SAEConfig,
    auroc,
    concat,
    evaluate_promotion,
    feature_stats,
    fit_conditional_directions,
    match_features,
    pair_frames,
    representation_residual_energy,
    residual_energy,
    train_sae,
)

# ================================================================ SAE


def _dictionary_data(seed=0, d=16, n_atoms=24, N=4000, k=3):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n_atoms, d))
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    Z = np.zeros((N, n_atoms))
    for i in range(N):
        Z[i, rng.choice(n_atoms, k, replace=False)] = rng.uniform(1, 3, k)
    return Z @ A + 0.02 * rng.standard_normal((N, d)), A, Z


@pytest.fixture(scope="module")
def sae_fit():
    X, A, Z = _dictionary_data()
    return X, A, Z, train_sae(X, SAEConfig(n_latents=48, k=3, steps=1500))


def test_topk_sae_recovers_a_planted_dictionary(sae_fit):
    X, A, _, fit = sae_fit
    assert fit.fvu(X) < 0.1 and fit.history[-1] < fit.history[0]
    D = fit.directions * fit.std  # back to input units
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    assert np.mean(np.abs(A @ D.T).max(1) > 0.9) >= 0.9  # ≥ 90 % of the atoms have a matching latent
    z = fit.codes(X[:50])
    assert np.all((z > 0).sum(1) <= 3) and np.all(z >= 0)  # TopK sparsity
    st = feature_stats(fit, X)
    assert st.energy_share.sum() == pytest.approx(1.0) and np.all(st.frequency <= 1)


def test_sae_is_deterministic_and_validates_input(sae_fit):
    X = sae_fit[0][:600]
    cfg = SAEConfig(n_latents=16, k=2, steps=50)
    assert np.allclose(train_sae(X, cfg).codes(X[:20]), train_sae(X, cfg).codes(X[:20]))
    bad = X.copy()
    bad[3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        train_sae(bad, cfg)
    with pytest.raises(ValueError):
        train_sae(X[:10], cfg)


# ================================================================ matching


def test_auroc_and_feature_matching():
    assert auroc(np.array([1, 2, 3, 4.0]), np.array([0, 0, 1, 1], bool)) == 1.0
    assert auroc(np.array([1, 1, 1, 1.0]), np.array([0, 1, 0, 1], bool)) == 0.5
    assert np.isnan(auroc(np.array([1, 2.0]), np.array([1, 1], bool)))
    rng = np.random.default_rng(0)
    N = 800
    label = rng.integers(0, 2, N)
    level = rng.normal(0, 1, N)
    codes = np.abs(rng.normal(0, 0.3, (N, 6)))
    codes[:, 2] += 2.0 * label  # latent 2 ↔ label
    codes[:, 4] += np.exp(level)  # latent 4 ↔ a DSP-like continuous feature
    codes[:, 5] = 0.0  # a dead latent
    dsp = level.copy()
    dsp[:100] = np.nan  # unknown frames are skipped
    cat = np.array(["a" if v else "b" for v in label], dtype=object)
    cat[:50] = None
    mt = match_features(codes, {"technique": (label, "classify"), "h1h2c": (dsp, "regress"), "cat": (cat, "classify")})
    assert mt.best_for_target("technique", 1)[0].feature == 2 and mt.best_for_target("technique", 1)[0].score > 0.9
    assert mt.best_for_target("cat", 1)[0].feature == 2
    assert mt.best_for_target("h1h2c", 1)[0].feature == 4 and mt.best_for_target("h1h2c", 1)[0].detail == "+"
    assert mt.best_for_feature(5).score == 0.0
    novel = mt.novel(0.3, active=np.array([True] * 5 + [False]))
    assert 2 not in novel and 4 not in novel and 5 not in novel and set(novel) == {0, 1, 3}


# ================================================================ conditional directions


def _paired_world(seed=0, n_groups=4, N=1200, D=10, bump=40.0):
    """on = off + v(f0 band) — but "on" takes are sung ``bump`` cents higher, and f0 moves the features."""
    rng = np.random.default_rng(seed)
    v_low, v_high, f_dir = (rng.standard_normal(D) for _ in range(3))
    v_low, v_high = v_low / np.linalg.norm(v_low), v_high / np.linalg.norm(v_high)
    f0 = rng.uniform(-1500, 0, N)
    loud = rng.uniform(-30, -10, N)
    base = rng.standard_normal((N, D)) * 0.3 + np.outer(f0 / 1200, f_dir) * 3 + np.outer(loud / 10, rng.standard_normal(D))
    f0_on = f0 + bump
    v = np.where((f0 < -700)[:, None], v_low, v_high)
    on = base + np.outer((f0_on - f0) / 1200, f_dir) * 3 + 0.5 * v + rng.standard_normal((N, D)) * 0.1
    group = rng.integers(0, n_groups, N)
    return PairedFrames(base, on, f0, f0_on, loud, loud, group), v_low, v_high


def test_directions_regress_out_f0_and_are_conditional():
    pf, v_low, v_high = _paired_world()
    cfg = DirectionConfig(f0_edges_cents=(-700.0,), min_pairs=50)
    cd = fit_conditional_directions(pf, cfg)
    low, high = cd.cells[(0, 0, None)], cd.cells[(1, 0, None)]
    assert low.unit @ v_low > 0.95 and high.unit @ v_high > 0.95
    assert abs(cd.similarity()[((0, 0, None), (1, 0, None))] - v_low @ v_high) < 0.1
    assert low.sign_consistency > 0.95 and low.group_t > 5 and low.n_groups == 4
    naive = fit_conditional_directions(pf, DirectionConfig(f0_edges_cents=(-700.0,), min_pairs=50, regress_out=False))
    assert naive.cells[(0, 0, None)].unit @ v_low < low.unit @ v_low - 0.05  # the pitch bump leaks in without regression
    s_on = cd.score(pf.on, pf.f0_on, pf.loud_on)
    s_off = cd.score(pf.off, pf.f0_off, pf.loud_off)
    assert auroc(np.r_[s_off, s_on], np.r_[np.zeros(len(s_off)), np.ones(len(s_on))].astype(bool)) > 0.8  # single unpaired frames
    # a cell without enough pairs falls back to the pooled direction
    assert cd.direction_for((7, 0, None)) is cd.pooled
    with pytest.raises(ValueError):
        fit_conditional_directions(PairedFrames(pf.off[:5], pf.on[:5], pf.f0_off[:5], pf.f0_on[:5], pf.loud_off[:5],
                                                pf.loud_on[:5], pf.group[:5]), cfg)
    with pytest.raises(ValueError):
        PairedFrames(pf.off, pf.on[:-1], pf.f0_off, pf.f0_on, pf.loud_off, pf.loud_on, pf.group)


def test_pair_frames_aligns_with_the_warp():
    T, D = 50, 3
    off = np.arange(T * D, dtype=float).reshape(T, D)
    on = off[np.clip(np.arange(T) - 5, 0, T - 1)] + 100.0  # "on" is 5 frames late
    f0 = np.full(T, -600.0)
    f0_nan = f0.copy()
    f0_nan[:3] = np.nan
    pf = pair_frames(off, on, np.arange(T) - 5.0, f0, f0_nan, np.zeros(T), np.zeros(T), "s1")
    assert np.allclose(pf.on[5:] - pf.off[5:], 100.0) and len(pf.off) == T - 3 and set(pf.group) == {"s1"}
    both = concat([pf, pf])
    assert len(both.off) == 2 * len(pf.off)


# ================================================================ transfer tests and promotion


def _world(confound=None, seed=0, n_singers=6, recs_per=10, frames=40, D=8, language=False):
    rng = np.random.default_rng(seed)
    sv = rng.standard_normal((n_singers, D))
    tech, pdir = (v / np.linalg.norm(v) for v in rng.standard_normal((2, D)))
    X, y, units, singers, pitch, lang = [], [], [], [], [], []
    for s in range(n_singers):
        for r in range(recs_per):
            base = rng.choice([-1500, -900, -300])
            p = base + rng.normal(0, 80, frames)
            if confound == "pitch":
                lab = float(rng.random() < {-1500: 0.1, -900: 0.5, -300: 0.9}[base])
            elif confound in ("singer", "language"):
                lab = float(s < n_singers // 2)
            else:
                lab = float(r % 2)
            x = sv[s] * (0.3 if confound == "pitch" else 1.0) + np.outer(p / 1200, pdir) * (6 if confound == "pitch" else 2)
            x = x + rng.standard_normal((frames, D)) * 0.5
            if confound is None:
                x = x + lab * tech * 2.0
            X.append(x)
            y += [lab] * frames
            units += [s * 100 + r] * frames
            singers += [s] * frames
            pitch.append(p)
            lang += ["ko" if s < n_singers // 2 else "en"] * frames
    X = np.vstack(X)
    out = dict(X=X, y=np.array(y), units=np.array(units), singers=np.array(singers), pitch=np.concatenate(pitch),
               lang=np.array(lang) if language else None)
    out["cand"] = {None: X @ tech, "pitch": X @ pdir, "singer": X, "language": X}[confound]
    return out


def test_genuine_feature_is_promoted_and_confounds_are_rejected(tmp_path):
    reg = PromotionRegistry(tmp_path / "promoted.json")
    w = _world()
    d = evaluate_promotion("technique", w["cand"], w["y"], w["units"], w["singers"], w["pitch"])
    assert d.passed and d.kind == "binary" and d.required == ("singer", "pitch_range")
    assert d.results["singer"].gain > 0.5 and d.results["pitch_range"].gain > 0.5
    attr = reg.register(d, w["cand"], w["y"], {"type": "sae_feature", "input": "phonation", "latents": [3]})
    s = attr.score(w["cand"])
    assert auroc(s, w["y"] > 0.5) > 0.9 and attr.task_spec().labels == ("technique",)
    assert PromotionRegistry(tmp_path / "promoted.json").names() == ["technique"]  # persisted
    with pytest.raises(PromotionError, match="already"):
        reg.register(d, w["cand"], w["y"], {})

    p = _world("pitch")
    dp = evaluate_promotion("pitchy", p["cand"], p["y"], p["units"], p["singers"], p["pitch"])
    assert not dp.passed and dp.results["singer"].passed and not dp.results["pitch_range"].passed
    with pytest.raises(PromotionError, match="pitch_range"):
        reg.register(dp, p["cand"], p["y"], {})

    s = _world("singer")
    ds = evaluate_promotion("singerish", s["cand"], s["y"], s["units"], s["singers"], s["pitch"])
    assert not ds.passed and not ds.results["singer"].passed
    assert ds.results["singer"].in_distribution_gain > 0.9  # looks perfect in-distribution
    with pytest.raises(PromotionError):
        reg.register(ds, s["cand"], s["y"], {})
    with pytest.raises(PromotionError):
        reg.register(None, s["cand"], s["y"], {})
    assert reg.names() == ["technique"]


def test_language_transfer_is_required_when_languages_are_known():
    w = _world("language", language=True, n_singers=8)
    d = evaluate_promotion("lang", w["cand"], w["y"], w["units"], w["singers"], w["pitch"], language=w["lang"])
    assert "language" in d.required and not d.results["language"].passed and not d.passed


def test_continuous_target_promotion():
    w = _world(seed=3)
    level = w["y"] * 2 + np.repeat(np.random.default_rng(0).uniform(0, 1, len(np.unique(w["units"]))), 40)
    cand = w["cand"] + (level - w["y"] * 2)  # the feature tracks the continuous level
    d = evaluate_promotion("level", cand, level, w["units"], w["singers"], w["pitch"])
    assert d.kind == "continuous" and d.passed
    from gyeol.discover.transfer import PromotedAttribute

    pa = PromotedAttribute("level", "continuous", {"unit": "dB"}, [1.0], 0.0, [0.0], [1.0], {}, "now")
    spec = pa.task_spec()  # regression head (M8)
    assert spec.kind == "regression" and spec.n == 1 and spec.unit == "dB" and spec.out_dim == 2


# ================================================================ residual monitor


def test_residual_monitor_flags_coverage_gaps_and_trends():
    rng = np.random.default_rng(0)
    r = rng.standard_normal((100, 4))
    mask = np.zeros(100, bool)
    mask[:50] = True
    assert residual_energy(r, mask) == pytest.approx(np.mean(np.sum(r[:50] ** 2, 1)))
    mon = ResidualMonitor()
    with pytest.raises(RuntimeError):
        mon.report()
    mon.set_baseline(list(rng.lognormal(0, 0.1, 30)))
    for _ in range(10):
        mon.add("ballad", float(rng.lognormal(0, 0.1)))
    for _ in range(10):
        mon.add("trot_kkeokki", float(rng.lognormal(0.6, 0.1)))  # a style the attributes do not cover
    rep = mon.report()
    by = {s.slice: s for s in rep.slices}
    assert not by["ballad"].flagged and by["trot_kkeokki"].flagged and by["trot_kkeokki"].ratio_to_baseline > 1.5
    assert "coverage_gap:trot_kkeokki" in rep.flags
    rising = ResidualMonitor()
    rising.set_baseline([1.0, 1.1, 0.9, 1.05])
    for t in range(20):
        rising.add("all", float(np.exp(0.03 * t + rng.normal(0, 0.02))), time=t)
    assert rising.report().trend.rising and "residual_rising" in rising.report().flags
    with pytest.raises(ValueError):
        mon.set_baseline([1.0])


def test_representation_residual_energy_needs_a_residual():
    from .helpers import make_melody, rep_of

    rep = rep_of(make_melody(dur=0.3, gap=0.1).audio).unwrap()
    with pytest.raises(ValueError, match="residual"):
        representation_residual_energy(rep)
    rep.residual = np.ones((rep.grid.n_frames, 3))
    assert representation_residual_energy(rep) == pytest.approx(3.0)


# ================================================================ end to end on synthetic singing


def test_breathiness_direction_is_discovered_and_transfers():
    """Paired on/off (breathy) takes from two 'discovery' singers give a conditional
    direction on log-mel spectral shape; it must transfer to two unseen singers and
    across pitch bands.  Raw pitch as a candidate must not."""
    import torch

    from gyeol.encoders.mel import LogMel

    from .helpers import make_melody, rep_of

    lm = LogMel(n_mels=24)
    recs = {}
    for s, oq in enumerate((0.45, 0.55, 0.62, 0.7)):
        for tr in (-500, 400):
            for lab, asp in ((0, 0.02), (1, 0.35)):
                m = make_melody(dur=0.3, gap=0.1, transpose=tr, seed=100 * s + 10 * (tr > 0) + lab, open_quotient=oq, aspiration=asp)
                rep = rep_of(m.audio).unwrap()
                L = lm(torch.tensor(m.audio, dtype=torch.float32)[None])[0].numpy()[: rep.grid.n_frames]
                f0 = np.where(rep.curves["f0_cents"].confidence > 0.5, rep.curves["f0_cents"].values, np.nan)
                recs[(s, tr, lab)] = dict(F=L - L.mean(1, keepdims=True), f0=f0, loud=rep.curves["loudness"].values, notes=rep.meta["notes"])
    parts = []
    for s in (0, 1):
        for tr in (-500, 400):
            a, b = recs[(s, tr, 0)], recs[(s, tr, 1)]
            n = min(len(a["F"]), len(b["F"]))
            parts.append(pair_frames(a["F"][:n], b["F"][:n], np.arange(n, dtype=float), a["f0"][:n], b["f0"][:n], a["loud"][:n],
                                     b["loud"][:n], s))
    cd = fit_conditional_directions(concat(parts), DirectionConfig(f0_edges_cents=(-700.0,)))
    assert cd.pooled.sign_consistency > 0.85 and len(cd.cells) == 2
    X, y, units, singers, pitch = [], [], [], [], []
    for (s, tr, lab), r in recs.items():
        if s < 2:
            continue  # evaluate only on singers the direction never saw
        v = np.isfinite(r["f0"])
        note = np.full(len(r["F"]), -1)
        for k, (a, b) in enumerate(r["notes"]):
            note[a:b] = k
        X.append(cd.score(r["F"][v], r["f0"][v], r["loud"][v]))
        y += [lab] * int(v.sum())
        units += [f"{s}{tr}{lab}n{k}" for k in note[v]]  # notes as units
        singers += [s] * int(v.sum())
        pitch.append(r["f0"][v])
    X, y, units, singers, pitch = np.concatenate(X), np.array(y, float), np.array(units), np.array(singers), np.concatenate(pitch)
    d = evaluate_promotion("breathy_direction", X, y, units, singers, pitch)
    assert d.passed, d.reasons
    control = evaluate_promotion("pitch_as_breathy", pitch, y, units, singers, pitch)
    assert not control.passed
    # the SAE view of the same features: some latent tracks pitch (a known attribute), reconstruction is good
    F = np.vstack([r["F"][np.isfinite(r["f0"])] for r in recs.values()])
    P = np.concatenate([r["f0"][np.isfinite(r["f0"])] for r in recs.values()])
    fit = train_sae(F, SAEConfig(n_latents=32, k=4, steps=800))
    assert fit.fvu(F) < 0.1
    assert match_features(fit.codes(F), {"pitch": (P, "regress")}).best_for_target("pitch", 1)[0].score > 0.5
