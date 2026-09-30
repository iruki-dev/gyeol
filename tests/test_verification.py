import numpy as np
import pytest

from gyeol.verification import Record, degrade, invariance_report, operating_threshold, stats


def test_icc_extremes_and_known_value():
    rng = np.random.default_rng(0)
    true = rng.normal(0, 10, 50)
    y = np.stack([true, true, true], axis=1)
    assert stats.icc(y, "2,1") == pytest.approx(1.0)
    noise = rng.normal(0, 1, (500, 3))
    assert abs(stats.icc(noise, "2,1")) < 0.1
    # a constant device offset lowers absolute agreement, not consistency
    off = np.stack([true, true + 5.0], axis=1)
    assert stats.icc(off, "3,1") == pytest.approx(1.0)
    assert stats.icc(off, "2,1") < 0.9
    # Shrout & Fleiss (1979) Table 2 example: ICC(2,1)=0.29, ICC(3,1)=0.71, ICC(1,1)=0.17
    sf = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], float)
    assert stats.icc(sf, "1,1") == pytest.approx(0.17, abs=0.01)
    assert stats.icc(sf, "2,1") == pytest.approx(0.29, abs=0.01)
    assert stats.icc(sf, "3,1") == pytest.approx(0.71, abs=0.01)


def test_bland_altman_sem_mdc():
    a = np.arange(100.0)
    b = a + 2.0 + np.random.default_rng(1).normal(0, 1, 100)
    ba = stats.bland_altman(a, b)
    assert ba.bias == pytest.approx(2.0, abs=0.3)
    assert ba.loa_high - ba.loa_low == pytest.approx(2 * 1.96 * ba.sd)
    assert stats.mdc95(stats.sem(10.0, 0.91)) == pytest.approx(1.96 * np.sqrt(2) * 3.0)
    assert stats.koo_li(0.95) == "excellent" and stats.koo_li(0.6) == "moderate"


def test_within_between_chance_eer_tost_holm():
    subj = np.repeat(np.arange(10), 5)
    vals = subj * 10.0 + np.random.default_rng(2).normal(0, 1, 50)
    assert stats.within_between_ratio(vals, subj) < 0.05
    assert stats.chance_test(20, 100, 5) > 0.05  # 20 % = chance for 5 classes
    assert stats.chance_test(60, 100, 5) < 1e-6
    scores = np.r_[np.random.default_rng(3).normal(2, 1, 500), np.random.default_rng(4).normal(-2, 1, 500)]
    labels = np.r_[np.ones(500), np.zeros(500)]
    assert stats.equal_error_rate(scores, labels) == pytest.approx(0.023, abs=0.015)
    a = np.random.default_rng(5).normal(0, 1, 200)
    assert stats.tost_paired(a, a + 0.01, margin=0.5) < 0.01
    assert stats.tost_paired(a, a + 1.0, margin=0.5) > 0.5
    assert stats.holm({"x": 0.001, "y": 0.02, "z": 0.04}) == {"x": True, "y": True, "z": True}
    assert stats.holm({"x": 0.001, "y": 0.04, "z": 0.04}) == {"x": True, "y": False, "z": False}


def test_operating_threshold_finds_crossing():
    rng = np.random.default_rng(0)
    snr = rng.uniform(0, 50, 2000)
    err = 10 * np.exp(-snr / 10) * np.abs(rng.normal(1, 0.2, 2000))  # error shrinks with SNR
    res = operating_threshold(snr, err, mdc=1.0, higher_is_better=True)
    # bound = q95 of 10·exp(−snr/10)·|N(1, .2)| ≈ 13.3·exp(−snr/10) → crosses 1 near 26 dB
    assert 20 < res.threshold < 32
    assert res.is_valid(40) and not res.is_valid(10)
    t60 = rng.uniform(0, 2, 2000)
    res2 = operating_threshold(t60, t60 * abs(rng.normal(1, 0.1, 2000)), mdc=0.5, higher_is_better=False)
    assert 0.2 < res2.threshold < 0.5


def test_degradations():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16000)
    y = degrade.add_noise(x, 20, active_only=False)
    assert 10 * np.log10(np.mean(x**2) / np.mean((y - x) ** 2)) == pytest.approx(20, abs=0.1)
    rir = degrade.synthetic_rir(16000, t60=0.5, drr_db=0.0)
    energy = np.cumsum(rir[::-1] ** 2)[::-1]
    edc = 10 * np.log10(energy / energy[0])
    t = np.arange(len(rir)) / 16000
    sel = (edc < -5) & (edc > -25)
    slope = np.polyfit(t[sel], edc[sel], 1)[0]
    assert -60 / slope == pytest.approx(0.5, rel=0.15)
    assert len(degrade.standard_grid()) >= 1 + 11 + 4 + 3


def test_invariance_report():
    rng = np.random.default_rng(0)
    records = []
    for s in range(20):
        base = rng.normal(15, 3)
        for dev, (bias, sd) in {"ref": (0, 0.1), "phone": (0.05, 0.2), "zoom": (-4.5, 0.2)}.items():
            records.append(Record(f"s{s}", dev, {"cpps": base + bias + rng.normal(0, sd)}))
    good = invariance_report([r for r in records if r.condition != "zoom"], reference="ref", n_boot=100)
    bad = invariance_report(records, reference="ref", n_boot=100)
    assert good["cpps"].accepted and good["cpps"].band == "excellent"
    assert not bad["cpps"].accepted
    assert bad["cpps"].bias_vs_reference["zoom"] == pytest.approx(-4.5, abs=0.3)
