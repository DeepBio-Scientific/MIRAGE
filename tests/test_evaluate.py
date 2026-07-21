import numpy as np

from fambench.evaluate import evaluate_redundancy, evaluate_temporal


def test_perfect_predictor_scores_high():
    rng = np.random.default_rng(0)
    n = 600
    y = rng.uniform(3, 10, n)
    fam = rng.choice([1, 3, 10, 50, 150, 400], n)
    rep = evaluate_redundancy(y, y.copy(), fam)  # perfect predictions
    assert rep.overall_pearson > 0.99
    assert rep.novel_family_pearson > 0.9


def test_noise_predictor_scores_zero():
    rng = np.random.default_rng(1)
    n = 600
    y = rng.uniform(3, 10, n)
    fam = rng.choice([1, 3, 10, 50, 150, 400], n)
    rep = evaluate_redundancy(y, rng.normal(size=n), fam)
    assert abs(rep.overall_pearson) < 0.15


def test_lower_is_stronger_flips_sign():
    rng = np.random.default_rng(2)
    n = 400
    y = rng.uniform(3, 10, n)
    fam = rng.choice([1, 10, 100], n)
    # a log10(IC50)-style score: perfectly anti-correlated with pK
    rep = evaluate_redundancy(y, -y, fam, higher_is_stronger=False)
    assert rep.overall_pearson > 0.99


def test_leakage_slope_detects_redundancy_dependence():
    rng = np.random.default_rng(3)
    n = 1200
    y = rng.uniform(3, 10, n)
    fam = rng.choice([1, 3, 10, 50, 150, 400], n)
    # predictions good only for big families, noise for small ones
    pred = np.where(fam >= 50, y + rng.normal(0, 0.5, n), rng.normal(6, 2, n))
    rep = evaluate_redundancy(y, pred, fam)
    assert rep.leakage_slope < 0  # abs-error falls as family size grows
    assert rep.redundant_family_pearson > rep.novel_family_pearson


def test_temporal_report_and_residual():
    rng = np.random.default_rng(4)
    n = 300
    y = rng.uniform(3, 8, n)
    mw = y + rng.normal(0, 1, n)          # MW carries most of the signal
    pred = mw + rng.normal(0, 0.5, n)     # model ~ MW + a little extra
    rep = evaluate_temporal(y, pred, baselines={"molecular_weight": mw})
    assert "molecular_weight" in rep.baseline_spearman
    assert np.isfinite(rep.residual_vs_mw_spearman)
