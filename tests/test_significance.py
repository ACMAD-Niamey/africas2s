"""Tests for cross-validation + significance helpers (africas2s.metrics)."""
import numpy as np
import pytest

from africas2s.metrics import loo_predict, loo_corr, leverage, permutation_test, fdr


# ---- leave-one-out cross-validation --------------------------------------------------

def test_loo_predict_matches_bruteforce_refit():
    rng = np.random.default_rng(0)
    x = rng.normal(size=25)
    y = 0.7 * x + rng.normal(size=25)
    closed = loo_predict(x, y)
    brute = np.empty_like(y)
    for i in range(len(y)):
        keep = np.ones(len(y), bool); keep[i] = False
        b, a = np.polyfit(x[keep], y[keep], 1)
        brute[i] = a + b * x[i]
    assert np.allclose(closed, brute, atol=1e-10)


def test_loo_predict_constant_predictor_is_nan():
    y = np.arange(10.0)
    assert np.all(np.isnan(loo_predict(np.ones(10), y)))


def test_leverage_sums_to_two_for_simple_regression():
    # sum of leverages in a simple linear regression is exactly the number of parameters (2)
    x = np.linspace(-3, 3, 40)
    assert leverage(x).sum() == pytest.approx(2.0, abs=1e-9)


def test_loo_corr_perfect_and_none():
    x = np.linspace(0, 1, 30)
    assert loo_corr(x, 2 * x + 1) == pytest.approx(1.0, abs=1e-9)      # perfect linear
    assert np.isnan(loo_corr(np.ones(30), x))                          # degenerate predictor


def test_loo_corr_negative_for_noise():
    # the defining property: a no-skill predictor gives NEGATIVE loo correlation on average
    rng = np.random.default_rng(1)
    vals = [loo_corr(rng.normal(size=24), rng.normal(size=24)) for _ in range(200)]
    assert np.nanmean(vals) < 0


def test_loo_corr_too_few_points():
    assert np.isnan(loo_corr([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], min_finite=5))


# ---- permutation test ----------------------------------------------------------------

def test_permutation_detects_real_signal():
    rng = np.random.default_rng(2)
    x = rng.normal(size=40)
    y = 0.8 * x + rng.normal(size=40) * 0.5
    obs, p = permutation_test(x, y, n=2000, rng=np.random.default_rng(0))
    assert obs > 0.4 and p < 0.01


def test_permutation_null_is_not_significant():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=30), rng.normal(size=30)
    _, p = permutation_test(x, y, n=2000, rng=np.random.default_rng(0))
    assert p > 0.10


def test_permutation_is_reproducible_with_seed():
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=28), rng.normal(size=28)
    p1 = permutation_test(x, y, n=500, rng=np.random.default_rng(7))[1]
    p2 = permutation_test(x, y, n=500, rng=np.random.default_rng(7))[1]
    assert p1 == p2


def test_permutation_pvalue_never_zero():
    x = np.linspace(0, 1, 20)
    _, p = permutation_test(x, 3 * x, n=100, rng=np.random.default_rng(0))
    assert p == pytest.approx(1 / 101)          # observed is the max, so count == 1


def test_permutation_custom_statistic_two_sided():
    from scipy.stats import pearsonr
    rng = np.random.default_rng(5)
    x = rng.normal(size=50); y = -0.7 * x + rng.normal(size=50) * 0.5
    stat = lambda a, b: pearsonr(a, b)[0]
    obs, p = permutation_test(x, y, statistic=stat, alternative="two-sided", n=1000,
                              rng=np.random.default_rng(0))
    assert obs < 0 and p < 0.01                 # strong negative correlation is two-sided-significant


def test_permutation_rejects_bad_alternative():
    with pytest.raises(ValueError):
        permutation_test([1, 2, 3], [1, 2, 3], alternative="sideways")


# ---- Benjamini-Hochberg FDR ----------------------------------------------------------

def test_fdr_monotone_and_bounded():
    p = np.array([0.001, 0.008, 0.02, 0.04, 0.2, 0.5, 0.9])
    q = fdr(p)
    assert np.all(q >= p)                        # q is never below its p
    assert np.all(np.diff(q) >= -1e-12)          # non-decreasing in p order
    assert q.max() <= 1.0


def test_fdr_passes_through_nan():
    p = [0.01, np.nan, 0.5, np.nan, 0.001]
    q = fdr(p)
    assert np.isnan(q[1]) and np.isnan(q[3])
    assert np.isfinite(q[0]) and np.isfinite(q[4])
    # m counts only the 3 finite p-values
    assert q[4] == pytest.approx(0.001 * 3 / 1)


def test_fdr_all_nan():
    assert np.all(np.isnan(fdr([np.nan, np.nan])))


def test_fdr_rejects_unknown_method():
    with pytest.raises(ValueError):
        fdr([0.1, 0.2], method="holm")
