import numpy as np
import pytest
from africas2s.methods import smoothed_regression as pb


def test_fit_ab_constrained_recovers_amplitude():
    rng = np.random.default_rng(0); n = 4000
    signal = rng.standard_normal(n)
    mu_f = signal + 0.1 * rng.standard_normal(n)
    sigma_f = np.full(n, 0.5)
    o = 0.8 * signal + 0.3 * rng.standard_normal(n)
    a, b = pb.fit_ab(mu_f, sigma_f, o, constrained=True)
    assert a == pytest.approx(0.8, abs=0.05)
    assert (a**2 * np.var(mu_f) + b**2 * np.mean(sigma_f**2)) == pytest.approx(np.var(o), rel=0.05)


def test_fit_ab_unconstrained_not_worse_than_raw_crps():
    from africas2s.metrics.crpss import crps_normal
    rng = np.random.default_rng(1); n = 2000
    signal = rng.standard_normal(n); mu_f = signal; sigma_f = np.full(n, 1.0)
    o = 0.5 * signal + 0.2 * rng.standard_normal(n)
    a, b = pb.fit_ab(mu_f, sigma_f, o, constrained=False)
    assert crps_normal(a * mu_f, b * sigma_f, o).mean() <= crps_normal(mu_f, sigma_f, o).mean() + 1e-9


def test_normal_category_probs_sum_to_one():
    p = pb.normal_category_probs(2.0, 1.0, -0.43, 0.43)
    assert p.shape == (3,) and p.sum() == pytest.approx(1.0) and p[2] > p[0]


def test_smooth_ab_season_axis_and_constant():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((12, 2, 3)); b = np.abs(rng.standard_normal((12, 2, 3))) + 0.5
    a_c, _ = pb.smooth_ab(a, b, "constant")
    assert float(np.std(a_c, axis=0).max()) < 1e-12
    np.testing.assert_allclose(a_c[0], a.mean(axis=0))
    a_f, _ = pb.smooth_ab(a, b, 1.5)
    assert float(np.std(a_f, axis=0).mean()) <= float(np.std(a, axis=0).mean()) + 1e-9
    a_n, _ = pb.smooth_ab(a, b, None); np.testing.assert_allclose(a_n, a)


def test_gamma_roundtrip_and_moments():
    rng = np.random.default_rng(3)
    x = rng.gamma(2.0, 1.5, 100000)
    k, th = pb.fit_gamma(x)
    assert k == pytest.approx(2.0, rel=0.05) and th == pytest.approx(1.5, rel=0.05)
    x2 = rng.gamma(2.0, 1.5, 1000) + 0.01
    k2, th2 = pb.fit_gamma(x2)
    np.testing.assert_allclose(pb.normal_to_gamma(pb.gamma_to_normal(x2, k2, th2), k2, th2),
                               x2, rtol=1e-4, atol=1e-4)
