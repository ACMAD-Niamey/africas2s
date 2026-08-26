import numpy as np
import pytest
import xarray as xr
from scipy import stats
from scipy.integrate import trapezoid

from africas2s.metrics.crpss import crps_normal, crps_climatology, crpss
from africas2s.registry import get_metric


def test_crps_normal_matches_bruteforce():
    mu, sigma, y = 1.0, 2.0, 0.5
    x = np.linspace(mu - 40 * sigma, mu + 40 * sigma, 400001)
    brute = trapezoid((stats.norm.cdf(x, mu, sigma) - (x >= y).astype(float)) ** 2, x)
    assert crps_normal(mu, sigma, y) == pytest.approx(brute, rel=1e-4)


def test_crpss_perfect_and_reference():
    assert crpss(0.0, 1.5) == pytest.approx(1.0)
    assert crpss(1.5, 1.5) == pytest.approx(0.0)


def test_crpss_metric_registered_and_scores_gaussian():
    m = get_metric("crpss")()
    ny, nla, nlo = 30, 2, 3
    rng = np.random.default_rng(0)
    dims = ("year", "lat", "lon")
    coords = {"year": np.arange(ny), "lat": [0, 1], "lon": [0, 1, 2]}
    o = xr.DataArray(rng.standard_normal((ny, nla, nlo)), dims=dims, coords=coords)
    fc = xr.Dataset({"mu": o * 0.7, "sigma": xr.ones_like(o) * 0.8})
    val = m.compute(fc, o, spatial=False)
    assert np.isfinite(val) and val <= 1.0


def test_crpss_metric_rejects_bare_dataarray():
    m = get_metric("crpss")()
    o = xr.DataArray(np.zeros((5, 2, 2)), dims=("year", "lat", "lon"))
    with pytest.raises((ValueError, KeyError, AttributeError)):
        m.compute(o, o)
