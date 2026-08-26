import numpy as np
import pytest
import xarray as xr

from africas2s.methods.smoothed_regression import (
    seasonal_coefficients, _smooth_seasonal_coefficients,
)

SEASONS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ",
           "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]


def _cube(n_year=200, n_member=8, nlat=3, nlon=4, seed=0):
    rng = np.random.default_rng(seed)
    ns = len(SEASONS)
    a_season = 0.5 + 0.5 * np.cos(2 * np.pi * np.arange(ns) / ns)
    a_true = a_season[:, None, None] + 0.05 * rng.standard_normal((ns, nlat, nlon))
    signal = rng.standard_normal((ns, n_year, nlat, nlon))
    members = signal[:, :, None] + 0.3 * rng.standard_normal((ns, n_year, n_member, nlat, nlon))
    obs = a_true[:, None] * signal + 0.05 * rng.standard_normal((ns, n_year, nlat, nlon))
    coords = {"season": SEASONS, "year": np.arange(1991, 1991 + n_year),
              "member": np.arange(n_member), "lat": np.linspace(50, 60, nlat),
              "lon": np.linspace(-110, -100, nlon)}
    fc = xr.DataArray(members, dims=("season", "year", "member", "lat", "lon"), coords=coords)
    ob = xr.DataArray(obs, dims=("season", "year", "lat", "lon"),
                      coords={k: coords[k] for k in ("season", "year", "lat", "lon")})
    return fc, ob, a_true


def test_seasonal_coefficients_recovers_planted_unsmoothed():
    fc, ob, a_true = _cube()
    a = seasonal_coefficients(fc, ob, temporal_sigma=None)
    assert a.dims == ("season", "lat", "lon")
    np.testing.assert_allclose(a.values, a_true, atol=0.08)


def test_constant_shares_one_coefficient_across_seasons():
    fc, ob, _ = _cube()
    a = seasonal_coefficients(fc, ob, temporal_sigma="constant")
    assert float(a.std("season").max()) < 1e-9


def test_constant_is_pooled_regression():
    # 'constant' must be the pooled regression (the paper's time-invariant C): per-season
    # anomalies pooled, one slope per cell -- not the plain average of per-season slopes.
    fc, ob, _ = _cube(n_year=60)
    const = seasonal_coefficients(fc, ob, temporal_sigma="constant").isel(season=0)
    fbar = fc.mean("member")
    fa = fbar - fbar.mean("year")
    oa = ob - ob.mean("year")
    pooled = (fa * oa).sum(("season", "year")) / (fa * fa).sum(("season", "year"))
    np.testing.assert_allclose(const.values, pooled.transpose("lat", "lon").values,
                               rtol=1e-6, atol=1e-6)


def test_smoothing_reduces_season_variance():
    fc, ob, _ = _cube()
    unsm = seasonal_coefficients(fc, ob, temporal_sigma=None)
    sm = seasonal_coefficients(fc, ob, temporal_sigma=1.5)
    assert float(sm.std("season").mean()) < float(unsm.std("season").mean())


def test_smooth_is_cyclic():
    # spike at season 0 leaks to the last season (wrap), symmetric with season 1
    a = xr.DataArray(np.zeros((12, 1, 1)), dims=("season", "lat", "lon"),
                     coords={"season": SEASONS, "lat": [0], "lon": [0]})
    a[0, 0, 0] = 1.0
    sm = _smooth_seasonal_coefficients(a, 1.0)
    assert float(sm[-1, 0, 0]) > 0.0
    assert float(sm[1, 0, 0]) == pytest.approx(float(sm[-1, 0, 0]), abs=1e-9)


def test_sigma_none_is_identity():
    a = xr.DataArray(np.arange(12.0).reshape(12, 1, 1), dims=("season", "lat", "lon"),
                     coords={"season": SEASONS, "lat": [0], "lon": [0]})
    np.testing.assert_allclose(_smooth_seasonal_coefficients(a, None).values, a.values)
