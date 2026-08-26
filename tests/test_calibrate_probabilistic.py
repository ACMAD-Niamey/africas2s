import numpy as np
import pytest
import xarray as xr
import africas2s

SEASONS = ["DJF","JFM","FMA","MAM","AMJ","MJJ","JJA","JAS","ASO","SON","OND","NDJ"]

def _cube(seed=0, nla=2, nlo=3, ny=25, nm=8):
    rng = np.random.default_rng(seed); ns = 12
    sig = rng.standard_normal((ns, ny, nla, nlo))
    mem = sig[:, :, None] + 0.4 * rng.standard_normal((ns, ny, nm, nla, nlo))
    obs = 0.6 * sig + 0.3 * rng.standard_normal((ns, ny, nla, nlo))
    c = {"season": SEASONS, "year": np.arange(1991, 1991+ny), "member": np.arange(nm),
         "lat": np.linspace(50,60,nla), "lon": np.linspace(-110,-100,nlo)}
    fc = xr.DataArray(mem, dims=("season","year","member","lat","lon"), coords=c)
    ob = xr.DataArray(obs, dims=("season","year","lat","lon"),
                      coords={k:c[k] for k in ("season","year","lat","lon")})
    return fc, ob

def test_tercile_output_shape_and_simplex_normal():
    fc, ob = _cube()
    out = africas2s.calibrate(fc, ob, method="smoothed_regression", output_type="tercile",
                              distribution="normal", temporal_sigma="constant", forecast_year=2010)
    assert out.dims == ("season", "tercile", "lat", "lon")
    assert out.sizes["tercile"] == 3
    np.testing.assert_allclose(out.sum("tercile").values, 1.0, atol=1e-9)

def test_tercile_output_gamma_nonneg_consistent():
    fc, ob = _cube(seed=1)
    fc = np.abs(fc) + 0.1; ob = np.abs(ob) + 0.1     # positive "precip"
    out = africas2s.calibrate(fc, ob, method="smoothed_regression", output_type="tercile",
                              distribution="gamma", temporal_sigma="1", forecast_year=2010)
    assert out.dims == ("season", "tercile", "lat", "lon")
    np.testing.assert_allclose(out.sum("tercile").values, 1.0, atol=1e-9)
    assert bool((out >= 0).all())

def test_deterministic_path_unchanged():
    fc, ob = _cube()
    out = africas2s.calibrate(fc, ob, method="smoothed_regression",
                              output_type="deterministic", temporal_sigma="constant", forecast_year=2010)
    assert out.dims == ("season", "lat", "lon")

def test_tercile_requires_member_dim():
    # tercile needs the ensemble spread; a member-less predictor is rejected up front.
    fc, ob = _cube()
    with pytest.raises(ValueError, match="requires an ensemble predictor"):
        africas2s.calibrate(fc.mean("member"), ob, method="smoothed_regression",
                            output_type="tercile", temporal_sigma="constant", forecast_year=2010)

def test_tercile_unknown_distribution_raises():
    fc, ob = _cube()
    with pytest.raises(ValueError, match="unknown distribution"):
        africas2s.calibrate(fc, ob, method="smoothed_regression", output_type="tercile",
                            distribution="lognormal", temporal_sigma="constant", forecast_year=2010)
