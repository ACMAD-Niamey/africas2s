"""Unit + equivalence tests for `calibrate(method="ereg", native_years=True)`.

`native_years=True` lets each model calibrate on its own `hcst.year ∩
obs.year` overlap instead of requiring every model's hindcast to cover every
obs year (the default, strict behavior in `_common_obs_hindcast_years`).
"""
import numpy as np
import pytest
import xarray as xr

import africas2s


def _grid(values, *, year_coords, name="x"):
    n_year, n_lat, n_lon = values.shape
    return xr.DataArray(
        values,
        dims=("year", "lat", "lon"),
        coords={
            "year": list(year_coords),
            "lat": np.linspace(-5.0, 5.0, n_lat),
            "lon": np.linspace(30.0, 40.0, n_lon),
        },
        name=name,
    )


def _make_predictor(values, *, member, year_coords, name="m"):
    n_year, n_mem, n_lat, n_lon = values.shape
    return xr.DataArray(
        values,
        dims=("year", "member", "lat", "lon"),
        coords={
            "year": list(year_coords),
            "member": list(range(n_mem)),
            "lat": np.linspace(-5.0, 5.0, n_lat),
            "lon": np.linspace(30.0, 40.0, n_lon),
        },
        name=name,
    )


def _minimal_obs(years):
    n_year = len(years)
    rng = np.random.default_rng(0)
    return _grid(rng.standard_normal((n_year, 4, 4)), year_coords=years, name="obs")


def _minimal_hcst(years, seed=1):
    n_year = len(years)
    rng = np.random.default_rng(seed)
    return _make_predictor(
        rng.standard_normal((n_year, 3, 4, 4)),
        member=3, year_coords=years, name="m",
    )


OBS_YEARS = list(range(1993, 2001))  # 1993-2000
HCST_YEARS = list(range(1995, 2001))  # 1995-2000: misses 1993, 1994


def test_default_still_raises_on_missing_obs_years():
    """native_years=False (default): unchanged strict behavior."""
    obs = _minimal_obs(OBS_YEARS)
    hcst = _minimal_hcst(HCST_YEARS)
    fcst = hcst.isel(year=[-1]).assign_coords(year=[2001])
    with pytest.raises(ValueError, match="missing obs years"):
        africas2s.calibrate(
            {"m1": (hcst, fcst)}, obs=obs, method="ereg", forecast_year=2001,
        )


def test_native_years_true_does_not_raise_and_returns_simplex():
    """native_years=True: trims to hcst ∩ obs and does not raise."""
    obs = _minimal_obs(OBS_YEARS)
    hcst = _minimal_hcst(HCST_YEARS)
    fcst = hcst.isel(year=[-1]).assign_coords(year=[2001])
    out = africas2s.calibrate(
        {"m1": (hcst, fcst)}, obs=obs, method="ereg",
        native_years=True, forecast_year=2001,
    )
    assert set(out.dims) == {"tercile", "lat", "lon"}
    assert out.sizes["tercile"] == 3
    total = out.sum("tercile")
    finite = np.isfinite(total)
    assert finite.any()
    np.testing.assert_allclose(total.values[finite.values], 1.0, atol=1e-8)


def test_native_years_true_raises_below_floor():
    """native_years=True still enforces a minimum overlap (>=3 years)."""
    obs = _minimal_obs(OBS_YEARS)
    hcst = _minimal_hcst([1999, 2000])  # only 2 overlap years with obs
    fcst = hcst.isel(year=[-1]).assign_coords(year=[2001])
    with pytest.raises(ValueError):
        africas2s.calibrate(
            {"m1": (hcst, fcst)}, obs=obs, method="ereg",
            native_years=True, forecast_year=2001,
        )


def test_native_years_false_default_unaffected_when_overlap_full():
    """native_years=False with a hindcast that fully covers obs years still works
    exactly as before (no regression to the non-native_years default path)."""
    obs = _minimal_obs(OBS_YEARS)
    hcst = _minimal_hcst(OBS_YEARS)
    fcst = hcst.isel(year=[-1]).assign_coords(year=[2001])
    out = africas2s.calibrate(
        {"m1": (hcst, fcst)}, obs=obs, method="ereg", forecast_year=2001,
    )
    assert out.sizes["tercile"] == 3
