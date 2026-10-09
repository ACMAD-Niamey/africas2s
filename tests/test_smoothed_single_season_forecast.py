"""A single-season forecast must land in its own season's slot.

smoothed_regression's tercile path applies the gamma parameters and the (a, b)
coefficients season by season, indexed positionally. Before the season
alignment, a forecast covering only the target season was transformed through
season-zero's gamma — silently, with no error — so the result was wrong
whenever the target was not the first season.
"""
import numpy as np
import pytest
import xarray as xr

import africas2s

SEASONS = ["JFM", "FMA", "MAM", "AMJ"]
TARGET = "MAM"
TI = SEASONS.index(TARGET)


def _case(seed=0, ny=25, nm=6, nlat=4, nlon=5):
    rng = np.random.default_rng(seed)
    lat = np.linspace(-5, 5, nlat)
    lon = np.linspace(30, 40, nlon)
    years = np.arange(1990, 1990 + ny)
    # season-dependent scale, so using the wrong season's gamma is detectable
    s4 = np.array([1.0, 3.0, 8.0, 5.0]).reshape(len(SEASONS), 1, 1, 1)
    s5 = s4.reshape(len(SEASONS), 1, 1, 1, 1)
    obs = rng.gamma(2.0, 1.0, (len(SEASONS), ny, nlat, nlon)) * s4
    pred = (obs[:, :, None, :, :] * 0.7
            + rng.gamma(2.0, 1.0, (len(SEASONS), ny, nm, nlat, nlon)) * 0.3 * s5)
    obs_da = xr.DataArray(obs, dims=("season", "year", "lat", "lon"),
                          coords={"season": SEASONS, "year": years, "lat": lat, "lon": lon})
    pred_da = xr.DataArray(pred, dims=("season", "year", "member", "lat", "lon"),
                           coords={"season": SEASONS, "year": years,
                                   "member": np.arange(nm), "lat": lat, "lon": lon})
    fmem = xr.DataArray(rng.gamma(2.0, 1.0, (nm, nlat, nlon)) * float(s4[TI, 0, 0, 0]),
                        dims=("member", "lat", "lon"),
                        coords={"member": np.arange(nm), "lat": lat, "lon": lon})
    return pred_da, obs_da, fmem


def _run(pred, obs, fcst):
    return africas2s.calibrate(pred, obs=obs, method="smoothed_regression",
                               forecast=fcst, output_type="tercile",
                               distribution="gamma", temporal_sigma=1.0,
                               constrained=True)


def test_single_season_forecast_matches_manually_padded():
    """Passing only the target season == padding it into the full season axis."""
    pred, obs, fmem = _case()
    single = fmem.expand_dims(season=[TARGET])

    padded = np.full((len(SEASONS),) + fmem.shape, np.nan)
    padded[TI] = fmem.values
    full = xr.DataArray(padded, dims=("season", "member", "lat", "lon"),
                        coords={"season": SEASONS, "member": fmem["member"].values,
                                "lat": fmem["lat"].values, "lon": fmem["lon"].values})

    a = _run(pred, obs, single).sel(season=TARGET)
    b = _run(pred, obs, full).sel(season=TARGET)
    np.testing.assert_allclose(a.values, b.values, rtol=0, atol=0)


def test_single_season_forecast_uses_its_own_gamma():
    """The target's own season must be used, not season zero's."""
    pred, obs, fmem = _case()
    got = _run(pred, obs, fmem.expand_dims(season=[TARGET])).sel(season=TARGET)
    # what the old positional behaviour produced: the same numbers read as season 0
    wrong = _run(pred, obs, fmem.expand_dims(season=[SEASONS[0]])).sel(season=SEASONS[0])
    assert np.nanmax(np.abs(got.values - wrong.values)) > 1e-6
    np.testing.assert_allclose(got.sum("tercile").values, 1.0, rtol=1e-9)


def test_unknown_season_label_is_an_error():
    pred, obs, fmem = _case()
    with pytest.raises(ValueError, match="not in the predictor"):
        _run(pred, obs, fmem.expand_dims(season=["ZZZ"]))
