"""Unit tests for `africas2s.pev.prediction_error_variance`."""
import numpy as np
import pytest
import xarray as xr

from africas2s.pev import prediction_error_variance


def _gridded(values, years):
    """Helper: wrap a (year, lat, lon) array into a DataArray with coords."""
    n_year, n_lat, n_lon = values.shape
    return xr.DataArray(
        values,
        dims=("year", "lat", "lon"),
        coords={
            "year": list(years),
            "lat": np.linspace(-5.0, 5.0, n_lat),
            "lon": np.linspace(30.0, 40.0, n_lon),
        },
    )


def test_recovery_against_known_sigma():
    """With n=50 years of zero-mean Gaussian noise σ²=1, the spatial-mean
    of recovered PEV should be within 5% of σ². Per-cell tolerance would
    be flaky (per-cell stdev ≈ σ²·√(2/n) ≈ 20%), so we check the spatial
    mean — that's the quantity the spec's "within a percent or two" intent
    was about.
    """
    rng = np.random.default_rng(seed=42)
    n_year, n_lat, n_lon = 50, 8, 8
    sigma2 = 1.0
    obs_vals = rng.standard_normal((n_year, n_lat, n_lon))
    # pred = obs + noise, where noise has variance sigma2. Then
    # (pred - obs) = noise, so the true residual variance is exactly sigma2.
    pred_vals = obs_vals + rng.standard_normal((n_year, n_lat, n_lon)) * np.sqrt(sigma2)

    years = list(range(1991, 1991 + n_year))
    obs = _gridded(obs_vals, years)
    pred = _gridded(pred_vals, years)

    pev = prediction_error_variance(pred, obs)
    assert pev.dims == ("lat", "lon")
    assert abs(float(pev.mean()) - sigma2) / sigma2 < 0.05


def test_disjoint_years_raises():
    obs = _gridded(np.zeros((3, 2, 2)), years=[2000, 2001, 2002])
    pred = _gridded(np.zeros((3, 2, 2)), years=[2010, 2011, 2012])
    with pytest.raises(ValueError) as exc:
        prediction_error_variance(pred, obs)
    msg = str(exc.value)
    assert "2010" in msg and "2000" in msg


def test_extra_year_in_pred_raises():
    obs = _gridded(np.zeros((3, 2, 2)), years=[2000, 2001, 2002])
    pred = _gridded(np.zeros((4, 2, 2)), years=[2000, 2001, 2002, 2003])
    with pytest.raises(ValueError) as exc:
        prediction_error_variance(pred, obs)
    assert "2003" in str(exc.value)


def test_extra_year_in_obs_raises():
    obs = _gridded(np.zeros((4, 2, 2)), years=[2000, 2001, 2002, 2003])
    pred = _gridded(np.zeros((3, 2, 2)), years=[2000, 2001, 2002])
    with pytest.raises(ValueError) as exc:
        prediction_error_variance(pred, obs)
    assert "2003" in str(exc.value)


def test_year_order_does_not_matter():
    rng = np.random.default_rng(0)
    obs = _gridded(rng.standard_normal((5, 2, 2)), years=[2000, 2001, 2002, 2003, 2004])
    pred = _gridded(rng.standard_normal((5, 2, 2)), years=[2004, 2003, 2002, 2001, 2000])
    # Should not raise; alignment by coordinate.
    pev = prediction_error_variance(pred, obs)
    assert pev.dims == ("lat", "lon")


def test_nan_propagation_per_cell():
    """Ocean cells (all-NaN over years in obs) should stay NaN in the
    output. Land cells are unaffected.
    """
    rng = np.random.default_rng(0)
    obs_vals = rng.standard_normal((10, 3, 3))
    pred_vals = obs_vals + rng.standard_normal((10, 3, 3)) * 0.5
    # Mark one cell as all-NaN across years in obs (ocean cell).
    obs_vals[:, 0, 0] = np.nan
    obs = _gridded(obs_vals, years=list(range(2000, 2010)))
    pred = _gridded(pred_vals, years=list(range(2000, 2010)))

    pev = prediction_error_variance(pred, obs)
    assert np.isnan(float(pev.isel(lat=0, lon=0)))
    assert not np.isnan(float(pev.isel(lat=1, lon=1)))


def test_one_dimensional_station_layout():
    """Function should work when the spatial layout is a 1D station dim,
    not a (lat, lon) grid. Output dims = input dims minus 'year'.
    """
    n_year, n_station = 30, 5
    rng = np.random.default_rng(1)
    obs_vals = rng.standard_normal((n_year, n_station))
    pred_vals = obs_vals + rng.standard_normal((n_year, n_station))
    coords = {
        "year": list(range(2000, 2000 + n_year)),
        "station": [f"s{i}" for i in range(n_station)],
    }
    obs = xr.DataArray(obs_vals, dims=("year", "station"), coords=coords)
    pred = xr.DataArray(pred_vals, dims=("year", "station"), coords=coords)

    pev = prediction_error_variance(pred, obs)
    assert pev.dims == ("station",)
    assert pev.shape == (n_station,)


def test_top_level_export():
    import africas2s
    assert africas2s.prediction_error_variance is prediction_error_variance
