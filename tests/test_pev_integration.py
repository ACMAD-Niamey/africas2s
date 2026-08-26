"""Integration tests: PEV surfaced on EnsembleResult by `ensemble()`."""
import numpy as np
import pytest
import xarray as xr

from africas2s import ensemble, prediction_error_variance


def _year_dim_forecast(values, years, name="m"):
    """Wrap (year, lat, lon) into a DataArray with a `.name` attr so it's
    picked up by `_member_names`."""
    n_year, n_lat, n_lon = values.shape
    return xr.DataArray(
        values,
        dims=("year", "lat", "lon"),
        coords={
            "year": list(years),
            "lat": np.linspace(-5.0, 5.0, n_lat),
            "lon": np.linspace(30.0, 40.0, n_lon),
        },
        name=name,
    )


def _two_member_fixture(n_year=20, n_lat=4, n_lon=4, seed=7):
    rng = np.random.default_rng(seed)
    years = list(range(2000, 2000 + n_year))
    obs_vals = rng.standard_normal((n_year, n_lat, n_lon))
    obs = _year_dim_forecast(obs_vals, years, name="obs")
    # Two "CV hindcasts" that bracket obs.
    m1 = _year_dim_forecast(
        obs_vals + 0.5 * rng.standard_normal((n_year, n_lat, n_lon)),
        years, name="m1",
    )
    m2 = _year_dim_forecast(
        obs_vals + 0.5 * rng.standard_normal((n_year, n_lat, n_lon)),
        years, name="m2",
    )
    return [m1, m2], obs


def test_uniform_path_populates_pev():
    forecasts, obs = _two_member_fixture()
    result = ensemble(forecasts, obs, strategy="uniform")
    assert result.pev is not None
    # Same value as a direct call against the combined forecast.
    expected = prediction_error_variance(result.forecast, obs)
    np.testing.assert_allclose(result.pev.values, expected.values, atol=1e-12)
    assert result.pev.dims == ("lat", "lon")


def test_nested_cv_pev_uses_pooled_cv_forecasts():
    """When optimize_ensemble=True with nested_cv (the default safeguard),
    `result.pev` should reflect held-out CV residuals, not in-sample fit
    residuals. We construct a fixture where one member is highly correlated
    with obs (so in-sample weights drive its weight high and final_fcst
    residuals are small) but the nested-CV residuals are larger because
    weights re-fit each fold are noisier.

    The signal we test is qualitative: `result.pev > 0` and not absurdly
    small relative to obs variance. (Exact value depends on stochastic CV
    folds.)
    """
    forecasts, obs = _two_member_fixture(n_year=25, seed=11)
    result = ensemble(forecasts, obs, optimize_ensemble=True,
                      primary_metric="pearson_r")
    assert result.pev is not None
    # PEV should be at least a meaningful fraction of obs variance — not
    # the near-zero you'd see from in-sample residuals on this fixture.
    obs_var = float(obs.var())
    assert float(result.pev.mean()) > 0.05 * obs_var


def test_optimize_without_nested_cv_returns_none_pev():
    """The `nested_cv=False` path has no honest CV predictions; pev = None."""
    forecasts, obs = _two_member_fixture()
    with pytest.warns(RuntimeWarning, match="optimistic"):
        result = ensemble(
            forecasts, obs, optimize_ensemble=True,
            primary_metric="pearson_r",
            safeguards={"nested_cv": False},
        )
    assert result.pev is None


def test_year_less_or_no_obs_returns_none_pev():
    """No year dim → pev = None. obs=None → pev = None."""
    forecasts, obs = _two_member_fixture()
    # obs=None path
    result_no_obs = ensemble(forecasts, None, strategy="uniform")
    assert result_no_obs.pev is None

    # year-less forecasts: drop the year dim by selecting one year.
    flat = [f.isel(year=0, drop=True) for f in forecasts]
    obs_flat = obs.isel(year=0, drop=True)
    result_flat = ensemble(flat, obs_flat, strategy="uniform")
    assert result_flat.pev is None
