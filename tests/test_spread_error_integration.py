"""End-to-end check for the spread-error metrics through the public skill API.

Exercises the realistic call path (`skill(...)` with a raw-member forecast)
to catch composition regressions that unit tests in isolation can miss.
"""
import math

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def ensemble_forecast_and_obs():
    np.random.seed(7)
    n_year, n_member, n_lat, n_lon = 20, 6, 4, 4
    coords = {
        "year": np.arange(n_year),
        "member": np.arange(n_member),
        "lat": np.linspace(-1, 1, n_lat),
        "lon": np.linspace(0, 1, n_lon),
    }
    truth = np.random.randn(n_year, n_lat, n_lon)
    amplitude = np.linspace(0.5, 3.0, n_year)
    noise = (
        np.random.randn(n_year, n_member, n_lat, n_lon)
        * amplitude[:, None, None, None]
    )
    fcst = truth[:, None, :, :] + noise

    forecast = xr.DataArray(
        fcst, dims=["year", "member", "lat", "lon"], coords=coords
    )
    obs = xr.DataArray(
        truth, dims=["year", "lat", "lon"],
        coords={k: coords[k] for k in ("year", "lat", "lon")},
    )
    return forecast, obs


def test_skill_includes_spread_error_metrics(ensemble_forecast_and_obs):
    """skill() resolves both names and returns finite values for both."""
    from africas2s.skill import skill

    forecast, obs = ensemble_forecast_and_obs
    report = skill(
        forecast, obs,
        metrics=["spread_error_ratio", "spread_error_correlation"],
    )

    assert "spread_error_ratio" in report.scores
    assert "spread_error_correlation" in report.scores

    ratio = report.scores["spread_error_ratio"]
    corr = report.scores["spread_error_correlation"]
    assert math.isfinite(ratio), f"ratio not finite: {ratio}"
    assert math.isfinite(corr), f"corr not finite: {corr}"
    # Synthetic ensemble has year-varying noise → spread should track error.
    assert corr > 0.5, f"expected positive tracking, got {corr}"
