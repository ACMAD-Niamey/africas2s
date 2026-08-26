"""Integration test for the climatology baseline downscaling method.

Exercises the full ``africas2s.downscale(method="climatology", ...)`` API
against synthetic GCM hindcast + obs data, asserting end-to-end behaviour
that the unit tests in test_methods.py don't cover individually:

- Method is reachable via the high-level API (not just direct import).
- Auto-forecast extraction from the trailing year of ``predictor_hindcast``
  works (mirrors the CV-loop / production calling pattern).
- Output shape is on the obs grid.
- Output equals the obs climatology at every member.
"""

import numpy as np

import africas2s


def test_downscale_climatology_end_to_end(synthetic_gcm_hindcast, synthetic_obs):
    """downscale(method='climatology') returns the obs climatology tiled per-member."""
    result = africas2s.downscale(
        predictor_hindcast=synthetic_gcm_hindcast,
        obs=synthetic_obs,
        method="climatology",
        output_type="continuous",
        verbose=False,
    )
    assert result.dims == ("member", "lat", "lon")
    assert len(result.lat) == len(synthetic_obs.lat)
    assert len(result.lon) == len(synthetic_obs.lon)

    # With auto-extracted forecast, the fit uses all years except the last.
    # So the climatology is over (year[:-1]).
    expected_clim = synthetic_obs.isel(year=slice(None, -1)).mean("year").values
    for m_idx in range(len(result.member)):
        np.testing.assert_allclose(result.isel(member=m_idx).values, expected_clim)


def test_downscale_climatology_ignores_forecast_choice(
    synthetic_gcm_hindcast, synthetic_gcm_forecast, synthetic_obs,
):
    """Passing an explicit (different) forecast must yield the obs climatology."""
    result = africas2s.downscale(
        predictor_hindcast=synthetic_gcm_hindcast,
        obs=synthetic_obs,
        forecast=synthetic_gcm_forecast,
        method="climatology",
        output_type="continuous",
        verbose=False,
    )
    # With explicit forecast, fit uses full hindcast, so climatology is over all years.
    expected_clim = synthetic_obs.mean("year").values
    for m_idx in range(len(result.member)):
        np.testing.assert_allclose(result.isel(member=m_idx).values, expected_clim)
