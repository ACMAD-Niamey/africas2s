"""Integration test for the rank-analog downscaling method.

Exercises the full `africas2s.downscale(method="rank-analog", ...)` API
against synthetic GCM hindcast + obs data, asserting end-to-end behavior
that the unit tests in test_methods.py don't cover individually:

- Method is reachable via the high-level API (not just via direct import).
- The auto-forecast extraction from the trailing year of `predictor_hindcast`
  works (mirrors the CV-loop / production calling pattern).
- Output shape is on the obs grid.
- Output values fall in the observed climatology range.
"""

import numpy as np
import pytest

import africas2s


def test_downscale_rank_analog_end_to_end(synthetic_gcm_hindcast, synthetic_obs):
    """downscale() with method='rank-analog' returns an obs-grid forecast
    derived by ranking the last hindcast year against the rest."""
    result = africas2s.downscale(
        predictor_hindcast=synthetic_gcm_hindcast,
        obs=synthetic_obs,
        method="rank-analog",
        output_type="continuous",
        verbose=False,
    )
    assert result.dims == ("member", "lat", "lon")
    assert len(result.lat) == len(synthetic_obs.lat)
    assert len(result.lon) == len(synthetic_obs.lon)


def test_downscale_rank_analog_values_in_obs_range(synthetic_gcm_hindcast, synthetic_obs):
    """Rank-analog output cannot exceed the observed climatology — values
    are looked up from obs_sorted, so the range is bounded by obs."""
    result = africas2s.downscale(
        predictor_hindcast=synthetic_gcm_hindcast,
        obs=synthetic_obs,
        method="rank-analog",
        output_type="continuous",
        verbose=False,
    )
    obs_min = float(synthetic_obs.min())
    obs_max = float(synthetic_obs.max())
    assert float(result.min()) >= obs_min - 1e-6
    assert float(result.max()) <= obs_max + 1e-6


def test_downscale_rank_analog_with_explicit_forecast(
    synthetic_gcm_hindcast, synthetic_gcm_forecast, synthetic_obs,
):
    """Passing an explicit forecast (the Plan 4 calling pattern) bypasses
    the trailing-year auto-extraction and uses the full hindcast for fit."""
    result = africas2s.downscale(
        predictor_hindcast=synthetic_gcm_hindcast,
        obs=synthetic_obs,
        forecast=synthetic_gcm_forecast,
        method="rank-analog",
        output_type="continuous",
        verbose=False,
    )
    assert result.dims == ("member", "lat", "lon")
    assert len(result.member) == len(synthetic_gcm_forecast.member)


def test_downscale_rank_analog_vs_bcsd_produce_different_output(
    synthetic_gcm_hindcast, synthetic_obs,
):
    """A sanity check that rank-analog is doing something distinct from
    BCSD on the same input (so we know we have two real methods on the
    method-comparison panel, not one method aliased twice)."""
    rank_result = africas2s.downscale(
        predictor_hindcast=synthetic_gcm_hindcast,
        obs=synthetic_obs,
        method="rank-analog",
        output_type="continuous",
        verbose=False,
    )
    bcsd_result = africas2s.downscale(
        predictor_hindcast=synthetic_gcm_hindcast,
        obs=synthetic_obs,
        method="bcsd",
        output_type="continuous",
        verbose=False,
    )
    diff = float(np.abs(rank_result.mean("member") - bcsd_result.mean("member")).mean())
    assert diff > 1e-3, (
        f"rank-analog and bcsd produced near-identical mean fields (diff={diff})"
    )
