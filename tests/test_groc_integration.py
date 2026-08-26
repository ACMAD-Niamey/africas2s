"""End-to-end check for GROC through the public skill API.

Exercises the realistic call path (`skill(...)` with a tercile-probability
forecast) so composition regressions surface alongside unit-level changes.
"""
import math

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def perfect_tercile_fixture():
    np.random.seed(11)
    n_year, n_lat, n_lon = 20, 4, 4
    coords = {
        "year": np.arange(n_year),
        "lat": np.linspace(-1, 1, n_lat),
        "lon": np.linspace(0, 1, n_lon),
    }
    obs = xr.DataArray(
        np.random.randn(n_year, n_lat, n_lon),
        dims=["year", "lat", "lon"], coords=coords,
    )

    from africas2s.metrics.rpss import _cpt_boundaries
    obs_vals = obs.values
    t33, t67 = _cpt_boundaries(obs_vals)
    obs_cat = np.where(t33 > obs_vals, 0, np.where(t67 > obs_vals, 1, 2))
    fcst = np.zeros((n_year, 3, n_lat, n_lon))
    for k in range(3):
        fcst[:, k, :, :] = (obs_cat == k).astype(float)
    forecast = xr.DataArray(
        fcst, dims=["year", "tercile", "lat", "lon"],
        coords={**coords, "tercile": [0, 1, 2]},
    )
    return forecast, obs


def test_skill_includes_generalized_roc(perfect_tercile_fixture):
    """skill(metrics=['generalized_roc']) returns a finite [0,1] score; perfect → 1.0."""
    from africas2s.skill import skill

    forecast, obs = perfect_tercile_fixture
    report = skill(forecast, obs, metrics=["generalized_roc"])
    assert "generalized_roc" in report.scores
    score = report.scores["generalized_roc"]
    assert math.isfinite(score), f"expected finite GROC, got {score}"
    assert 0.0 <= score <= 1.0, f"GROC out of range: {score}"
    assert score == pytest.approx(1.0, abs=1e-9), f"expected perfect → 1.0, got {score}"
