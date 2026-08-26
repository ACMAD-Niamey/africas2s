"""§5 Flex forecast unit tests (issue #20).

Synthetic-data sanity checks: a Gaussian forecast centred on the deterministic
forecast with scale sqrt(PEV) should give exceedance ~0.5 at its own mean,
~0 far above, and percentile thresholds reference the *climatological*
distribution (from obs), not the forecast distribution.
"""
import numpy as np
import pytest
import xarray as xr

from africas2s.flex_forecast import flex_forecast, FlexForecastResult


def _grid(vals):
    return xr.DataArray(
        np.asarray(vals, dtype=float),
        dims=("lat", "lon"),
        coords={"lat": [0.0, 1.0], "lon": [10.0, 11.0]},
    )


def _obs(arr):
    arr = np.asarray(arr, dtype=float)
    n = arr.shape[0]
    return xr.DataArray(
        arr,
        dims=("year", "lat", "lon"),
        coords={"year": list(range(2000, 2000 + n)),
                "lat": [0.0, 1.0], "lon": [10.0, 11.0]},
    )


def test_exceedance_at_forecast_mean_is_half():
    det = _grid(np.full((2, 2), 5.0))
    pev = _grid(np.full((2, 2), 4.0))      # scale = sqrt(4) = 2
    obs = _obs(np.full((10, 2, 2), 5.0))
    r = flex_forecast(det, pev, obs, threshold=5.0, is_percentile=False)
    assert isinstance(r, FlexForecastResult)
    np.testing.assert_allclose(r.exceedance_prob.values, 0.5, atol=1e-6)


def test_threshold_far_above_gives_zero():
    det = _grid(np.full((2, 2), 5.0))
    pev = _grid(np.full((2, 2), 1.0))
    obs = _obs(np.zeros((10, 2, 2)))
    r = flex_forecast(det, pev, obs, threshold=100.0, is_percentile=False)
    assert float(r.exceedance_prob.max()) < 1e-6


def test_percentile_threshold_uses_climatology_median():
    # obs ~ N(5, 2); the 50th climatological percentile is ~5.
    rng = np.random.default_rng(0)
    arr = 5.0 + 2.0 * rng.standard_normal((400, 2, 2))
    obs = _obs(arr)
    det = _grid(np.full((2, 2), 5.0))
    pev = _grid(np.full((2, 2), 4.0))
    r = flex_forecast(det, pev, obs, threshold=0.5, is_percentile=True)
    np.testing.assert_allclose(r.transformed_threshold.values, 5.0, atol=0.3)
    np.testing.assert_allclose(r.exceedance_prob.values, 0.5, atol=0.05)


def test_zero_threshold_on_precip_is_near_one():
    det = _grid(np.full((2, 2), 5.0))
    pev = _grid(np.full((2, 2), 1.0))
    obs = _obs(np.abs(np.random.default_rng(1).standard_normal((20, 2, 2))) * 3)
    r = flex_forecast(det, pev, obs, threshold=0.0, is_percentile=False)
    np.testing.assert_allclose(r.exceedance_prob.values, 1.0, atol=1e-3)


def test_nan_propagates_per_cell():
    det = _grid([[5.0, np.nan], [5.0, 5.0]])
    pev = _grid(np.full((2, 2), 4.0))
    obs = _obs(np.full((10, 2, 2), 5.0))
    r = flex_forecast(det, pev, obs, threshold=5.0, is_percentile=False)
    assert np.isnan(r.exceedance_prob.values[0, 1])
    assert np.isfinite(r.exceedance_prob.values[0, 0])


def test_gamma_distribution_not_implemented():
    det = _grid(np.full((2, 2), 5.0))
    pev = _grid(np.full((2, 2), 1.0))
    obs = _obs(np.full((10, 2, 2), 5.0))
    with pytest.raises(NotImplementedError):
        flex_forecast(det, pev, obs, threshold=5.0, distribution="gamma")


def test_to_dict_round_trips():
    det = _grid(np.full((2, 2), 5.0))
    pev = _grid(np.full((2, 2), 4.0))
    obs = _obs(np.full((10, 2, 2), 5.0))
    d = flex_forecast(det, pev, obs, threshold=5.0, is_percentile=False).to_dict()
    assert set(d) >= {"exceedance_prob", "fcst_mu", "fcst_scale",
                      "climo_mu", "climo_scale", "transformed_threshold", "metadata"}
    assert d["exceedance_prob"]["dims"] == ["lat", "lon"]


def test_exports_available():
    import africas2s
    assert hasattr(africas2s, "flex_forecast")
    assert hasattr(africas2s, "FlexForecastResult")
