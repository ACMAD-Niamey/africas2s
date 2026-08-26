"""Tests for Detrended Quantile Mapping (§19.2, #43)."""
import numpy as np
import pytest
import xarray as xr


def _same_grid_data(seed=2, n_years=30, n_members=3, gcm_loc=10.0, obs_loc=5.0,
                    gcm_trend_per_year=0.0):
    """GCM and obs on the SAME 3x3 grid; optional deterministic GCM trend."""
    rng = np.random.default_rng(seed)
    years = np.arange(1990, 1990 + n_years)
    members = np.arange(n_members)
    lat, lon = np.linspace(0, 4, 3), np.linspace(30, 34, 3)
    trend = gcm_trend_per_year * np.arange(n_years)
    gcm = xr.DataArray(
        gcm_loc + trend[:, None, None, None]
        + rng.normal(0, 0.5, (n_years, n_members, 3, 3)),
        dims=["year", "member", "lat", "lon"],
        coords={"year": years, "member": members, "lat": lat, "lon": lon},
    )
    obs = xr.DataArray(
        obs_loc + rng.normal(0, 0.5, (n_years, 3, 3)),
        dims=["year", "lat", "lon"],
        coords={"year": years, "lat": lat, "lon": lon},
    )
    return gcm, obs


def test_dqm_registry_lookup():
    from africas2s.registry import get_method
    from africas2s.methods.dqm import DetrendedQuantileMappingMethod
    assert get_method("dqm") is DetrendedQuantileMappingMethod


def test_dqm_fit_stores_trend_and_cdf_state(synthetic_gcm_hindcast, synthetic_obs):
    from africas2s.methods.dqm import DetrendedQuantileMappingMethod
    m = DetrendedQuantileMappingMethod()
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    assert hasattr(m, "gcm_slope_")
    assert hasattr(m, "t_forecast_")
    assert hasattr(m, "gcm_sorted_")
    assert m.gcm_slope_.shape == (len(synthetic_gcm_hindcast.lat),
                                  len(synthetic_gcm_hindcast.lon))


@pytest.mark.parametrize("variant", ["empirical", "parametric"])
def test_dqm_predict_shape_on_obs_grid(variant, synthetic_gcm_hindcast,
                                       synthetic_gcm_forecast, synthetic_obs):
    from africas2s.methods.dqm import DetrendedQuantileMappingMethod
    m = DetrendedQuantileMappingMethod(variant=variant)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    result = m.predict(synthetic_gcm_forecast)
    assert result.dims == ("member", "lat", "lon")
    assert len(result.lat) == len(synthetic_obs.lat)
    assert not np.all(np.isnan(result.values))


def test_dqm_reduces_bias_on_untrended_data():
    """With no trend, DQM still bias-corrects the GCM toward the obs climatology."""
    from africas2s.methods.dqm import DetrendedQuantileMappingMethod
    gcm, obs = _same_grid_data(gcm_loc=10.0, obs_loc=5.0, gcm_trend_per_year=0.0)
    m = DetrendedQuantileMappingMethod(variant="empirical")
    m.fit(gcm, obs)
    out = m.predict(gcm.isel(year=-1))
    assert abs(float(out.mean()) - 5.0) < 1.5  # pulled from ~10 toward obs mean


def test_dqm_preserves_gcm_trend_vs_plain_qm():
    """DQM re-adds the GCM trend, so on a strongly-trended GCM its forecast sits
    well above plain QM (which absorbs the trend into the CDF)."""
    from africas2s.methods.dqm import DetrendedQuantileMappingMethod
    from africas2s.methods.qm import QuantileMappingMethod
    gcm, obs = _same_grid_data(gcm_loc=5.0, obs_loc=5.0, gcm_trend_per_year=0.5)
    forecast = gcm.isel(year=-1)  # the highest-trend year

    qm = QuantileMappingMethod(variant="empirical"); qm.fit(gcm, obs)
    dqm = DetrendedQuantileMappingMethod(variant="empirical"); dqm.fit(gcm, obs)

    qm_out = float(qm.predict(forecast).mean())
    dqm_out = float(dqm.predict(forecast).mean())
    assert dqm_out > qm_out + 2.0


def test_dqm_downscale_integration():
    import africas2s
    gcm, obs = _same_grid_data(gcm_trend_per_year=0.3)
    cont = africas2s.downscale(gcm, obs, method="dqm", verbose=False)
    assert cont.dims == ("member", "lat", "lon")
    terc = africas2s.downscale(gcm, obs, method="dqm", output_type="tercile",
                               verbose=False)
    assert "tercile" in terc.dims
    np.testing.assert_allclose(terc.sum("tercile").values, 1.0, atol=1e-9)
