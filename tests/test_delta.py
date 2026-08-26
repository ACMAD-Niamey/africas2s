"""Tests for Delta scaling (§19.7, #48)."""
import numpy as np
import pytest
import xarray as xr


def _same_grid_data(seed=3, n_years=20, n_members=3, gcm_loc=12.0, obs_loc=5.0):
    """GCM and obs on the SAME 3x3 grid, so the fine-interp is the identity."""
    rng = np.random.default_rng(seed)
    years = np.arange(2000, 2000 + n_years)
    members = np.arange(n_members)
    lat, lon = np.linspace(0, 4, 3), np.linspace(30, 34, 3)
    gcm = xr.DataArray(
        rng.normal(gcm_loc, 2.0, (n_years, n_members, 3, 3)),
        dims=["year", "member", "lat", "lon"],
        coords={"year": years, "member": members, "lat": lat, "lon": lon},
    )
    obs = xr.DataArray(
        rng.normal(obs_loc, 1.0, (n_years, 3, 3)),
        dims=["year", "lat", "lon"],
        coords={"year": years, "lat": lat, "lon": lon},
    )
    return gcm, obs


def test_delta_registry_lookup():
    from africas2s.registry import get_method
    from africas2s.methods.delta import DeltaScalingMethod
    assert get_method("delta") is DeltaScalingMethod


def test_delta_fit_stores_climatologies(synthetic_gcm_hindcast, synthetic_obs):
    from africas2s.methods.delta import DeltaScalingMethod
    m = DeltaScalingMethod()
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    assert hasattr(m, "gcm_hist_clim_")
    assert hasattr(m, "obs_clim_")
    assert m.gcm_hist_clim_.dims == ("lat", "lon")
    assert m.obs_clim_.dims == ("lat", "lon")


def test_delta_predict_shape_on_obs_grid(synthetic_gcm_hindcast,
                                         synthetic_gcm_forecast, synthetic_obs):
    from africas2s.methods.delta import DeltaScalingMethod
    m = DeltaScalingMethod()
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    result = m.predict(synthetic_gcm_forecast)
    assert result.dims == ("member", "lat", "lon")
    assert len(result.lat) == len(synthetic_obs.lat)
    assert len(result.lon) == len(synthetic_obs.lon)
    assert not np.all(np.isnan(result.values))


def test_delta_zero_anomaly_returns_obs_clim():
    """Forecast == GCM historical climatology -> zero anomaly -> obs climatology."""
    from africas2s.methods.delta import DeltaScalingMethod
    gcm, obs = _same_grid_data()
    m = DeltaScalingMethod()
    m.fit(gcm, obs)
    gcm_hist_clim = gcm.mean(["member", "year"])
    fc = gcm_hist_clim.expand_dims(member=[0])
    out = m.predict(fc).isel(member=0)
    np.testing.assert_allclose(out.values, obs.mean("year").values, atol=1e-6)


def test_delta_applies_anomaly_exactly_on_same_grid():
    """On a shared grid (identity interp): result == obs_clim + (forecast - gcm_hist_clim)."""
    from africas2s.methods.delta import DeltaScalingMethod
    gcm, obs = _same_grid_data()
    m = DeltaScalingMethod()
    m.fit(gcm, obs)
    fc = gcm.isel(year=-1)  # (member, lat, lon)
    out = m.predict(fc)
    expected = (fc - gcm.mean(["member", "year"])) + obs.mean("year")
    np.testing.assert_allclose(out.values, expected.transpose("member", "lat", "lon").values,
                               atol=1e-6)


def test_delta_downscale_integration():
    import africas2s
    gcm, obs = _same_grid_data()
    cont = africas2s.downscale(gcm, obs, method="delta", verbose=False)
    assert cont.dims == ("member", "lat", "lon")
    terc = africas2s.downscale(gcm, obs, method="delta", output_type="tercile",
                               verbose=False)
    assert "tercile" in terc.dims
    np.testing.assert_allclose(terc.sum("tercile").values, 1.0, atol=1e-9)
