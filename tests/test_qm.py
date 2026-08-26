"""Tests for Quantile Mapping (§19.1, #42)."""
import numpy as np
import pytest
import xarray as xr


def _same_grid_data(seed=1, n_years=30, n_members=4,
                    gcm_loc=10.0, gcm_scale=3.0, obs_loc=5.0, obs_scale=1.0):
    """GCM and obs on the SAME 3x3 grid, so the fine-interp is the identity."""
    rng = np.random.default_rng(seed)
    years = np.arange(2000, 2000 + n_years)
    members = np.arange(n_members)
    lat = np.linspace(0, 4, 3)
    lon = np.linspace(30, 34, 3)
    gcm = xr.DataArray(
        rng.normal(gcm_loc, gcm_scale, (n_years, n_members, 3, 3)),
        dims=["year", "member", "lat", "lon"],
        coords={"year": years, "member": members, "lat": lat, "lon": lon},
    )
    obs = xr.DataArray(
        rng.normal(obs_loc, obs_scale, (n_years, 3, 3)),
        dims=["year", "lat", "lon"],
        coords={"year": years, "lat": lat, "lon": lon},
    )
    return gcm, obs


def test_qm_registry_lookup():
    from africas2s.registry import get_method
    from africas2s.methods.qm import QuantileMappingMethod
    assert get_method("qm") is QuantileMappingMethod


def test_qm_invalid_variant_raises():
    from africas2s.methods.qm import QuantileMappingMethod
    with pytest.raises(ValueError, match="variant"):
        QuantileMappingMethod(variant="bogus")


def test_qm_empirical_fit_stores_sorted_state(synthetic_gcm_hindcast, synthetic_obs):
    from africas2s.methods.qm import QuantileMappingMethod
    m = QuantileMappingMethod()  # empirical is the default
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    assert hasattr(m, "gcm_sorted_")
    assert hasattr(m, "obs_sorted_")
    assert m.gcm_sorted_.shape[0] == len(synthetic_gcm_hindcast.year)


def test_qm_parametric_fit_stores_moments(synthetic_gcm_hindcast, synthetic_obs):
    from africas2s.methods.qm import QuantileMappingMethod
    m = QuantileMappingMethod(variant="parametric")
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    for attr in ("gcm_mean_", "gcm_std_", "obs_mean_", "obs_std_"):
        assert hasattr(m, attr)


@pytest.mark.parametrize("variant", ["empirical", "parametric"])
def test_qm_predict_shape_on_obs_grid(variant, synthetic_gcm_hindcast,
                                      synthetic_gcm_forecast, synthetic_obs):
    from africas2s.methods.qm import QuantileMappingMethod
    m = QuantileMappingMethod(variant=variant)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    result = m.predict(synthetic_gcm_forecast)
    assert result.dims == ("member", "lat", "lon")
    assert len(result.lat) == len(synthetic_obs.lat)
    assert len(result.lon) == len(synthetic_obs.lon)
    assert not np.all(np.isnan(result.values))


def test_qm_parametric_maps_clim_mean_to_obs_mean():
    """Parametric QM of the GCM climatological mean (z=0) returns obs clim mean."""
    from africas2s.methods.qm import QuantileMappingMethod
    gcm, obs = _same_grid_data()
    m = QuantileMappingMethod(variant="parametric")
    m.fit(gcm, obs)
    gcm_clim = gcm.mean(["year", "member"])
    fc = gcm_clim.expand_dims(member=[0])
    out = m.predict(fc).isel(member=0)
    np.testing.assert_allclose(out.values, obs.mean("year").values, rtol=1e-6, atol=1e-6)


def test_qm_empirical_output_bounded_by_obs():
    """Empirical QM clamps to the obs support — outputs stay within the obs range."""
    from africas2s.methods.qm import QuantileMappingMethod
    gcm, obs = _same_grid_data()
    m = QuantileMappingMethod(variant="empirical")
    m.fit(gcm, obs)
    fc = gcm.isel(year=-1) + 1000.0  # far above the GCM training range
    out = m.predict(fc)
    assert float(out.max()) <= float(obs.max()) + 1e-6
    assert float(out.min()) >= float(obs.min()) - 1e-6


def test_qm_empirical_reduces_bias():
    """Empirical QM pulls the biased GCM toward the obs climatology."""
    from africas2s.methods.qm import QuantileMappingMethod
    gcm, obs = _same_grid_data(gcm_loc=10.0, obs_loc=5.0)
    m = QuantileMappingMethod(variant="empirical")
    m.fit(gcm, obs)
    fc = gcm.isel(year=-1)
    out = m.predict(fc)
    raw_bias = abs(float(fc.mean()) - float(obs.mean()))
    qm_bias = abs(float(out.mean()) - float(obs.mean()))
    assert qm_bias < raw_bias


def test_qm_downscale_integration():
    """End-to-end via the public API, continuous and tercile output."""
    import africas2s
    gcm, obs = _same_grid_data()
    cont = africas2s.downscale(gcm, obs, method="qm", verbose=False)
    assert cont.dims == ("member", "lat", "lon")
    terc = africas2s.downscale(gcm, obs, method="qm", variant="parametric",
                               output_type="tercile", verbose=False)
    assert "tercile" in terc.dims
    np.testing.assert_allclose(terc.sum("tercile").values, 1.0, atol=1e-9)
