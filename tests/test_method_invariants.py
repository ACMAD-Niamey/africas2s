"""Synthetic invariant tests for downscaling method audits.

These tests use tiny constructed arrays rather than observational data. The goal
is to check method-defining properties that should hold independent of CHIRPS,
reference packages, or forecast-system skill.
"""

import numpy as np
import xarray as xr


def _coords(n_years=5, n_members=2, n_lat=2, n_lon=2):
    return {
        "year": np.arange(2000, 2000 + n_years),
        "member": np.arange(n_members),
        "lat": np.linspace(-1.0, 1.0, n_lat),
        "lon": np.linspace(30.0, 32.0, n_lon),
    }


def _hindcast_from_year_fields(fields, members=2):
    fields = np.asarray(fields, dtype=float)
    c = _coords(n_years=fields.shape[0], n_members=members,
                n_lat=fields.shape[1], n_lon=fields.shape[2])
    data = np.repeat(fields[:, None, :, :], members, axis=1)
    return xr.DataArray(
        data,
        dims=["year", "member", "lat", "lon"],
        coords={k: c[k] for k in ["year", "member", "lat", "lon"]},
    )


def _obs_from_year_fields(fields, lat=None, lon=None):
    fields = np.asarray(fields, dtype=float)
    years = np.arange(2000, 2000 + fields.shape[0])
    if lat is None:
        lat = np.linspace(-1.0, 1.0, fields.shape[1])
    if lon is None:
        lon = np.linspace(30.0, 32.0, fields.shape[2])
    return xr.DataArray(
        fields,
        dims=["year", "lat", "lon"],
        coords={"year": years, "lat": lat, "lon": lon},
    )


def test_delta_zero_anomaly_returns_observed_climatology_exactly():
    from africas2s.methods.delta import DeltaScalingMethod

    gcm_fields = np.arange(5 * 2 * 2, dtype=float).reshape(5, 2, 2)
    obs_fields = 100.0 + 2.0 * gcm_fields
    hindcast = _hindcast_from_year_fields(gcm_fields)
    obs = _obs_from_year_fields(obs_fields)

    method = DeltaScalingMethod()
    method.fit(hindcast, obs)
    forecast = method.gcm_hist_clim_.expand_dims(member=[0])

    result = method.predict(forecast).isel(member=0)
    np.testing.assert_allclose(result, obs.mean("year"), rtol=0, atol=1e-12)


def test_climatology_predicts_grouped_observed_mean_for_each_member():
    from africas2s.methods.climatology import ClimatologyMethod

    obs_fields = np.array(
        [
            [[1.0, np.nan], [3.0, 4.0]],
            [[2.0, 8.0], [5.0, 6.0]],
            [[3.0, 10.0], [7.0, 8.0]],
        ]
    )
    obs = _obs_from_year_fields(obs_fields)
    hindcast = _hindcast_from_year_fields(np.nan_to_num(obs_fields), members=3)
    forecast = hindcast.isel(year=0, drop=True)

    method = ClimatologyMethod()
    method.fit(hindcast, obs)
    result = method.predict(forecast)

    expected = obs.mean("year").values
    assert result.sizes["member"] == 3
    for member in result.member:
        np.testing.assert_allclose(result.sel(member=member).values, expected)


def test_qm_empirical_identity_when_model_and_obs_distributions_match():
    from africas2s.methods.qm import QuantileMappingMethod

    fields = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 3.0], [4.0, 5.0]],
            [[3.0, 4.0], [5.0, 6.0]],
            [[4.0, 5.0], [6.0, 7.0]],
        ]
    )
    hindcast = _hindcast_from_year_fields(fields)
    obs = _obs_from_year_fields(fields)
    forecast = hindcast.isel(year=2, drop=True)

    method = QuantileMappingMethod(variant="empirical")
    method.fit(hindcast, obs)
    result = method.predict(forecast)

    np.testing.assert_allclose(result.values, forecast.values, rtol=0, atol=1e-12)


def test_qm_empirical_transfer_function_is_monotonic_per_cell():
    from africas2s.methods.qm import QuantileMappingMethod

    gcm_fields = np.array([np.full((2, 2), value) for value in [1.0, 2.0, 3.0, 4.0]])
    obs_fields = 10.0 + 3.0 * gcm_fields
    hindcast = _hindcast_from_year_fields(gcm_fields)
    obs = _obs_from_year_fields(obs_fields)

    method = QuantileMappingMethod(variant="empirical")
    method.fit(hindcast, obs)

    lows = _hindcast_from_year_fields(np.array([np.full((2, 2), 1.5)])).isel(year=0, drop=True)
    highs = _hindcast_from_year_fields(np.array([np.full((2, 2), 3.5)])).isel(year=0, drop=True)
    low_out = method.predict(lows)
    high_out = method.predict(highs)

    assert np.all(high_out.values > low_out.values)


def test_qm_empirical_clamps_forecasts_to_observed_training_support():
    from africas2s.methods.qm import QuantileMappingMethod

    gcm_fields = np.array([np.full((2, 2), value) for value in [1.0, 2.0, 3.0, 4.0]])
    obs_fields = np.array([np.full((2, 2), value) for value in [10.0, 20.0, 30.0, 40.0]])
    hindcast = _hindcast_from_year_fields(gcm_fields)
    obs = _obs_from_year_fields(obs_fields)

    method = QuantileMappingMethod(variant="empirical")
    method.fit(hindcast, obs)

    low_forecast = _hindcast_from_year_fields(np.array([np.full((2, 2), -999.0)])).isel(year=0, drop=True)
    high_forecast = _hindcast_from_year_fields(np.array([np.full((2, 2), 999.0)])).isel(year=0, drop=True)

    np.testing.assert_allclose(method.predict(low_forecast).values, 10.0)
    np.testing.assert_allclose(method.predict(high_forecast).values, 40.0)


def test_dqm_zero_trend_collapses_to_qm_for_same_convention():
    from africas2s.methods.dqm import DetrendedQuantileMappingMethod
    from africas2s.methods.qm import QuantileMappingMethod

    # Symmetric sequence around the centered time axis gives zero fitted slope.
    gcm_series = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    obs_series = gcm_series + 10.0
    gcm_fields = np.array([np.full((2, 2), value) for value in gcm_series])
    obs_fields = np.array([np.full((2, 2), value) for value in obs_series])
    hindcast = _hindcast_from_year_fields(gcm_fields)
    obs = _obs_from_year_fields(obs_fields)
    forecast = hindcast.isel(year=1, drop=True)

    qm = QuantileMappingMethod(variant="empirical")
    dqm = DetrendedQuantileMappingMethod(variant="empirical")
    qm.fit(hindcast, obs)
    dqm.fit(hindcast, obs)

    np.testing.assert_allclose(
        dqm.predict(forecast).values,
        qm.predict(forecast).values,
        rtol=0,
        atol=1e-12,
    )


def test_dqm_parametric_preserves_known_additive_model_trend():
    from africas2s.methods.dqm import DetrendedQuantileMappingMethod

    centered_time = np.arange(5, dtype=float) - 2.0
    slope = 2.5
    gcm_fields = np.array([np.full((2, 2), 5.0 + slope * t) for t in centered_time])
    obs_fields = np.array([np.full((2, 2), 10.0) for _ in centered_time])
    hindcast = _hindcast_from_year_fields(gcm_fields)
    obs = _obs_from_year_fields(obs_fields)
    forecast = hindcast.isel(year=-1, drop=True)

    method = DetrendedQuantileMappingMethod(variant="parametric")
    method.fit(hindcast, obs)
    result = method.predict(forecast)

    expected = 10.0 + slope * centered_time[-1]
    np.testing.assert_allclose(method.gcm_slope_, slope, atol=1e-12)
    np.testing.assert_allclose(result.values, expected, atol=1e-12)


def test_bcsd_same_grid_output_matches_bias_corrected_coarse_stage():
    from africas2s.methods.bcsd import BCSDMethod

    gcm_fields = np.array([np.full((2, 2), value) for value in [1.0, 2.0, 3.0, 4.0]])
    obs_fields = 20.0 + 2.0 * gcm_fields
    hindcast = _hindcast_from_year_fields(gcm_fields)
    obs = _obs_from_year_fields(obs_fields)
    forecast = _hindcast_from_year_fields(np.array([np.full((2, 2), 2.5)])).isel(year=0, drop=True)

    method = BCSDMethod()
    method.fit(hindcast, obs)
    result = method.predict(forecast)

    # Mirrors BCSD's declared empirical coarse-stage convention for n=4:
    # searchsorted([1,2,3,4], 2.5)/4 = 0.5 -> index 1.5 in sorted obs.
    expected_corrected = np.full((2, 2), 25.0)
    np.testing.assert_allclose(result.isel(member=0).values, expected_corrected)


def test_bcsd_zero_spatial_detail_fine_grid_matches_interpolated_correction():
    from africas2s.methods.bcsd import BCSDMethod

    gcm_fields = np.array([np.full((2, 2), value) for value in [1.0, 2.0, 3.0, 4.0]])
    hindcast = _hindcast_from_year_fields(gcm_fields)
    fine_lat = np.linspace(-1.0, 1.0, 3)
    fine_lon = np.linspace(30.0, 32.0, 3)
    obs_fields = np.array([np.full((3, 3), 20.0 + 2.0 * value) for value in [1.0, 2.0, 3.0, 4.0]])
    obs = _obs_from_year_fields(obs_fields, lat=fine_lat, lon=fine_lon)
    forecast = _hindcast_from_year_fields(np.array([np.full((2, 2), 2.5)])).isel(year=0, drop=True)

    method = BCSDMethod()
    method.fit(hindcast, obs)
    result = method.predict(forecast)

    np.testing.assert_allclose(result.isel(member=0).values, 25.0)


def test_rank_analog_known_rank_indexes_expected_sorted_observation():
    from africas2s.methods.rank_analog import RankAnalogMethod

    offsets = np.array([[0.0, 0.1], [0.2, 0.3]])
    gcm_fields = np.array([np.full((2, 2), value) for value in [0.0, 1.0, 2.0, 3.0]])
    obs_fields = np.array([10.0 * year + offsets for year in range(4)])
    hindcast = _hindcast_from_year_fields(gcm_fields)
    obs = _obs_from_year_fields(obs_fields)
    forecast = _hindcast_from_year_fields(np.array([np.full((2, 2), 2.0)])).isel(year=0, drop=True)

    method = RankAnalogMethod(closing_size=1, gaussian_sigma=0.0, upscale_factor=1)
    method.fit(hindcast, obs)
    result = method.predict(forecast)

    np.testing.assert_allclose(result.isel(member=0).values, obs_fields[2])


def test_rank_analog_upscale_and_crop_indexing_uses_expected_ranks():
    from africas2s.methods.rank_analog import RankAnalogMethod

    gcm_fields = np.array([np.full((2, 2), value) for value in [0.0, 1.0, 2.0, 3.0]])
    fine_offsets = np.arange(9, dtype=float).reshape(3, 3) / 100.0
    obs_fields = np.array([10.0 * year + fine_offsets for year in range(4)])
    hindcast = _hindcast_from_year_fields(gcm_fields)
    obs = _obs_from_year_fields(obs_fields, lat=np.linspace(-1.0, 1.0, 3), lon=np.linspace(30.0, 32.0, 3))

    forecast_values = np.array([[[0.0, 1.0], [2.0, 3.0]]])
    forecast = _hindcast_from_year_fields(forecast_values).isel(year=0, drop=True)

    method = RankAnalogMethod(closing_size=1, gaussian_sigma=0.0, upscale_factor=2)
    method.fit(hindcast, obs)
    result = method.predict(forecast).isel(member=0)

    expected_rank_field = np.array([[0, 0, 1], [0, 0, 1], [2, 2, 3]])
    expected = np.take_along_axis(obs_fields, expected_rank_field[None, :, :], axis=0)[0]
    np.testing.assert_allclose(result.values, expected)


def test_cca_reconstructs_low_rank_training_pattern_up_to_sign_conventions():
    from africas2s.methods.cca import CCAMethod

    t = np.linspace(-2.0, 2.0, 6)
    x_pattern = np.array([[1.0, -0.5], [0.25, 0.75]])
    y_pattern = np.array([[2.0, -1.0], [0.5, 1.5]])
    gcm_fields = np.array([5.0 + value * x_pattern for value in t])
    obs_fields = np.array([10.0 + value * y_pattern for value in t])
    hindcast = _hindcast_from_year_fields(gcm_fields, members=1)
    obs = _obs_from_year_fields(obs_fields)
    forecast = hindcast.isel(year=-1, drop=True)

    method = CCAMethod(x_eof_modes=1, y_eof_modes=1, cca_modes=1)
    method.fit(hindcast, obs)
    result = method.predict(forecast).isel(member=0)

    assert method.mu_.shape == (1,)
    np.testing.assert_allclose(method.mu_[0], 1.0, atol=1e-12)
    np.testing.assert_allclose(result.values, obs.isel(year=-1).values, atol=1e-10)


def test_cca_canonical_correlation_is_invariant_to_predictand_sign_flip():
    from africas2s.methods.cca import CCAMethod

    t = np.linspace(-2.0, 2.0, 6)
    x_pattern = np.array([[1.0, -0.5], [0.25, 0.75]])
    y_pattern = np.array([[2.0, -1.0], [0.5, 1.5]])
    gcm_fields = np.array([5.0 + value * x_pattern for value in t])
    obs_fields = np.array([10.0 + value * y_pattern for value in t])
    hindcast = _hindcast_from_year_fields(gcm_fields, members=1)

    method_pos = CCAMethod(x_eof_modes=1, y_eof_modes=1, cca_modes=1)
    method_neg = CCAMethod(x_eof_modes=1, y_eof_modes=1, cca_modes=1)
    method_pos.fit(hindcast, _obs_from_year_fields(obs_fields))
    method_neg.fit(hindcast, _obs_from_year_fields(-obs_fields))

    np.testing.assert_allclose(method_pos.mu_, method_neg.mu_, atol=1e-12)
