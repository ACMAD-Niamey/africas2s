import numpy as np
import pytest
import xarray as xr


# ===================================================================
# 1. Method base & registry
# ===================================================================

def test_method_base_is_abstract():
    from africas2s.methods.base import MethodBase
    with pytest.raises(TypeError):
        MethodBase()


def test_method_registry_lookup():
    from africas2s.registry import get_method
    from africas2s.methods.bcsd import BCSDMethod
    from africas2s.methods.cca import CCAMethod
    assert get_method("bcsd") is BCSDMethod
    assert get_method("cca") is CCAMethod


def test_method_registry_missing():
    from africas2s.registry import get_method
    with pytest.raises(KeyError):
        get_method("nonexistent")


def test_register_method_decorator():
    from africas2s.registry import register_method, get_method
    from africas2s.methods.base import MethodBase

    @register_method("test_dummy_m")
    class DummyMethod(MethodBase):
        def fit(self, hindcast, obs, **kwargs): pass
        def predict(self, forecast, **kwargs): return forecast

    assert get_method("test_dummy_m") is DummyMethod


# ===================================================================
# 2. BCSD method
# ===================================================================

def test_bcsd_fit_stores_state(synthetic_gcm_hindcast, synthetic_obs):
    from africas2s.methods.bcsd import BCSDMethod
    m = BCSDMethod()
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    assert hasattr(m, "obs_clim_")
    assert hasattr(m, "gcm_sorted_")
    assert hasattr(m, "obs_coarse_sorted_")


def test_bcsd_predict_shape(synthetic_gcm_hindcast, synthetic_gcm_forecast, synthetic_obs):
    from africas2s.methods.bcsd import BCSDMethod
    m = BCSDMethod()
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    result = m.predict(synthetic_gcm_forecast)
    assert result.dims == ("member", "lat", "lon")
    assert len(result.lat) == len(synthetic_obs.lat)
    assert len(result.lon) == len(synthetic_obs.lon)


def test_bcsd_predict_values_plausible(synthetic_gcm_hindcast, synthetic_gcm_forecast, synthetic_obs):
    from africas2s.methods.bcsd import BCSDMethod
    m = BCSDMethod()
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    result = m.predict(synthetic_gcm_forecast)
    assert not np.all(np.isnan(result.values))
    assert np.std(result.values) > 0  # not constant


# ===================================================================
# 3. CCA method
# ===================================================================

def test_cca_fit_stores_state(synthetic_gcm_hindcast, synthetic_obs):
    from africas2s.methods.cca import CCAMethod
    m = CCAMethod(n_modes=2)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    assert hasattr(m, "mu_")  # canonical correlations
    assert hasattr(m, "x_mean_")


def test_cca_predict_shape(synthetic_gcm_hindcast, synthetic_gcm_forecast, synthetic_obs):
    from africas2s.methods.cca import CCAMethod
    m = CCAMethod(n_modes=2)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    result = m.predict(synthetic_gcm_forecast)
    assert result.dims == ("member", "lat", "lon")
    assert len(result.lat) == len(synthetic_obs.lat)
    assert len(result.lon) == len(synthetic_obs.lon)


def test_cca_with_few_modes(synthetic_gcm_hindcast, synthetic_gcm_forecast, synthetic_obs):
    from africas2s.methods.cca import CCAMethod
    m = CCAMethod(n_modes=2)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    result = m.predict(synthetic_gcm_forecast)
    assert not np.all(np.isnan(result.values))


def test_cca_with_short_hindcast(synthetic_gcm_forecast, synthetic_obs):
    from africas2s.methods.cca import CCAMethod
    # Only 5 years
    short_years = np.arange(2000, 2005)
    gcm = xr.DataArray(
        np.random.randn(5, 3, 5, 5) + 5,
        dims=["year", "member", "lat", "lon"],
        coords={"year": short_years, "member": np.arange(3),
                "lat": np.linspace(-4, 4, 5), "lon": np.linspace(30, 38, 5)},
    )
    obs = synthetic_obs.sel(year=short_years)
    m = CCAMethod(n_modes=2)
    m.fit(gcm, obs)
    result = m.predict(synthetic_gcm_forecast)
    assert not np.all(np.isnan(result.values))


# ===================================================================
# 3b. CCA with EOF truncation
# ===================================================================

def test_cca_eof_fit_stores_pca_state(synthetic_gcm_hindcast, synthetic_obs):
    from africas2s.methods.cca import CCAMethod
    m = CCAMethod(n_modes=2, x_eof_modes=3, y_eof_modes=3)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    assert hasattr(m, "eofx_")
    assert hasattr(m, "eofy_")
    assert hasattr(m, "mu_")


def test_cca_eof_predict_shape(synthetic_gcm_hindcast, synthetic_gcm_forecast, synthetic_obs):
    from africas2s.methods.cca import CCAMethod
    m = CCAMethod(n_modes=2, x_eof_modes=3, y_eof_modes=3)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    result = m.predict(synthetic_gcm_forecast)
    assert result.dims == ("member", "lat", "lon")
    assert len(result.lat) == len(synthetic_obs.lat)


def test_cca_eof_produces_nondegenerate_terciles(synthetic_gcm_hindcast, synthetic_obs):
    from africas2s.methods.cca import CCAMethod
    from africas2s.tercile import to_tercile
    m = CCAMethod(n_modes=2, x_eof_modes=3, y_eof_modes=3)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    forecast = synthetic_gcm_hindcast.isel(year=-1, drop=True)
    result = m.predict(forecast)
    tercile = to_tercile(result, synthetic_obs)
    p_normal = float(tercile.sel(tercile=1).mean())
    assert p_normal < 0.9, f"Degenerate: P(normal) = {p_normal:.1%}"


def test_cca_auto_eof_modes(synthetic_gcm_hindcast, synthetic_obs):
    from africas2s.methods.cca import CCAMethod
    m = CCAMethod(n_modes=2)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    assert hasattr(m, "eofx_")
    assert m.eofx_.shape[1] <= 10  # n_components


# ===================================================================
# 3c. CCA dual-grid predictor support (§1.1)
# ===================================================================

def test_cca_stores_predictor_coords(synthetic_gcm_hindcast, synthetic_obs):
    """Predictor lat/lon must be stored separately from predictand."""
    from africas2s.methods.cca import CCAMethod
    m = CCAMethod(n_modes=2)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    assert hasattr(m, "predictor_coords_")
    np.testing.assert_array_equal(
        m.predictor_coords_["lat"].values, synthetic_gcm_hindcast.lat.values
    )
    np.testing.assert_array_equal(
        m.predictor_coords_["lon"].values, synthetic_gcm_hindcast.lon.values
    )


def test_cca_stores_predictand_coords(synthetic_gcm_hindcast, synthetic_obs):
    """Predictand lat/lon must be stored separately from predictor."""
    from africas2s.methods.cca import CCAMethod
    m = CCAMethod(n_modes=2)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    assert hasattr(m, "predictand_coords_")
    np.testing.assert_array_equal(
        m.predictand_coords_["lat"].values, synthetic_obs.lat.values
    )
    np.testing.assert_array_equal(
        m.predictand_coords_["lon"].values, synthetic_obs.lon.values
    )


def test_cca_stores_distinct_predictor_and_predictand_shapes(synthetic_gcm_hindcast, synthetic_obs):
    """When grids differ, the two shapes must be stored independently."""
    from africas2s.methods.cca import CCAMethod
    m = CCAMethod(n_modes=2)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    assert m.predictor_shape_ == (
        len(synthetic_gcm_hindcast.lat),
        len(synthetic_gcm_hindcast.lon),
    )
    assert m.predictand_shape_ == (
        len(synthetic_obs.lat),
        len(synthetic_obs.lon),
    )
    assert m.predictor_shape_ != m.predictand_shape_


def test_cca_eofx_reconstructs_to_predictor_grid(synthetic_gcm_hindcast, synthetic_obs):
    """eofx_ + x_valid_ + predictor_shape_ must reconstruct a spatial map (§3.2 contract)."""
    from africas2s.methods.cca import CCAMethod
    m = CCAMethod(n_modes=2, x_eof_modes=3, y_eof_modes=3)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    n_predictor_pts = int(np.prod(m.predictor_shape_))
    assert m.x_valid_.shape == (n_predictor_pts,)
    mode0 = np.full(n_predictor_pts, np.nan)
    mode0[m.x_valid_] = m.eofx_[:, 0]
    spatial = mode0.reshape(m.predictor_shape_)
    assert spatial.shape == m.predictor_shape_


def test_cca_eofy_reconstructs_to_predictand_grid(synthetic_gcm_hindcast, synthetic_obs):
    """eofy_ + y_valid_ + predictand_shape_ must reconstruct a spatial map (§3.2 contract)."""
    from africas2s.methods.cca import CCAMethod
    m = CCAMethod(n_modes=2, x_eof_modes=3, y_eof_modes=3)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    n_predictand_pts = int(np.prod(m.predictand_shape_))
    assert m.y_valid_.shape == (n_predictand_pts,)
    mode0 = np.full(n_predictand_pts, np.nan)
    mode0[m.y_valid_] = m.eofy_[:, 0]
    spatial = mode0.reshape(m.predictand_shape_)
    assert spatial.shape == m.predictand_shape_


def test_cca_predicts_on_predictand_grid_when_grids_differ(
    synthetic_gcm_hindcast, synthetic_gcm_forecast, synthetic_obs
):
    """Predict output must use the predictand grid, not the predictor grid."""
    from africas2s.methods.cca import CCAMethod
    assert synthetic_gcm_hindcast.shape[2:] != synthetic_obs.shape[1:]
    m = CCAMethod(n_modes=2)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    result = m.predict(synthetic_gcm_forecast)
    np.testing.assert_array_equal(result.lat.values, synthetic_obs.lat.values)
    np.testing.assert_array_equal(result.lon.values, synthetic_obs.lon.values)


def _build_dual_grid_fixture(seed=0, n_years=25, signal_amp=2.0, noise_amp=0.3):
    """Synthetic SST→precip dual-grid fixture.

    Predictor: 'tropical Pacific' SST on a coarse 6x8 grid (lat ±10°, lon 180-240°).
    Predictand: 'East Africa' precip on a fine 12x12 grid (lat -5 to 15°, lon 30-50°).
    A shared yearly time signal modulates a dipole on each grid (different patterns),
    so CCA should recover the link.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2000, 2000 + n_years)
    members = np.arange(3)

    # Predictor grid (geographically disjoint from predictand)
    p_lat = np.linspace(-10, 10, 6)
    p_lon = np.linspace(180, 240, 8)
    # Predictand grid
    o_lat = np.linspace(-5, 15, 12)
    o_lon = np.linspace(30, 50, 12)

    # Shared time signal
    t = np.arange(n_years)
    time_signal = np.sin(2 * np.pi * t / 5.0)

    # Planted spatial patterns (different on each grid)
    p_pattern = np.outer(np.sin(np.deg2rad(p_lat) * 3), np.cos(np.deg2rad(p_lon) * 2))
    o_pattern = np.outer(np.cos(np.deg2rad(o_lat) * 2), np.sin(np.deg2rad(o_lon) * 4))

    p_signal = signal_amp * time_signal[:, None, None] * p_pattern[None, :, :]
    o_signal = signal_amp * time_signal[:, None, None] * o_pattern[None, :, :]

    p_noise = rng.standard_normal((n_years, len(members), len(p_lat), len(p_lon))) * noise_amp
    o_noise = rng.standard_normal((n_years, len(o_lat), len(o_lon))) * noise_amp

    predictor = xr.DataArray(
        p_signal[:, None, :, :] + p_noise + 290.0,
        dims=["year", "member", "lat", "lon"],
        coords={"year": years, "member": members, "lat": p_lat, "lon": p_lon},
    )
    predictand = xr.DataArray(
        o_signal + o_noise + 5.0,
        dims=["year", "lat", "lon"],
        coords={"year": years, "lat": o_lat, "lon": o_lon},
    )
    return predictor, predictand, o_pattern


def test_cca_dual_grid_integration_end_to_end():
    """Integration: full dual-grid call via the public downscale() API."""
    import africas2s
    predictor, predictand, _ = _build_dual_grid_fixture()
    forecast = predictor.isel(year=-1, drop=True)
    train_predictor = predictor.isel(year=slice(None, -1))
    train_predictand = predictand.isel(year=slice(None, -1))

    result = africas2s.downscale(
        train_predictor, train_predictand, method="cca", forecast=forecast
    )
    assert result.dims == ("member", "lat", "lon")
    assert len(result.lat) == len(predictand.lat)
    assert len(result.lon) == len(predictand.lon)
    np.testing.assert_array_equal(result.lat.values, predictand.lat.values)
    np.testing.assert_array_equal(result.lon.values, predictand.lon.values)
    assert not np.all(np.isnan(result.values))


def test_cca_dual_grid_recovers_planted_signal():
    """Integration: CCA on dual grids recovers a known planted relationship."""
    from africas2s.methods.cca import CCAMethod
    predictor, predictand, o_pattern = _build_dual_grid_fixture()

    train_predictor = predictor.isel(year=slice(None, -1))
    train_predictand = predictand.isel(year=slice(None, -1))
    forecast = predictor.isel(year=-1, drop=True)
    truth = predictand.isel(year=-1)

    m = CCAMethod(n_modes=2)
    m.fit(train_predictor, train_predictand)

    # CCA should pick up the shared time signal — first canonical correlation high.
    assert m.mu_[0] > 0.7, f"first canonical correlation too low: {m.mu_[0]:.3f}"

    # Predict held-out year and check predicted anomaly pattern correlates
    # with the truth anomaly pattern.
    pred = m.predict(forecast).mean("member")
    pred_anom = (pred - pred.mean()).values.ravel()
    truth_anom = (truth - truth.mean()).values.ravel()
    corr = np.corrcoef(pred_anom, truth_anom)[0, 1]
    assert corr > 0.5, f"predicted vs truth pattern correlation too low: {corr:.3f}"


def test_cca_predict_rejects_forecast_with_wrong_grid_shape(synthetic_gcm_hindcast, synthetic_obs):
    """Calling predict() with a forecast on a different grid than the
    training predictor used to crash with a cryptic boolean-indexing
    IndexError. Now raises a clear ValueError naming both shapes."""
    from africas2s.methods.cca import CCAMethod
    m = CCAMethod(n_modes=2)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)

    # Training was 5x5; build a forecast on a 3x3 grid.
    bad_forecast = xr.DataArray(
        np.random.randn(3, 3, 3),
        dims=["member", "lat", "lon"],
        coords={"member": [0, 1, 2],
                "lat": np.linspace(-4, 4, 3), "lon": np.linspace(30, 38, 3)},
    )
    with pytest.raises(ValueError, match="grid shape"):
        m.predict(bad_forecast)


def test_cca_same_grid_case_still_works():
    """Regression guard: predictor==predictand grid is just a special case."""
    from africas2s.methods.cca import CCAMethod
    rng = np.random.default_rng(0)
    years = np.arange(2000, 2010)
    members = np.arange(2)
    lat = np.linspace(-4, 4, 6)
    lon = np.linspace(30, 38, 6)
    signal = np.sin(np.arange(len(years)) * 0.5)[:, None, None]
    spatial = np.outer(np.sin(lat), np.cos(lon))[None, :, :]
    gcm = xr.DataArray(
        signal[:, None, :, :] * spatial[:, None, :, :]
        + rng.standard_normal((len(years), len(members), len(lat), len(lon))) * 0.3
        + 5.0,
        dims=["year", "member", "lat", "lon"],
        coords={"year": years, "member": members, "lat": lat, "lon": lon},
    )
    obs = xr.DataArray(
        signal * spatial + rng.standard_normal((len(years), len(lat), len(lon))) * 0.2 + 5.0,
        dims=["year", "lat", "lon"],
        coords={"year": years, "lat": lat, "lon": lon},
    )
    forecast = gcm.isel(year=-1, drop=True)
    m = CCAMethod(n_modes=2)
    m.fit(gcm, obs)
    result = m.predict(forecast)
    assert result.shape == (len(members), len(lat), len(lon))
    assert m.predictor_shape_ == m.predictand_shape_ == (len(lat), len(lon))
    assert not np.any(np.isnan(result.values))


# ===================================================================
# 5. Rank-analog method
# ===================================================================

def test_rank_analog_registry_lookup():
    """The rank-analog method registers under the canonical name."""
    from africas2s.registry import get_method
    from africas2s.methods.rank_analog import RankAnalogMethod
    assert get_method("rank-analog") is RankAnalogMethod


def test_rank_analog_fit_stores_state(synthetic_gcm_hindcast, synthetic_obs):
    from africas2s.methods.rank_analog import RankAnalogMethod
    m = RankAnalogMethod()
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    assert hasattr(m, "obs_sorted_")
    assert hasattr(m, "hindcast_mean_")
    assert hasattr(m, "obs_coords_")
    assert hasattr(m, "gcm_coords_")
    assert hasattr(m, "upscale_factor_")
    assert hasattr(m, "n_years_")


def test_rank_analog_fit_obs_sorted_is_ascending_per_cell(synthetic_gcm_hindcast, synthetic_obs):
    """obs_sorted_[k, i, j] must be non-decreasing in k for every cell."""
    import numpy as np
    from africas2s.methods.rank_analog import RankAnalogMethod
    m = RankAnalogMethod()
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    sorted_vals = m.obs_sorted_.values
    diffs = np.diff(sorted_vals, axis=0)
    assert (diffs >= 0).all(), "obs_sorted_ is not ascending along the year axis at every cell"


def test_rank_analog_fit_hindcast_mean_collapses_member(synthetic_gcm_hindcast, synthetic_obs):
    """fit() reduces ensemble members to their mean per (year, lat, lon)."""
    from africas2s.methods.rank_analog import RankAnalogMethod
    m = RankAnalogMethod()
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    assert "member" not in m.hindcast_mean_.dims
    assert m.hindcast_mean_.dims == ("year", "lat", "lon")


def test_rank_analog_fit_auto_upscale_factor(synthetic_gcm_hindcast, synthetic_obs):
    """Auto-derived upscale_factor_ is a positive integer matching the grid ratio."""
    from africas2s.methods.rank_analog import RankAnalogMethod
    m = RankAnalogMethod()
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    expected = max(
        round(len(synthetic_obs.lat) / len(synthetic_gcm_hindcast.lat)),
        round(len(synthetic_obs.lon) / len(synthetic_gcm_hindcast.lon)),
        1,
    )
    assert isinstance(m.upscale_factor_, int)
    assert m.upscale_factor_ == expected


def test_rank_analog_fit_explicit_upscale_factor_honored(synthetic_gcm_hindcast, synthetic_obs):
    """When upscale_factor is passed to __init__, fit() preserves it verbatim."""
    from africas2s.methods.rank_analog import RankAnalogMethod
    m = RankAnalogMethod(upscale_factor=7)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    assert m.upscale_factor_ == 7


def test_rank_analog_predict_shape(synthetic_gcm_hindcast, synthetic_gcm_forecast, synthetic_obs):
    """predict() returns (member, lat, lon) on the obs grid."""
    from africas2s.methods.rank_analog import RankAnalogMethod
    m = RankAnalogMethod()
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    result = m.predict(synthetic_gcm_forecast)
    assert result.dims == ("member", "lat", "lon")
    assert len(result.lat) == len(synthetic_obs.lat)
    assert len(result.lon) == len(synthetic_obs.lon)
    assert len(result.member) == len(synthetic_gcm_forecast.member)


def test_rank_analog_predict_values_bounded_by_obs(
    synthetic_gcm_hindcast, synthetic_gcm_forecast, synthetic_obs,
):
    """Output values fall inside the range of the obs climatology
    (rank-analog can only emit observed values)."""
    import numpy as np
    from africas2s.methods.rank_analog import RankAnalogMethod
    m = RankAnalogMethod()
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    result = m.predict(synthetic_gcm_forecast)
    obs_min = float(synthetic_obs.min())
    obs_max = float(synthetic_obs.max())
    assert float(result.min()) >= obs_min - 1e-6
    assert float(result.max()) <= obs_max + 1e-6


def test_rank_analog_predict_member_coord_preserved(
    synthetic_gcm_hindcast, synthetic_gcm_forecast, synthetic_obs,
):
    """Member coordinate values pass through unchanged."""
    import numpy as np
    from africas2s.methods.rank_analog import RankAnalogMethod
    m = RankAnalogMethod()
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    result = m.predict(synthetic_gcm_forecast)
    np.testing.assert_array_equal(
        result.member.values, synthetic_gcm_forecast.member.values
    )


def test_rank_analog_predict_high_forecast_yields_high_obs(
    synthetic_gcm_hindcast, synthetic_obs,
):
    """A forecast far above the hindcast climatology should map to the top
    of the obs climatology at every cell."""
    import numpy as np
    import xarray as xr
    from africas2s.methods.rank_analog import RankAnalogMethod

    m = RankAnalogMethod()
    m.fit(synthetic_gcm_hindcast, synthetic_obs)

    high_value = float(synthetic_gcm_hindcast.max()) + 100.0
    forecast = xr.full_like(
        synthetic_gcm_hindcast.isel(year=0, drop=True), high_value
    )
    result = m.predict(forecast)

    obs_top = synthetic_obs.max("year")
    np.testing.assert_array_less(
        (obs_top.values - result.mean("member").values),
        ((obs_top.values - synthetic_obs.min("year").values) /
         max(synthetic_obs.sizes["year"] - 1, 1)) + 1e-6,
    )


def test_rank_analog_predict_squeezes_singleton_year(
    synthetic_gcm_hindcast, synthetic_gcm_forecast, synthetic_obs,
):
    """If the forecast still carries a singleton year dim (CV-loop pattern),
    predict() squeezes it cleanly."""
    from africas2s.methods.rank_analog import RankAnalogMethod
    m = RankAnalogMethod()
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    fc_with_year = synthetic_gcm_forecast.expand_dims(year=[2030])
    result = m.predict(fc_with_year)
    assert result.dims == ("member", "lat", "lon")


# ===================================================================
# 6. Climatology baseline method
# ===================================================================

def test_climatology_registry_lookup():
    """The climatology method registers under the canonical name."""
    from africas2s.registry import get_method
    from africas2s.methods.climatology import ClimatologyMethod
    assert get_method("climatology") is ClimatologyMethod


def test_climatology_fit_stores_state(synthetic_gcm_hindcast, synthetic_obs):
    from africas2s.methods.climatology import ClimatologyMethod
    m = ClimatologyMethod()
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    assert hasattr(m, "climatology_")
    assert hasattr(m, "obs_coords_")


def test_climatology_fit_climatology_is_obs_mean(synthetic_gcm_hindcast, synthetic_obs):
    """climatology_ must equal the per-cell mean of obs along the year axis."""
    import numpy as np
    from africas2s.methods.climatology import ClimatologyMethod
    m = ClimatologyMethod()
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    expected = synthetic_obs.mean("year").values
    np.testing.assert_allclose(m.climatology_.values, expected)


def test_climatology_fit_climatology_is_on_obs_grid(synthetic_gcm_hindcast, synthetic_obs):
    """climatology_ has the obs lat/lon shape, no member, no year."""
    from africas2s.methods.climatology import ClimatologyMethod
    m = ClimatologyMethod()
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    assert m.climatology_.dims == ("lat", "lon")
    assert len(m.climatology_.lat) == len(synthetic_obs.lat)
    assert len(m.climatology_.lon) == len(synthetic_obs.lon)


def test_climatology_predict_shape(synthetic_gcm_hindcast, synthetic_gcm_forecast, synthetic_obs):
    """predict() returns (member, lat, lon) on the obs grid."""
    from africas2s.methods.climatology import ClimatologyMethod
    m = ClimatologyMethod()
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    result = m.predict(synthetic_gcm_forecast)
    assert result.dims == ("member", "lat", "lon")
    assert len(result.lat) == len(synthetic_obs.lat)
    assert len(result.lon) == len(synthetic_obs.lon)
    assert len(result.member) == len(synthetic_gcm_forecast.member)


def test_climatology_predict_is_constant_across_members(
    synthetic_gcm_hindcast, synthetic_gcm_forecast, synthetic_obs,
):
    """Every member must hold the same climatology field — no member-to-member variance."""
    import numpy as np
    from africas2s.methods.climatology import ClimatologyMethod
    m = ClimatologyMethod()
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    result = m.predict(synthetic_gcm_forecast)
    member0 = result.isel(member=0).values
    for m_idx in range(1, len(result.member)):
        np.testing.assert_array_equal(result.isel(member=m_idx).values, member0)


def test_climatology_predict_ignores_forecast_values(
    synthetic_gcm_hindcast, synthetic_obs,
):
    """Doubling the forecast values must not change predict()'s output."""
    import numpy as np
    from africas2s.methods.climatology import ClimatologyMethod
    m = ClimatologyMethod()
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    fcst_a = synthetic_gcm_hindcast.isel(year=0, drop=True)
    fcst_b = fcst_a * 2.0
    result_a = m.predict(fcst_a)
    result_b = m.predict(fcst_b)
    np.testing.assert_array_equal(result_a.values, result_b.values)


def test_climatology_predict_member_coord_preserved(
    synthetic_gcm_hindcast, synthetic_gcm_forecast, synthetic_obs,
):
    """Member coordinate values pass through unchanged."""
    import numpy as np
    from africas2s.methods.climatology import ClimatologyMethod
    m = ClimatologyMethod()
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    result = m.predict(synthetic_gcm_forecast)
    np.testing.assert_array_equal(
        result.member.values, synthetic_gcm_forecast.member.values
    )


def test_climatology_predict_squeezes_singleton_year(
    synthetic_gcm_hindcast, synthetic_gcm_forecast, synthetic_obs,
):
    """A forecast with a singleton year axis (CV-loop pattern) is handled cleanly."""
    from africas2s.methods.climatology import ClimatologyMethod
    m = ClimatologyMethod()
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    fc_with_year = synthetic_gcm_forecast.expand_dims(year=[2030])
    result = m.predict(fc_with_year)
    assert result.dims == ("member", "lat", "lon")


# ===================================================================
# 7. Checkpoint-aware MethodBase (#26): is_trained / save / load
# ===================================================================

import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402


def _fresh_methods():
    """Default-constructed instances of every picklable method."""
    from africas2s.methods.bcsd import BCSDMethod
    from africas2s.methods.cca import CCAMethod
    from africas2s.methods.climatology import ClimatologyMethod
    from africas2s.methods.rank_analog import RankAnalogMethod
    return [
        BCSDMethod(),
        CCAMethod(n_modes=2),
        ClimatologyMethod(),
        RankAnalogMethod(),
    ]


def _method_specs():
    """(Class, kwargs) for a fresh instance of each picklable method."""
    from africas2s.methods.bcsd import BCSDMethod
    from africas2s.methods.cca import CCAMethod
    from africas2s.methods.climatology import ClimatologyMethod
    from africas2s.methods.rank_analog import RankAnalogMethod
    return [
        (BCSDMethod, {}),
        (CCAMethod, {"n_modes": 2}),
        (ClimatologyMethod, {}),
        (RankAnalogMethod, {}),
    ]


@pytest.mark.parametrize("method", _fresh_methods(), ids=lambda m: type(m).__name__)
def test_is_trained_false_before_fit(method):
    assert method.is_trained is False


@pytest.mark.parametrize("method", _fresh_methods(), ids=lambda m: type(m).__name__)
def test_is_trained_true_after_fit(method, synthetic_gcm_hindcast, synthetic_obs):
    method.fit(synthetic_gcm_hindcast, synthetic_obs)
    assert method.is_trained is True


# --- save/load round-trip -------------------------------------------------

@pytest.mark.parametrize("cls,kwargs", _method_specs(), ids=lambda x: getattr(x, "__name__", ""))
def test_save_load_roundtrip_is_bit_identical(
    cls, kwargs, synthetic_gcm_hindcast, synthetic_gcm_forecast, synthetic_obs, tmp_path
):
    m1 = cls(**kwargs)
    m1.fit(synthetic_gcm_hindcast, synthetic_obs)
    expected = m1.predict(synthetic_gcm_forecast)

    ckpt = tmp_path / "ckpt.pkl"
    m1.save(ckpt)
    assert ckpt.exists()

    m2 = cls()  # fresh, default args — load() restores params + fitted state
    m2.load(ckpt)
    assert m2.is_trained is True
    got = m2.predict(synthetic_gcm_forecast)

    # assert_array_equal treats NaNs in matching positions as equal.
    np.testing.assert_array_equal(got.values, expected.values)


def test_load_returns_self_for_chaining(
    synthetic_gcm_hindcast, synthetic_gcm_forecast, synthetic_obs, tmp_path
):
    from africas2s.methods.cca import CCAMethod
    ckpt = tmp_path / "ckpt.pkl"
    m1 = CCAMethod(n_modes=2)
    m1.fit(synthetic_gcm_hindcast, synthetic_obs)
    m1.save(ckpt)
    m2 = CCAMethod().load(ckpt)  # chained
    assert isinstance(m2, CCAMethod)
    assert m2.is_trained is True


def test_save_unfitted_method_raises(tmp_path):
    from africas2s.methods.cca import CCAMethod
    m = CCAMethod(n_modes=2)
    with pytest.raises(RuntimeError, match="not fitted"):
        m.save(tmp_path / "x.pkl")


def test_save_accepts_str_path(synthetic_gcm_hindcast, synthetic_obs, tmp_path):
    from africas2s.methods.climatology import ClimatologyMethod
    m = ClimatologyMethod()
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    path_str = str(tmp_path / "ckpt.pkl")  # str, not Path
    m.save(path_str)
    ClimatologyMethod().load(path_str)  # must not raise


# --- cross-process checkpoint integration ---------------------------------

@pytest.mark.integration
def test_checkpoint_survives_fresh_process(
    synthetic_gcm_hindcast, synthetic_gcm_forecast, synthetic_obs, tmp_path
):
    """fit+save here; load+predict in a brand-new interpreter; assert identical."""
    from africas2s.methods.cca import CCAMethod

    ckpt = tmp_path / "model.pkl"
    fcst_nc = tmp_path / "forecast.nc"
    expected_nc = tmp_path / "expected.nc"

    m = CCAMethod(n_modes=2)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    expected = m.predict(synthetic_gcm_forecast)
    m.save(ckpt)

    synthetic_gcm_forecast.rename("forecast").to_netcdf(fcst_nc)
    expected.rename("expected").to_netcdf(expected_nc)

    child = tmp_path / "child.py"
    child.write_text(
        "import numpy as np\n"
        "import xarray as xr\n"
        "from africas2s.methods.cca import CCAMethod\n"
        f"forecast = xr.open_dataarray(r'{fcst_nc}')\n"
        f"expected = xr.open_dataarray(r'{expected_nc}')\n"
        "m = CCAMethod()\n"
        f"m.load(r'{ckpt}')\n"
        "assert m.is_trained\n"
        "got = m.predict(forecast)\n"
        "np.testing.assert_array_equal(got.values, expected.values)\n"
        "print('ROUNDTRIP OK')\n"
    )

    r = subprocess.run(
        [sys.executable, str(child)],
        capture_output=True, text=True, cwd=str(_Path(__file__).resolve().parents[1]),
    )
    assert r.returncode == 0, r.stderr
    assert "ROUNDTRIP OK" in r.stdout


# ===================================================================
# 8. ProbabilisticMethodBase (#29)
# ===================================================================

def _make_dummy_prob_cls():
    """A minimal ProbabilisticMethodBase whose predict_distribution returns an
    ensemble (the coarse forecast bilinearly regridded to the obs grid)."""
    from africas2s.methods.base import ProbabilisticMethodBase

    class _DummyProb(ProbabilisticMethodBase):
        def fit(self, hindcast, obs, **kwargs):
            self.obs_coords_ = {"lat": obs.lat, "lon": obs.lon}

        def predict_distribution(self, forecast, **kwargs):
            if "year" in forecast.dims and forecast.sizes.get("year") == 1:
                forecast = forecast.isel(year=0, drop=True)
            return forecast.interp(
                lat=self.obs_coords_["lat"], lon=self.obs_coords_["lon"],
                method="linear",
            )

    return _DummyProb


def test_probabilistic_base_is_abstract():
    """predict_distribution is abstract — the base can't be instantiated."""
    from africas2s.methods.base import ProbabilisticMethodBase
    with pytest.raises(TypeError):
        ProbabilisticMethodBase()


def test_probabilistic_predict_returns_ensemble_mean(
    synthetic_gcm_hindcast, synthetic_gcm_forecast, synthetic_obs
):
    """Default predict() collapses predict_distribution() to its ensemble mean."""
    m = _make_dummy_prob_cls()()
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    dist = m.predict_distribution(synthetic_gcm_forecast)
    det = m.predict(synthetic_gcm_forecast)
    assert "member" in dist.dims
    assert "member" not in det.dims
    np.testing.assert_allclose(det.values, dist.mean("member").values)


def test_probabilistic_is_trained_after_fit(synthetic_gcm_hindcast, synthetic_obs):
    """ProbabilisticMethodBase inherits is_trained from MethodBase."""
    m = _make_dummy_prob_cls()()
    assert m.is_trained is False
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    assert m.is_trained is True


def test_downscale_tercile_uses_counting_for_probabilistic(
    synthetic_gcm_hindcast, synthetic_obs
):
    """downscale(output_type='tercile') routes probabilistic methods through
    predict_distribution + member counting (no Gaussian fit-to-deterministic)."""
    import africas2s
    from africas2s.registry import register_method
    register_method("test_prob_counting")(_make_dummy_prob_cls())

    terc = africas2s.downscale(
        synthetic_gcm_hindcast, synthetic_obs,
        method="test_prob_counting", output_type="tercile", verbose=False,
    )
    assert "tercile" in terc.dims
    np.testing.assert_allclose(terc.sum("tercile").values, 1.0, atol=1e-9)
    assert float(terc.std()) > 0  # not degenerate
