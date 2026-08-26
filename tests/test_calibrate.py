"""Unit tests for the calibrate() family (ereg + logit)."""
import numpy as np
import pytest
import xarray as xr

import africas2s as ds
from africas2s.registry import get_calibrator


def _gcm_obs(slope=2.0, intercept=5.0, n_years=20, seed=0, bias=3.0):
    rng = np.random.default_rng(seed)
    years = np.arange(2000, 2000 + n_years)
    mem, lat, lon = np.arange(6), np.arange(4), np.arange(5)
    truth = rng.normal(0, 1, (n_years, len(lat), len(lon)))
    h = truth[:, None] + rng.normal(0, 0.3, (n_years, len(mem), len(lat), len(lon))) + bias
    hcst = xr.DataArray(h, dims=["year", "member", "lat", "lon"],
                        coords={"year": years, "member": mem, "lat": lat, "lon": lon})
    obs = xr.DataArray(slope * truth + intercept + rng.normal(0, 0.2, truth.shape),
                       dims=["year", "lat", "lon"], coords={"year": years, "lat": lat, "lon": lon})
    return hcst, obs


def _sst_from_index(index, seed=0):
    """Build an SST field whose Nino3.4 box carries `index`."""
    rng = np.random.default_rng(seed)
    years = np.asarray(index.year.values)
    lat = np.arange(-10, 11, 5.0)
    lon = np.arange(180, 251, 5.0)
    base = rng.normal(27.0, 0.2, (len(years), len(lat), len(lon)))
    nino = ((lat[:, None] >= -5) & (lat[:, None] <= 5)
            & (lon[None, :] >= 190) & (lon[None, :] <= 240))
    data = base + np.asarray(index).reshape(-1, 1, 1) * nino
    return xr.DataArray(
        data,
        dims=["year", "lat", "lon"],
        coords={"year": years, "lat": lat, "lon": lon},
    )


def test_calibrators_registered():
    assert get_calibrator("ereg").__name__ == "_calibrate_ereg"
    assert get_calibrator("logit").__name__ == "_calibrate_logit"
    with pytest.raises(KeyError):
        get_calibrator("nope")


def test_calibrate_ereg_single_model_matches_method():
    """calibrate(method='ereg') on one model == that model's predict_tercile."""
    from africas2s.methods.ensemble_regression import EnsembleRegressionMethod
    hcst, obs = _gcm_obs()
    expected = (EnsembleRegressionMethod().fit(hcst, obs)
                .predict_tercile(hcst.sel(year=[2019]), obs))
    got = ds.calibrate({"m": (hcst, None)}, obs, method="ereg", forecast_year=2019)
    assert got.dims[0] == "tercile" and got.sizes["tercile"] == 3
    np.testing.assert_allclose(
        got.transpose("tercile", "lat", "lon").values,
        expected.transpose("tercile", "lat", "lon").values, atol=1e-9)


def test_calibrate_ereg_passes_threshold_source():
    from africas2s.methods.ensemble_regression import EnsembleRegressionMethod
    hcst, obs = _gcm_obs(seed=9)
    expected = (EnsembleRegressionMethod().fit(hcst, obs)
                .predict_tercile(
                    hcst.sel(year=[2019]), obs, threshold_source="fitted"))

    got = ds.calibrate(
        {"m": (hcst, None)},
        obs,
        method="ereg",
        forecast_year=2019,
        threshold_source="fitted",
    )

    np.testing.assert_allclose(
        got.transpose("tercile", "lat", "lon").values,
        expected.transpose("tercile", "lat", "lon").values, atol=1e-9)


def test_calibrate_ereg_averages_models():
    h1, obs = _gcm_obs(seed=0, bias=3.0)
    h2, _ = _gcm_obs(seed=1, bias=-2.0)
    out = ds.calibrate({"a": (h1, None), "b": (h2, None)}, obs,
                       method="ereg", forecast_year=2019)
    assert out.attrs["n_models"] == 2
    s = out.sum("tercile", skipna=False).values
    fin = np.isfinite(s)
    assert fin.any() and np.allclose(s[fin], 1.0, atol=1e-9)


def test_calibrate_ereg_default_forecast_year_is_last():
    hcst, obs = _gcm_obs()
    out = ds.calibrate({"m": (hcst, None)}, obs, method="ereg")
    assert out.attrs["forecast_year"] == 2019


def test_calibrate_ereg_uses_top_level_forecast_argument():
    hcst, obs = _gcm_obs(seed=2)
    fcst_low = hcst.sel(year=[2019]) * 0.0
    fcst_high = hcst.sel(year=[2019]) * 100.0

    low = ds.calibrate({"m": hcst}, obs, method="ereg", forecast={"m": fcst_low})
    high = ds.calibrate({"m": hcst}, obs, method="ereg", forecast={"m": fcst_high})

    assert low.attrs["forecast_year"] == 2019
    assert high.attrs["forecast_year"] == 2019
    assert not np.allclose(low.values, high.values, equal_nan=True)


def test_calibrate_ereg_infers_year_from_provided_forecast():
    hcst, obs = _gcm_obs()
    fcst = hcst.sel(year=[2019]).assign_coords(year=[2020])
    out = ds.calibrate({"m": (hcst, fcst)}, obs, method="ereg")
    assert out.attrs["forecast_year"] == 2020


def test_calibrate_ereg_missing_forecast_year_raises_clear_error():
    hcst, obs = _gcm_obs()
    fcst = hcst.sel(year=[2019]).assign_coords(year=[2020])
    with pytest.raises(ValueError, match="forecast_year=2019"):
        ds.calibrate({"m": (hcst, fcst)}, obs, method="ereg", forecast_year=2019)


def test_calibrate_ereg_partial_hindcast_overlap_raises_clear_error():
    hcst, obs = _gcm_obs()
    with pytest.raises(ValueError, match="missing obs years"):
        ds.calibrate({"m": (hcst.isel(year=slice(1, None)), None)}, obs, method="ereg")


def test_calibrate_ereg_invalid_combine_rejected_for_single_model():
    hcst, obs = _gcm_obs()
    with pytest.raises(ValueError, match="unknown combine"):
        ds.calibrate({"m": (hcst, None)}, obs, method="ereg", combine="median")


def test_calibrate_ereg_accepts_common_lat_lon_dim_aliases():
    hcst, obs = _gcm_obs()
    hcst = hcst.rename(lat="latitude", lon="longitude")
    obs = obs.rename(lat="latitude", lon="longitude")

    out = ds.calibrate({"m": (hcst, None)}, obs, method="ereg", forecast_year=2019)

    assert out.dims == ("tercile", "latitude", "longitude")
    np.testing.assert_allclose(out.sum("tercile").values, 1.0, atol=1e-9)


def test_calibrate_logit_runs_and_normalizes():
    rng = np.random.default_rng(3)
    years = np.arange(2000, 2030)
    idx = xr.DataArray(np.linspace(-2, 2, 30), dims=["year"], coords={"year": years})
    obs = xr.DataArray(
        120 - 30 * np.linspace(-2, 2, 30)[:, None, None] + rng.normal(0, 12, (30, 3, 3)),
        dims=["year", "lat", "lon"], coords={"year": years, "lat": np.arange(3), "lon": np.arange(3)})
    out = ds.calibrate(idx, obs, method="logit", forecast=2.0)
    assert out.dims[0] == "tercile"
    s = out.sum("tercile", skipna=False).values
    fin = np.isfinite(s)
    assert fin.any() and np.allclose(s[fin], 1.0, atol=1e-9)
    assert out.attrs["method"] == "logit"


def test_calibrate_logit_rejects_year_misalignment():
    years = np.arange(2000, 2020)
    idx = xr.DataArray(np.linspace(-1, 1, 20), dims=["year"], coords={"year": years})
    obs = xr.DataArray(np.random.default_rng(0).normal(100, 10, (20, 2, 2)),
                       dims=["year", "lat", "lon"],
                       coords={"year": years + 1, "lat": [0, 1], "lon": [0, 1]})
    with pytest.raises(ValueError, match="index.year values must match"):
        ds.calibrate(idx, obs, method="logit", forecast=0.5)


def test_calibrate_logit_rejects_multi_value_forecast_index():
    years = np.arange(2000, 2020)
    idx = xr.DataArray(np.linspace(-1, 1, 20), dims=["year"], coords={"year": years})
    obs = xr.DataArray(np.random.default_rng(0).normal(100, 10, (20, 2, 2)),
                       dims=["year", "lat", "lon"],
                       coords={"year": years, "lat": [0, 1], "lon": [0, 1]})
    fcst = xr.DataArray([0.5, 1.0], dims=["year"], coords={"year": [2020, 2021]})
    with pytest.raises(ValueError, match="exactly one value"):
        ds.calibrate(idx, obs, method="logit", forecast=fcst)


def test_calibrate_logit_config_reduces_gridded_predictor():
    rng = np.random.default_rng(4)
    years = np.arange(2000, 2030)
    raw_index = xr.DataArray(
        np.linspace(-2, 2, 30),
        dims=["year"],
        coords={"year": years},
    )
    sst_hcst = _sst_from_index(raw_index)
    sst_fcst = _sst_from_index(
        xr.DataArray([2.5], dims=["year"], coords={"year": [2030]}),
        seed=1,
    )
    obs = xr.DataArray(
        120 - 25 * np.linspace(-2, 2, 30)[:, None, None]
        + rng.normal(0, 10, (30, 3, 3)),
        dims=["year", "lat", "lon"],
        coords={"year": years, "lat": np.arange(3), "lon": np.arange(3)},
    )
    index = ds.Index.custom(
        name="nino34_candidate",
        regions={"nino34": [-5, 5, 190, 240]},
        combine=lambda z: z["nino34"],
    )

    expected = ds.calibrate(
        index.reduce(sst_hcst),
        obs,
        method="logit",
        forecast=index.reduce(sst_fcst, climatology=sst_hcst),
    )
    got = ds.calibrate(
        predictor_hindcast=sst_hcst,
        obs=obs,
        predictor_forecast=sst_fcst,
        method=ds.LogitConfig(index=index),
    )

    assert got.attrs["method"] == "logit"
    np.testing.assert_allclose(got.values, expected.values, atol=1e-12)


def test_calibrate_logit_config_accepts_method_kwargs():
    years = np.arange(2000, 2030)
    raw_index = xr.DataArray(np.linspace(-2, 2, 30), dims=["year"], coords={"year": years})
    sst_hcst = _sst_from_index(raw_index)
    sst_fcst = _sst_from_index(
        xr.DataArray([2.5], dims=["year"], coords={"year": [2030]}),
        seed=1,
    )
    obs = xr.DataArray(
        120 - 25 * np.linspace(-2, 2, 30)[:, None, None]
        + np.random.default_rng(4).normal(0, 10, (30, 3, 3)),
        dims=["year", "lat", "lon"],
        coords={"year": years, "lat": np.arange(3), "lon": np.arange(3)},
    )
    index = ds.Index.custom(
        name="nino34_candidate",
        regions={"nino34": [-5, 5, 190, 240]},
        combine=lambda z: z["nino34"],
    )

    expected = ds.calibrate(
        predictor_hindcast=sst_hcst,
        obs=obs,
        predictor_forecast=sst_fcst,
        method=ds.LogitConfig(index=index, regularization=1.0),
    )
    got = ds.calibrate(
        predictor_hindcast=sst_hcst,
        obs=obs,
        predictor_forecast=sst_fcst,
        method=ds.LogitConfig(index=index),
        regularization=1.0,
    )
    np.testing.assert_allclose(got.values, expected.values, atol=1e-12)


def test_calibrate_logit_config_rejects_multi_year_forecast_without_year():
    years = np.arange(2000, 2020)
    raw_index = xr.DataArray(np.linspace(-1, 1, 20), dims=["year"], coords={"year": years})
    sst_hcst = _sst_from_index(raw_index)
    sst_fcst = _sst_from_index(
        xr.DataArray([1.0, 2.0], dims=["year"], coords={"year": [2020, 2021]}),
        seed=2,
    )
    obs = xr.DataArray(np.random.default_rng(0).normal(100, 10, (20, 2, 2)),
                       dims=["year", "lat", "lon"],
                       coords={"year": years, "lat": [0, 1], "lon": [0, 1]})
    index = ds.Index.custom(
        name="nino34_candidate",
        regions={"nino34": [-5, 5, 190, 240]},
        combine=lambda z: z["nino34"],
    )
    with pytest.raises(ValueError, match="exactly one year"):
        ds.calibrate(
            predictor_hindcast=sst_hcst,
            predictor_forecast=sst_fcst,
            obs=obs,
            method=ds.LogitConfig(index=index),
        )


def test_calibrate_logit_config_multimodel_keys_must_match():
    years = np.arange(2000, 2020)
    raw_index = xr.DataArray(np.linspace(-1, 1, 20), dims=["year"], coords={"year": years})
    sst_hcst = _sst_from_index(raw_index)
    sst_fcst = _sst_from_index(
        xr.DataArray([1.0], dims=["year"], coords={"year": [2020]}),
        seed=2,
    )
    obs = xr.DataArray(np.random.default_rng(0).normal(100, 10, (20, 2, 2)),
                       dims=["year", "lat", "lon"],
                       coords={"year": years, "lat": [0, 1], "lon": [0, 1]})
    index = ds.Index.custom(
        name="nino34_candidate",
        regions={"nino34": [-5, 5, 190, 240]},
        combine=lambda z: z["nino34"],
    )
    with pytest.raises(ValueError, match="predictor_forecast keys"):
        ds.calibrate(
            predictor_hindcast={"a": sst_hcst, "b": sst_hcst},
            predictor_forecast={"a": sst_fcst},
            obs=obs,
            method=ds.LogitConfig(index=index),
        )


def test_calibrate_logit_multimodel_keys_must_match():
    years = np.arange(2000, 2020)
    idx = xr.DataArray(np.linspace(-1, 1, 20), dims=["year"], coords={"year": years})
    obs = xr.DataArray(np.random.default_rng(0).normal(100, 10, (20, 2, 2)),
                       dims=["year", "lat", "lon"],
                       coords={"year": years, "lat": [0, 1], "lon": [0, 1]})
    with pytest.raises(ValueError, match="forecast keys"):
        ds.calibrate({"a": idx, "b": idx}, obs, method="logit", forecast={"a": 0.5})


def test_calibrate_logit_requires_forecast():
    years = np.arange(2000, 2020)
    idx = xr.DataArray(np.linspace(-1, 1, 20), dims=["year"], coords={"year": years})
    obs = xr.DataArray(np.random.default_rng(0).normal(100, 10, (20, 2, 2)),
                       dims=["year", "lat", "lon"],
                       coords={"year": years, "lat": [0, 1], "lon": [0, 1]})
    with pytest.raises(ValueError, match="requires forecast"):
        ds.calibrate(idx, obs, method="logit")


def test_detrend_index_uses_explicit_forecast_year():
    """For a bare-scalar forecast index (no year coord), _detrend_index must
    detrend at the supplied forecast_year, not blindly at years[-1] + 1."""
    from africas2s.calibrate import _detrend_index

    years = list(range(2000, 2020))
    idx = xr.DataArray([2.0 * (y - 2000) for y in years],
                       dims="year", coords={"year": years})
    fc = xr.DataArray(50.0)  # bare scalar, no year coordinate

    _, f_2001 = _detrend_index(idx, fc, forecast_year=2001)
    _, f_2030 = _detrend_index(idx, fc, forecast_year=2030)
    assert not np.isclose(float(f_2001), float(f_2030))

    # With no forecast_year the documented fallback is years[-1] + 1 = 2020.
    _, f_fallback = _detrend_index(idx, fc)
    _, f_2020 = _detrend_index(idx, fc, forecast_year=2020)
    assert np.isclose(float(f_fallback), float(f_2020))


def test_calibrate_logit_detrend_respects_forecast_year():
    """calibrate(method='logit', detrend=True, forecast_year=...) must thread
    forecast_year into the detrend for a bare-scalar forecast index."""
    rng = np.random.default_rng(7)
    years = np.arange(2000, 2030)
    # Linear trend + genuine variability, so detrending leaves real signal for
    # the logit (a perfectly linear index would detrend to ~zero variance).
    noise = rng.normal(0.0, 1.0, 30)
    idx = xr.DataArray(0.1 * (years - 2015) + noise,
                       dims=["year"], coords={"year": years})
    # Rainfall responds to the detrended (de-meaned, de-trended) component.
    obs = xr.DataArray(
        120.0 - 40.0 * noise[:, None, None] + rng.normal(0, 8, (30, 2, 2)),
        dims=["year", "lat", "lon"],
        coords={"year": years, "lat": [0, 1], "lon": [0, 1]},
    )
    early = ds.calibrate(idx, obs, method="logit", forecast=1.0,
                         detrend=True, forecast_year=2005)
    late = ds.calibrate(idx, obs, method="logit", forecast=1.0,
                        detrend=True, forecast_year=2025)
    assert not np.allclose(early.values, late.values, equal_nan=True)


def test_calibrate_unknown_method():
    hcst, obs = _gcm_obs()
    with pytest.raises(KeyError):
        ds.calibrate({"m": (hcst, None)}, obs, method="bogus")
