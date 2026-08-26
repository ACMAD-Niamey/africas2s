"""Unit tests for the ensemble_regression (eReg) calibration engine.

eReg is the per-model engine behind ``calibrate(method="ereg")`` (see
test_calibrate.py for the public multi-model path); these tests exercise the
engine class directly.
"""
import numpy as np
import pytest
import xarray as xr

from africas2s.methods.ensemble_regression import EnsembleRegressionMethod


def _synthetic(slope=2.0, intercept=5.0, n_years=25, n_mem=6, n_lat=3, n_lon=4,
               obs_noise=0.2, mem_noise=0.3, seed=0):
    """obs = slope*truth + intercept + noise; hindcast members = truth + noise."""
    rng = np.random.default_rng(seed)
    years = np.arange(2000, 2000 + n_years)
    lat, lon = np.arange(n_lat), np.arange(n_lon)
    truth = rng.normal(0, 1, (n_years, n_lat, n_lon))
    h = (truth[:, None]
         + rng.normal(0, mem_noise, (n_years, n_mem, n_lat, n_lon)))
    hindcast = xr.DataArray(
        h, dims=["year", "member", "lat", "lon"],
        coords={"year": years, "member": np.arange(n_mem), "lat": lat, "lon": lon},
    )
    obs = xr.DataArray(
        slope * truth + intercept + rng.normal(0, obs_noise, truth.shape),
        dims=["year", "lat", "lon"], coords={"year": years, "lat": lat, "lon": lon},
    )
    return hindcast, obs


def test_recovers_calibration_coefficients():
    hindcast, obs = _synthetic(slope=2.0, intercept=5.0)
    m = EnsembleRegressionMethod()
    m.fit(hindcast, obs)
    assert np.nanmean(m.slope_) == pytest.approx(2.0, abs=0.15)
    assert np.nanmean(m.intercept_) == pytest.approx(5.0, abs=0.15)
    assert m.is_trained


def test_predict_applies_fit():
    hindcast, obs = _synthetic(slope=2.0, intercept=5.0, obs_noise=0.0, mem_noise=0.0)
    m = EnsembleRegressionMethod()
    m.fit(hindcast, obs)
    pred = m.predict(hindcast.isel(year=-1))  # (member, lat, lon)
    assert pred.dims == ("lat", "lon")
    np.testing.assert_allclose(pred.values, obs.isel(year=-1).values, atol=1e-6)


def test_predict_grid_mismatch_raises():
    hindcast, obs = _synthetic()
    m = EnsembleRegressionMethod()
    m.fit(hindcast, obs)
    with pytest.raises(ValueError, match="grid shape"):
        m.predict(hindcast.isel(year=-1, lat=slice(0, 2)))


def test_clip_negative():
    hindcast, obs = _synthetic(slope=1.0, intercept=-50.0, obs_noise=0.0, mem_noise=0.0)
    m = EnsembleRegressionMethod(clip_negative=True)
    m.fit(hindcast, obs)
    assert float(m.predict(hindcast.isel(year=-1)).min()) >= 0.0


def test_save_load_roundtrip(tmp_path):
    hindcast, obs = _synthetic()
    m = EnsembleRegressionMethod()
    m.fit(hindcast, obs)
    p = tmp_path / "ereg.pkl"
    m.save(p)
    m2 = EnsembleRegressionMethod().load(p)
    np.testing.assert_allclose(
        m2.predict(hindcast.isel(year=-1)).values,
        m.predict(hindcast.isel(year=-1)).values,
    )


def test_predict_tercile_sums_to_one_and_is_directional():
    hindcast, obs = _synthetic(slope=2.0, intercept=5.0)
    m = EnsembleRegressionMethod()
    m.fit(hindcast, obs)
    t = m.predict_tercile(hindcast.isel(year=[-1]), obs)
    assert t.dims == ("tercile", "lat", "lon")
    np.testing.assert_allclose(t.sum("tercile").values, 1.0, atol=1e-9)
    wet = hindcast.isel(year=[-1]) + 10.0
    t_wet = m.predict_tercile(wet, obs)
    assert float(t_wet.sel(tercile=0).mean()) < float(t.sel(tercile=0).mean())


def test_predict_tercile_can_use_fitted_hindcast_thresholds():
    hindcast, obs = _synthetic(obs_noise=5.0, mem_noise=0.0, n_lat=1, n_lon=1)
    m = EnsembleRegressionMethod()
    m.fit(hindcast, obs)

    fcst = hindcast.isel(year=[-1])
    obs_thresholds = m.predict_tercile(fcst, obs, threshold_source="obs")
    fitted_thresholds = m.predict_tercile(fcst, obs, threshold_source="fitted")

    assert not np.allclose(obs_thresholds, fitted_thresholds)
    assert m.fitted_hindcast_.dims == ("year", "lat", "lon")


def test_fitted_hindcast_is_lazy():
    """fit() must not eagerly build the fitted hindcast: it is unused on the
    default threshold_source='obs' path, so building it per fit/CV-fold is wasted
    work. It is computed on first access and then cached."""
    hindcast, obs = _synthetic()
    m = EnsembleRegressionMethod()
    m.fit(hindcast, obs)
    assert m.__dict__.get("_fitted_hindcast_cache") is None  # not built by fit()
    fitted = m.fitted_hindcast_                               # triggers compute
    assert fitted.dims == ("year", "lat", "lon")
    assert m.__dict__.get("_fitted_hindcast_cache") is not None  # cached


def test_fitted_hindcast_thresholds_ignore_clip_negative():
    """clip_negative is a deterministic-output floor; it must not distort the
    'fitted' tercile boundaries by compressing the lower tail at zero. The
    fitted-threshold hindcast is therefore the un-clipped calibrated hindcast."""
    n = 24
    years = np.arange(2000, 2000 + n)
    x = np.linspace(-3.0, 3.0, n)
    y = 2.0 * x + np.random.default_rng(0).normal(0, 0.1, n)  # calibrated fit spans <0
    hind = xr.DataArray(
        x[:, None, None, None], dims=["year", "member", "lat", "lon"],
        coords={"year": years, "member": [0], "lat": [0], "lon": [0]},
    )
    obs = xr.DataArray(y[:, None, None], dims=["year", "lat", "lon"],
                       coords={"year": years, "lat": [0], "lon": [0]})

    clipped = EnsembleRegressionMethod(clip_negative=True).fit(hind, obs)
    unclipped = EnsembleRegressionMethod(clip_negative=False).fit(hind, obs)

    # The natural calibrated hindcast goes negative; the fitted-threshold source
    # preserves that under clip_negative rather than flooring it to zero.
    assert float(clipped.fitted_hindcast_.min()) < 0.0
    np.testing.assert_allclose(
        clipped.fitted_hindcast_.values, unclipped.fitted_hindcast_.values, atol=1e-12)
    # The deterministic predict() still clips (unchanged behaviour).
    assert float(clipped.predict(hind.isel(year=0)).min()) >= 0.0


def test_predict_tercile_rejects_unknown_threshold_source():
    hindcast, obs = _synthetic()
    m = EnsembleRegressionMethod()
    m.fit(hindcast, obs)

    with pytest.raises(ValueError, match="threshold_source"):
        m.predict_tercile(hindcast.isel(year=[-1]), obs, threshold_source="unknown")


def test_predict_tercile_matches_prediction_error_variance_formula():
    """sigma^2 = pev * (1 + 1/n + (xf - xbar)^2 / Sxx) (Wilks 2006 eq 6.22),
    re-derived independently and checked against predict_tercile."""
    from scipy.stats import norm

    rng = np.random.default_rng(11)
    n = 20
    years = np.arange(2000, 2000 + n)
    x_pred = np.linspace(-2.0, 2.0, n)
    y = 2.0 * x_pred + 10.0 + rng.normal(0, 3.0, n)

    hind = xr.DataArray(  # single member → ensemble mean == x_pred
        x_pred[:, None, None, None], dims=["year", "member", "lat", "lon"],
        coords={"year": years, "member": [0], "lat": [0], "lon": [0]},
    )
    obs = xr.DataArray(y[:, None, None], dims=["year", "lat", "lon"],
                       coords={"year": years, "lat": [0], "lon": [0]})
    m = EnsembleRegressionMethod().fit(hind, obs)

    xf = 1.7
    fcst = xr.DataArray([[xf]], dims=["lat", "lon"], coords={"lat": [0], "lon": [0]})
    got = m.predict_tercile(fcst, obs)

    b, a = np.polyfit(x_pred, y, 1)
    pev = np.sum((y - (a + b * x_pred)) ** 2) / (n - 2)
    xbar = x_pred.mean()
    sxx = np.sum((x_pred - xbar) ** 2)
    lev = 1.0 / n + (xf - xbar) ** 2 / sxx
    sigma = np.sqrt(pev * (1.0 + lev))
    mu = a + b * xf
    t33, t67 = np.quantile(y, [1 / 3, 2 / 3])

    assert float(got.sel(tercile=0).values.reshape(-1)[0]) == pytest.approx(
        float(norm.cdf(t33, mu, sigma)), abs=1e-9)
    assert float(got.sel(tercile=2).values.reshape(-1)[0]) == pytest.approx(
        float(1.0 - norm.cdf(t67, mu, sigma)), abs=1e-9)


def test_predict_tercile_leverage_widens_variance_at_fixed_mean():
    """The leverage term strictly widens sigma away from the training mean."""
    from scipy.stats import norm

    n = 21
    years = np.arange(2000, 2000 + n)
    x_pred = np.linspace(-1.0, 1.0, n)
    rng = np.random.default_rng(7)
    y = x_pred + rng.normal(0, 0.5, n)
    hind = xr.DataArray(
        x_pred[:, None, None, None], dims=["year", "member", "lat", "lon"],
        coords={"year": years, "member": [0], "lat": [0], "lon": [0]},
    )
    obs = xr.DataArray(y[:, None, None], dims=["year", "lat", "lon"],
                       coords={"year": years, "lat": [0], "lon": [0]})
    m = EnsembleRegressionMethod().fit(hind, obs)

    xbar = float(m.x_mean_.ravel()[0])
    sxx = float(m.sxx_.ravel()[0])
    pev = float(m.pev_.ravel()[0])
    sig_mean = np.sqrt(pev * (1 + 1 / n))
    sig_far = np.sqrt(pev * (1 + 1 / n + (3.0 - xbar) ** 2 / sxx))
    assert sig_far > sig_mean

    t = m.predict_tercile(
        xr.DataArray([[3.0]], dims=["lat", "lon"], coords={"lat": [0], "lon": [0]}), obs)
    t33 = float(np.quantile(y, 1 / 3))
    mu_far = float(m.slope_.ravel()[0]) * 3.0 + float(m.intercept_.ravel()[0])
    assert float(t.sel(tercile=0).values.reshape(-1)[0]) == pytest.approx(
        float(norm.cdf(t33, mu_far, sig_far)), abs=1e-9)
