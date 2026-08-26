"""Tests for train / inference separation (§10.2, #27)."""
import numpy as np
import pytest
import xarray as xr


def _data(n_years=12, n_members=3):
    rng = np.random.default_rng(7)
    years = np.arange(2000, 2000 + n_years)
    members = np.arange(n_members)
    c_lat, c_lon = np.linspace(-4, 4, 5), np.linspace(30, 38, 5)
    f_lat, f_lon = np.linspace(-4, 4, 12), np.linspace(30, 38, 12)
    sig = np.sin(np.arange(n_years) * 0.5)[:, None, None]
    gcm = xr.DataArray(
        sig[:, None] * np.outer(np.sin(c_lat * 0.5), np.cos(c_lon * 0.3))[None, None]
        + rng.standard_normal((n_years, n_members, 5, 5)) * 0.3 + 5.0,
        dims=["year", "member", "lat", "lon"],
        coords={"year": years, "member": members, "lat": c_lat, "lon": c_lon},
    )
    obs = xr.DataArray(
        sig * np.outer(np.sin(f_lat * 0.5), np.cos(f_lon * 0.3))[None]
        + rng.standard_normal((n_years, 12, 12)) * 0.2 + 5.0,
        dims=["year", "lat", "lon"],
        coords={"year": years, "lat": f_lat, "lon": f_lon},
    )
    return gcm, obs


def test_train_returns_fitted_method():
    import africas2s.training as T
    gcm, obs = _data()
    m = T.train("cca", gcm, obs, n_modes=2, verbose=False)
    assert m.is_trained is True


def test_train_writes_loadable_checkpoint(tmp_path):
    import africas2s.training as T
    from africas2s.methods.cca import CCAMethod
    gcm, obs = _data()
    ckpt = tmp_path / "cca.pkl"
    T.train("cca", gcm, obs, save_to=ckpt, n_modes=2, verbose=False)
    assert ckpt.exists()
    m2 = CCAMethod()
    m2.load(ckpt)
    assert m2.is_trained is True


def test_train_then_downscale_weights_path_roundtrip(tmp_path):
    """The whole point of #27: train once, save, then inference-only via downscale."""
    import africas2s
    import africas2s.training as T
    gcm, obs = _data()
    forecast = gcm.isel(year=-1, drop=True)
    ckpt = tmp_path / "cca.pkl"
    m = T.train(
        "cca",
        gcm.isel(year=slice(None, -1)),
        obs.isel(year=slice(None, -1)),
        save_to=ckpt, n_modes=2, verbose=False,
    )
    expected = m.predict(forecast)
    result = africas2s.downscale(
        predictor_hindcast=forecast, method="cca",
        weights_path=str(ckpt), verbose=False,
    )
    np.testing.assert_array_equal(result.values, expected.values)


def test_downscale_requires_training_without_weights_raises():
    import africas2s
    from africas2s.registry import register_method
    from africas2s.methods.base import MethodBase

    @register_method("test_needs_training")
    class _NeedsTraining(MethodBase):
        requires_training = True

        def fit(self, hindcast, obs, **kwargs):
            self.fitted_ = True

        def predict(self, forecast, **kwargs):
            return forecast

    gcm, obs = _data()
    with pytest.raises(RuntimeError, match="requires separate training"):
        africas2s.downscale(gcm, obs, method="test_needs_training", verbose=False)


def test_downscale_requires_training_with_weights_ok(tmp_path):
    import africas2s
    import africas2s.training as T
    from africas2s.registry import register_method
    from africas2s.methods.base import MethodBase

    @register_method("test_needs_training2")
    class _NeedsTraining2(MethodBase):
        requires_training = True

        def fit(self, hindcast, obs, **kwargs):
            self.obs_coords_ = {"lat": obs.lat, "lon": obs.lon}

        def predict(self, forecast, **kwargs):
            if "year" in forecast.dims and forecast.sizes.get("year") == 1:
                forecast = forecast.isel(year=0, drop=True)
            return forecast.interp(
                lat=self.obs_coords_["lat"], lon=self.obs_coords_["lon"]
            )

    gcm, obs = _data()
    ckpt = tmp_path / "nt.pkl"
    T.train("test_needs_training2", gcm, obs, save_to=ckpt, verbose=False)
    forecast = gcm.isel(year=-1, drop=True)
    result = africas2s.downscale(
        predictor_hindcast=forecast, method="test_needs_training2",
        weights_path=str(ckpt), verbose=False,
    )
    assert result.dims == ("member", "lat", "lon")


def test_default_method_does_not_require_training():
    from africas2s.methods.base import MethodBase
    from africas2s.methods.cca import CCAMethod
    assert MethodBase.requires_training is False
    assert CCAMethod.requires_training is False
