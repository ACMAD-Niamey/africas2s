"""Unit tests for the CCCma calibration engine's building blocks and the
``engine="cccma"`` knobs of ``calibrate(method="smoothed_regression")``. Reference
parity lives in ``test_cccma_parity.py``; these cover behaviour the fixtures do not
reach (errors, the single-season path, missing cells, custom filters)."""
import numpy as np
import pytest
import xarray as xr
from scipy.stats import gamma as gamma_dist, norm

import africas2s
from africas2s.methods import cccma_calibration as cc

SEASONS = [f"{m:02d}" for m in range(1, 13)]


def _cube(n_year=20, n_member=5, nlat=2, nlon=3, seed=0, positive=False):
    rng = np.random.default_rng(seed)
    ns = len(SEASONS)
    signal = rng.standard_normal((ns, n_year, nlat, nlon))
    members = signal[:, :, None] + 0.6 * rng.standard_normal((ns, n_year, n_member, nlat, nlon))
    obs = 0.7 * signal + 0.5 * rng.standard_normal((ns, n_year, nlat, nlon))
    if positive:
        members = 40 * np.exp(0.4 * members)
        obs = 45 * np.exp(0.4 * obs)
    coords = {"season": SEASONS, "year": np.arange(1997, 1997 + n_year), "member": np.arange(n_member),
              "lat": np.linspace(-1.5, -0.5, nlat), "lon": np.linspace(36.5, 38.5, nlon)}
    hind = xr.DataArray(members, dims=("season", "year", "member", "lat", "lon"), coords=coords)
    ob = xr.DataArray(obs, dims=("season", "year", "lat", "lon"),
                      coords={k: coords[k] for k in ("season", "year", "lat", "lon")})
    return hind, ob


# ── pieces ────────────────────────────────────────────────────────────────────

def test_gaussian_filter_weights_shape_and_norm():
    w = cc.gaussian_filter_weights(2)
    assert w.shape == (12,) and np.isclose(w.sum(), 1.0)
    assert w[0] == w.max()
    np.testing.assert_allclose(w[1:6], w[-1:-6:-1])          # cyclic symmetry
    assert w[6] > 0                                            # the antipodal month gets one weight
    assert np.array_equal(cc.gaussian_filter_weights(0), np.eye(12)[0])
    np.testing.assert_allclose(cc.gaussian_filter_weights(-1), np.full(12, 1 / 12))


def test_resolve_filter_custom_vector():
    custom = [0.8, 0.1] + [0.0] * 9 + [0.1]
    np.testing.assert_array_equal(cc.resolve_filter(custom), custom)
    with pytest.raises(ValueError, match="12 weights"):
        cc.resolve_filter([0.5, 0.5])


def test_adjusted_tercile_small_sample_and_large_sample():
    assert cc.c4_factor(24) == 0.9891927                       # the reference's C4 table, N = 24
    t, p_below, p_above = cc.adjusted_tercile(24)
    assert t == pytest.approx(0.4307273 / 0.9891927)
    assert p_below == pytest.approx(norm.cdf(-t)) and p_above == pytest.approx(norm.cdf(t))
    assert cc.adjusted_tercile(101)[0] == 0.4307273


def test_cyclic_filter_identity_and_uniform():
    da = xr.DataArray(np.random.default_rng(0).random((2, 3, 12)),
                      dims=("latitude", "longitude", "month"), coords={"month": SEASONS})
    cond = xr.DataArray(np.ones((2, 3), dtype=bool), dims=("latitude", "longitude"))
    np.testing.assert_array_equal(cc.cyclic_filter(da, cc.gaussian_filter_weights(0), "month", cond).values, da.values)
    uniform = cc.cyclic_filter(da, cc.gaussian_filter_weights(-1), "month", cond)
    np.testing.assert_allclose(uniform.values, np.broadcast_to(da.mean("month").values[..., None], da.shape))
    # condition False keeps the original value
    cond[0, 0] = False
    mixed = cc.cyclic_filter(da, cc.gaussian_filter_weights(1), "month", cond)
    np.testing.assert_array_equal(mixed.values[0, 0], da.values[0, 0])
    assert not np.array_equal(mixed.values[1, 1], da.values[1, 1])


def test_gamma_lmom_fit_recovers_parameters_and_flags_degenerate():
    x = gamma_dist.rvs(a=3.0, scale=2.0, size=20000, random_state=1)
    shape, scale = cc.gamma_lmom_fit(x)
    assert shape == pytest.approx(3.0, rel=0.05) and scale == pytest.approx(2.0, rel=0.05)
    assert all(np.isnan(cc.gamma_lmom_fit(np.array([1.0, 1.0, 1.0, 1.0]))))


def test_tercile_probabilities_sum_to_one_and_fall_back_to_climatology():
    ano = xr.DataArray([[0.5, -1.0, 0.0]], dims=("latitude", "longitude"))
    sde = xr.DataArray([[0.8, 1.2, 0.0]], dims=("latitude", "longitude"))
    p = cc.tercile_probabilities(ano, sde, 0.43)
    np.testing.assert_allclose(p.sum("tercile").values, 1.0)
    np.testing.assert_allclose(p.values[:, 0, 2], 1 / 3)       # zero spread -> climatology
    assert p.values[2, 0, 0] > p.values[0, 0, 0]               # positive anomaly favours above


def test_golden_finds_minimum_of_parabola():
    xmin = cc.golden(lambda x: (x - 0.3) ** 2, (-2.0, 0.5, 2.0), tol=1e-6)
    assert float(xmin) == pytest.approx(0.3, abs=1e-5)


# ── fit / predict behaviour ───────────────────────────────────────────────────

def test_fit_requires_twelve_seasons_or_none():
    hind, obs = _cube()
    with pytest.raises(ValueError, match="12 initialisation months"):
        cc.fit(hind.isel(season=slice(0, 3)), obs.isel(season=slice(0, 3)))
    with pytest.raises(ValueError, match="same years"):
        cc.fit(hind, obs.isel(year=slice(0, 10)))


def test_single_season_path_constant_only():
    hind, obs = _cube(n_year=15)
    fit = cc.fit(hind.sel(season="03", drop=True), obs.sel(season="03", drop=True))
    assert not fit.seasonal
    p = cc.hindcast_tercile(fit, "constant")
    assert p.dims == ("year", "tercile", "lat", "lon")
    np.testing.assert_allclose(p.sum("tercile").values, 1.0)
    with pytest.raises(ValueError, match="needs a season axis"):
        cc.hindcast_tercile(fit, "smoothed")
    fc = hind.sel(season="03", drop=True).isel(year=0, drop=True)
    q = cc.predict_tercile(fit, fc, "constant")
    assert q.dims == ("tercile", "lat", "lon") and np.isfinite(q.values).all()


def test_missing_cell_is_climatology_and_masked_without_poisoning_neighbours():
    """The reference gives a data-less cell climatological probabilities (its anomaly and
    spread fall back to 0) and flags it in SMSK; the deterministic field there is NaN."""
    hind, obs = _cube(positive=True)
    hind[:, :, :, 0, 0] = np.nan
    obs[:, :, 0, 0] = np.nan
    fit = cc.fit(hind, obs, gamma_transform=True)
    p = cc.hindcast_tercile(fit, "smoothed", year=2000)
    np.testing.assert_allclose(p.values[:, :, 0, 0], 1 / 3)
    assert np.isnan(fit.masks["SMSK"].values[0, 0]).all()
    assert np.isfinite(fit.masks["SMSK"].values[1, 1]).all()
    assert np.isfinite(p.values[:, :, 1, 1]).all()
    np.testing.assert_allclose(p.values[:, :, 1, 1].sum(axis=1), 1.0)
    det = cc.hindcast_deterministic(fit, "smoothed", year=2000)
    assert np.isnan(det.values[:, 0, 0]).all() and np.isfinite(det.values[:, 1, 1]).all()


def test_coefficient_validation():
    hind, obs = _cube()
    fit = cc.fit(hind, obs)
    with pytest.raises(ValueError, match="coefficient must be one of"):
        cc.hindcast_tercile(fit, "bogus")
    assert fit.coefficient("smoothed").dims == ("season", "lat", "lon")


def test_forecast_mode_equals_hindcast_mode_up_to_rounding():
    """Applying the fit to a hindcast year's members as an out-of-sample forecast must
    reproduce the in-sample probabilities (the two paths differ only in whether the
    anomaly is (X - XAVG).mean or X.mean - XAVG)."""
    hind, obs = _cube()
    fit = cc.fit(hind, obs)
    year = 2005
    ins = cc.hindcast_tercile(fit, "smoothed", year=year)
    out = cc.predict_tercile(fit, hind.sel(year=year, drop=True), "smoothed")
    np.testing.assert_allclose(out.values, ins.values, rtol=1e-10, atol=1e-12)
    det_in = cc.hindcast_deterministic(fit, "smoothed", year=year)
    det_out = cc.predict_deterministic(fit, hind.sel(year=year, drop=True), "smoothed")
    np.testing.assert_allclose(det_out.values, det_in.values, rtol=1e-10, atol=1e-12)


def test_stack_models_pads_members():
    a = xr.DataArray(np.ones((2, 3)), dims=("member", "lat"), coords={"member": [10, 11]})
    b = xr.DataArray(np.ones((4, 3)), dims=("member", "lat"), coords={"member": [0, 1, 2, 3]})
    st = cc.stack_models({"A": a, "B": b})
    assert st.dims == ("model", "member", "lat") and st.sizes["member"] == 4
    assert np.isnan(st.sel(model="A").isel(member=slice(2, 4)).values).all()
    assert np.isfinite(st.sel(model="B").values).all()


# ── calibrate() wrapper knobs ─────────────────────────────────────────────────

def test_calibrate_engine_dispatch_and_errors():
    hind, obs = _cube()
    with pytest.raises(ValueError, match="unknown engine"):
        africas2s.calibrate(hind, obs, method="smoothed_regression", engine="fortran")
    with pytest.raises(TypeError, match="unexpected keyword"):
        africas2s.calibrate(hind, obs, method="smoothed_regression", engine="cccma",
                            output_type="tercile", gaussian_width=2)
    with pytest.raises(ValueError, match="mutually exclusive"):
        africas2s.calibrate(hind, obs, method="smoothed_regression", engine="cccma",
                            output_type="tercile", forecast=hind.isel(year=0, drop=True),
                            forecast_year=2000)
    with pytest.raises(ValueError, match="not understood"):
        africas2s.calibrate(hind, obs, method="smoothed_regression", engine="cccma",
                            output_type="tercile", temporal_sigma="pooled")


def test_calibrate_cccma_variants_and_custom_filter():
    hind, obs = _cube()
    fc = hind.isel(year=-1, drop=True) + 0.3
    out = {}
    for sigma in (None, 2, "constant", [0.8, 0.1] + [0.0] * 9 + [0.1]):
        p = africas2s.calibrate(hind, obs, method="smoothed_regression", engine="cccma",
                                output_type="tercile", temporal_sigma=sigma, forecast=fc,
                                constrained=False)               # constrained is ignored
        assert p.dims == ("season", "tercile", "lat", "lon")
        np.testing.assert_allclose(p.sum("tercile").values, 1.0)
        out[str(sigma)] = p
    assert out["None"].attrs["coefficient"] == "independent"
    assert out["2"].attrs["coefficient"] == "smoothed"
    assert out["constant"].attrs["coefficient"] == "constant"
    assert not np.allclose(out["None"].values, out["constant"].values)
    # the wrapper's knobs reach the engine
    p2 = africas2s.calibrate(hind, obs, method="smoothed_regression", engine="cccma",
                             output_type="tercile", temporal_sigma=2, forecast=fc,
                             obs_variance_width=0, fcst_variance_width=0, golden_tolerance=1e-5)
    assert not np.array_equal(p2.values, out["2"].values)


def test_calibrate_cccma_multi_model_dict_matches_stacked_array():
    hind, obs = _cube(n_member=6, positive=True)
    h1 = hind.isel(member=slice(0, 4))
    h2 = hind.isel(member=slice(4, 6)) * 1.1
    f1 = h1.isel(year=-1, drop=True)
    f2 = h2.isel(year=-1, drop=True)
    via_dict = africas2s.calibrate({"A": (h1, f1), "B": (h2, f2)}, obs, method="smoothed_regression",
                                   engine="cccma", output_type="tercile", temporal_sigma=2,
                                   distribution="gamma")
    stacked = cc.stack_models({"A": h1, "B": h2})
    fit = cc.fit(stacked, obs, gamma_transform=True)
    manual = cc.predict_tercile(fit, cc.stack_models({"A": f1, "B": f2}), "smoothed")
    np.testing.assert_array_equal(via_dict.values, manual.values)
    assert via_dict.attrs["distribution"] == "gamma"


def test_calibrate_cccma_deterministic_hindcast_year_is_in_obs_units():
    hind, obs = _cube(positive=True)
    det = africas2s.calibrate(hind, obs, method="smoothed_regression", engine="cccma",
                              output_type="deterministic", temporal_sigma="constant",
                              distribution="gamma", forecast_year=2010)
    assert det.dims == ("season", "lat", "lon")
    assert np.isfinite(det.values).all() and (det.values > 0).all()
    clim = obs.mean("year")
    assert np.abs(det.values / clim.values - 1).mean() < 1.0
