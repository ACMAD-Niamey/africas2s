"""Unit tests for the ICPAC-parity knobs on the eReg stream.

Each knob mirrors one piece of ICPAC's zStatFunctionsEnsBiasRegPrec.R /
zFcstPrec11LeadDriverEnsBiasReg.R; defaults must leave legacy behavior
untouched (covered by the existing calibrate/ereg tests plus the default
checks here).
"""
import numpy as np
import pytest
import xarray as xr
from scipy.stats import norm

from africas2s.methods.ensemble_regression import EnsembleRegressionMethod
from africas2s.calibrate import calibrate


def _data(n_years=20, n_mem=5, n_lat=2, n_lon=2, seed=0):
    rng = np.random.default_rng(seed)
    years = np.arange(1990, 1990 + n_years)
    signal = rng.normal(200.0, 40.0, (n_years, 1, 1, 1))
    hind = xr.DataArray(
        signal + rng.normal(0, 25.0, (n_years, n_mem, n_lat, n_lon)),
        dims=["year", "member", "lat", "lon"],
        coords={"year": years, "member": np.arange(n_mem),
                "lat": np.arange(n_lat), "lon": np.arange(n_lon)},
    )
    obs = xr.DataArray(
        0.8 * signal[:, 0] + rng.normal(30.0, 30.0, (n_years, n_lat, n_lon)),
        dims=["year", "lat", "lon"],
        coords={"year": years, "lat": np.arange(n_lat), "lon": np.arange(n_lon)},
    )
    fcst = xr.DataArray(
        rng.normal(230.0, 25.0, (n_mem, n_lat, n_lon)),
        dims=["member", "lat", "lon"],
        coords={"member": np.arange(n_mem),
                "lat": np.arange(n_lat), "lon": np.arange(n_lon)},
    )
    return hind, obs, fcst


def _r_transcription(hind, obs, fcst, la, lo):
    """ICPAC's regrEns default path, transcribed for one cell."""
    hm = hind.mean("member").values[:, la, lo]
    yo = obs.values[:, la, lo]
    n = len(hm)
    b, a = np.polyfit(hm, yo, 1)
    fitted = a + b * hm
    eps2 = np.sum((yo - fitted) ** 2) / (n - 2)
    m = hind.sizes["member"]
    spread2 = hind.std("member", ddof=1).values[:, la, lo] ** 2 / m
    sigma_e2 = spread2.sum() / (n - 1)
    sig_a = ((n - 2) / n**2) * eps2 + ((n - 1) / n**2) * b**2 * sigma_e2
    sig_b = ((n - 2) / n) * eps2 / np.sum(hm**2) \
        + ((n - 1) / n) * sigma_e2 * np.sum(yo**2) / np.sum(hm**2) ** 2
    xfm = fcst.mean("member").values[la, lo]
    ef2 = fcst.std("member", ddof=1).values[la, lo] ** 2 / m
    sigma_f = eps2 + sig_a + sig_b * xfm**2 + b**2 * ef2
    mu = max(a + b * xfm, 0.0)
    t33, t67 = np.quantile(fitted, [1 / 3, 2 / 3])
    p_bn = norm.cdf(t33, mu, np.sqrt(sigma_f))
    p_an = 1.0 - norm.cdf(t67, mu, np.sqrt(sigma_f))
    return np.array([p_bn, 1.0 - p_bn - p_an, p_an])


def test_icpac_variance_matches_r_transcription():
    hind, obs, fcst = _data()
    m = EnsembleRegressionMethod(clip_negative=True, variance="icpac",
                                 fitted_threshold_years="paired")
    m.fit(hind, obs)
    p = m.predict_tercile(fcst, obs, threshold_source="fitted")
    for la in range(2):
        for lo in range(2):
            np.testing.assert_allclose(
                p.values[:, la, lo], _r_transcription(hind, obs, fcst, la, lo),
                atol=1e-12)


def test_default_variance_unchanged():
    """variance='wilks' (default) must reproduce the pre-knob formula."""
    hind, obs, fcst = _data()
    m_def = EnsembleRegressionMethod(clip_negative=True)
    m_def.fit(hind, obs)
    p_def = m_def.predict_tercile(fcst, obs)
    mu = m_def.slope_ * fcst.mean("member").values + m_def.intercept_
    lev = 1.0 / m_def.n_eff_ + (fcst.mean("member").values - m_def.x_mean_) ** 2 / m_def.sxx_
    sigma = np.sqrt(m_def.pev_ * (1.0 + lev))
    t33 = obs.quantile(1 / 3, dim="year").values
    np.testing.assert_allclose(p_def.values[0], norm.cdf(t33, mu, sigma), atol=1e-12)


def test_wet_freq_guard_masks_dry_cells():
    hind, obs, fcst = _data()
    obs = obs.copy()
    obs[:, 0, 0] = 0.5                       # never exceeds 1 mm -> fails (1.0, 5.0)
    m = EnsembleRegressionMethod(variance="icpac", wet_freq=(1.0, 5.0))
    m.fit(hind, obs)
    p = m.predict_tercile(fcst, obs, threshold_source="fitted")
    assert np.isnan(p.values[:, 0, 0]).all()
    assert np.isfinite(p.values[:, 1, 1]).all()


def test_min_valid_each_and_obs_variance_guards():
    hind, obs, fcst = _data()
    obs = obs.copy()
    obs[8:, 0, 0] = np.nan                   # 8 finite obs years < 15
    obs[:, 0, 1] = 100.0                     # constant obs -> sd == 0
    m = EnsembleRegressionMethod(min_valid_each=15, require_obs_variance=True)
    m.fit(hind, obs)
    p = m.predict_tercile(fcst, obs)
    assert np.isnan(p.values[:, 0, 0]).all()
    assert np.isnan(p.values[:, 0, 1]).all()
    assert np.isfinite(p.values[:, 1, 1]).all()


def test_tercile_floor_masks_low_fitted_tercile():
    hind, obs, fcst = _data()
    p_all = EnsembleRegressionMethod().fit(hind, obs).predict_tercile(
        fcst, obs, threshold_source="fitted")
    assert np.isfinite(p_all.values).all()
    # An absurdly high floor masks every cell; None masks none.
    m = EnsembleRegressionMethod()
    m.fit(hind, obs)
    p_floor = m.predict_tercile(fcst, obs, threshold_source="fitted",
                                tercile_floor=1e6)
    assert np.isnan(p_floor.values).all()


def test_paired_threshold_years_uses_only_paired_years():
    hind, obs, fcst = _data()
    obs = obs.copy()
    obs[:5, 0, 0] = np.nan                   # 5 unpaired hindcast years at one cell
    m_all = EnsembleRegressionMethod(fitted_threshold_years="all")
    m_all.fit(hind, obs)
    m_pair = EnsembleRegressionMethod(fitted_threshold_years="paired")
    m_pair.fit(hind, obs)
    t_all = m_all.fitted_hindcast_.quantile(1 / 3, dim="year").values
    t_pair = m_pair.fitted_hindcast_.quantile(1 / 3, dim="year").values
    assert not np.isclose(t_all[0, 0], t_pair[0, 0])      # masked years move t33
    np.testing.assert_allclose(t_all[1, 1], t_pair[1, 1])  # full cells unchanged


def test_calibrate_passes_ereg_knobs_through():
    hind, obs, fcst = _data()
    p = calibrate({"m": (hind, fcst)}, obs=obs, method="ereg",
                  clip_negative=True, threshold_source="fitted",
                  variance="icpac", min_valid_each=15, wet_freq=(1.0, 5.0),
                  require_obs_variance=True, fitted_threshold_years="paired",
                  tercile_floor=0.0, forecast_year=2010)
    for la in range(2):
        for lo in range(2):
            np.testing.assert_allclose(
                p.values[:, la, lo], _r_transcription(hind, obs, fcst, la, lo),
                atol=1e-12)
