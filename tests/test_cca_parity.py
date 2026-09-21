"""Unit tests for the CPT.x-parity knobs on the CCA stream.

Each knob mirrors a verified behavior of CPT 17/18 Fortran (space.F95
weightByLatitude, pcs.F95 calcPCs lpos, cv wraparound, regression.F95
calcRegrProbs snap, cpt.ini climatological period); defaults leave legacy
behavior untouched.
"""
import numpy as np
import pytest
import xarray as xr

from africas2s.cv import loyo, loyo_cyclic
from africas2s.methods.cca import CCAMethod, cpt_latitude_weights
from africas2s.tercile import cpt_tercile_forecast


def _data(n=20, seed=0, ny=(6, 7), nx=(5, 9)):
    rng = np.random.default_rng(seed)
    years = np.arange(2000, 2000 + n)
    hind = xr.DataArray(rng.normal(5, 2, (n, 3, *nx)),
                        dims=["year", "member", "lat", "lon"],
                        coords={"year": years, "member": np.arange(3),
                                "lat": np.linspace(-20, 20, nx[0]),
                                "lon": np.linspace(0, 80, nx[1])})
    obs = xr.DataArray(rng.normal(100, 30, (n, *ny)),
                       dims=["year", "lat", "lon"],
                       coords={"year": years,
                               "lat": np.linspace(-10, 10, ny[0]),
                               "lon": np.linspace(30, 42, ny[1])})
    fcst = hind.isel(year=0, drop=True) + 1.0
    return hind, obs, fcst


# ── cpt_latitude_weights ─────────────────────────────────────────────────────

def test_cpt_latitude_weights_matches_fortran_formula():
    lats = np.array([22.875, 22.625, 22.375])           # 0.25 deg, descending
    w = cpt_latitude_weights(lats)
    r1 = (3 * 22.875 - 22.625) * np.pi / 360
    r2 = (22.875 + 22.625) * np.pi / 360
    exp0 = np.sqrt(abs((np.sin(r1) - np.sin(r2)) / (r1 - r2)))
    np.testing.assert_allclose(w[0], exp0, rtol=1e-12)
    # interior band straddles the latitude; close to sqrt(cos) but not equal
    assert abs(w[1] - np.sqrt(np.cos(np.deg2rad(22.625)))) < 1e-4
    # symmetric under reversal
    np.testing.assert_allclose(w, cpt_latitude_weights(lats[::-1])[::-1], rtol=1e-12)


# ── sign convention ──────────────────────────────────────────────────────────

def test_cpt_sign_convention_canonicalizes_eofs_and_preserves_predictions():
    hind, obs, fcst = _data()
    m0 = CCAMethod(x_eof_modes=3, y_eof_modes=3, cca_modes=2, standardize=True,
                   lat_weights="cpt")
    m1 = CCAMethod(x_eof_modes=3, y_eof_modes=3, cca_modes=2, standardize=True,
                   lat_weights="cpt", sign_convention="cpt")
    m0.fit(hind, obs)
    m1.fit(hind, obs)
    # canonicalized loadings: largest-|.|, lat-rescaled entry positive
    disp = m1.eofx_ * m1.x_wt_[:, None]
    for c in range(disp.shape[1]):
        assert abs(disp[:, c].max()) >= abs(disp[:, c].min())
    # deterministic predictions are sign-invariant
    np.testing.assert_allclose(m0.predict(fcst).values, m1.predict(fcst).values,
                               rtol=0, atol=1e-8)
    # per-mode loadings equal up to sign
    for c in range(m0.eofx_.shape[1]):
        assert np.allclose(m0.eofx_[:, c], m1.eofx_[:, c], atol=1e-10) or \
            np.allclose(m0.eofx_[:, c], -m1.eofx_[:, c], atol=1e-10)


def test_leverage_is_sign_dependent_by_construction():
    """CPT's xvp = 1/n + (sum of canonical scores)^2 changes when a mode's
    sign flips — the reason sign_convention='cpt' exists at all."""
    hind, obs, fcst = _data(seed=3)
    m = CCAMethod(x_eof_modes=3, y_eof_modes=3, cca_modes=2, standardize=True,
                  sign_convention="cpt")
    m.fit(hind, obs)
    lev1 = m.leverage(fcst)
    m.s_ = m.s_.copy()
    m.s_[0, :] = -m.s_[0, :]   # flip one canonical mode by hand
    lev2 = m.leverage(fcst)
    assert not np.isclose(lev1, lev2)


# ── cyclic CV ────────────────────────────────────────────────────────────────

def test_loyo_cyclic_wraps_and_holds_out_full_window():
    years = list(range(2000, 2010))
    folds = list(loyo_cyclic(years, window=5))
    assert len(folds) == 10
    train0, test0 = folds[0]
    assert test0 == 2000
    # first fold holds out 2000±2 cyclically: 2008, 2009, 2000, 2001, 2002
    assert set(years) - set(train0) == {2008, 2009, 2000, 2001, 2002}
    train_last, test_last = folds[-1]
    assert set(years) - set(train_last) == {2007, 2008, 2009, 2000, 2001}
    # every fold trains on exactly n - window years
    assert all(len(tr) == 5 for tr, _ in folds)
    # interior folds match plain loyo
    assert list(loyo(years, window=5))[5] == folds[5]


# ── snap rule ────────────────────────────────────────────────────────────────

def test_cpt_tercile_snap_on_zero_variance():
    grid = dict(lat=[0.0, 1.0], lon=[10.0, 11.0])
    fc = xr.DataArray([[1.0, 5.0], [10.0, np.nan]], dims=["lat", "lon"], coords=grid)
    t33 = xr.DataArray(np.full((2, 2), 3.0), dims=["lat", "lon"], coords=grid)
    t67 = xr.DataArray(np.full((2, 2), 7.0), dims=["lat", "lon"], coords=grid)
    s2 = xr.DataArray(np.zeros((2, 2)), dims=["lat", "lon"], coords=grid)
    p = cpt_tercile_forecast(fc, t33, t67, s2, dofr=10, leverage=0.0)
    np.testing.assert_allclose(p.values[:, 0, 0], [1, 0, 0])   # 1.0 <= t33 -> below
    np.testing.assert_allclose(p.values[:, 0, 1], [0, 1, 0])   # between -> normal
    np.testing.assert_allclose(p.values[:, 1, 0], [0, 0, 1])   # > t67 -> above
    assert np.isnan(p.values[:, 1, 1]).all()                   # NaN forecast stays NaN


def test_cpt_tercile_normal_variance_unchanged():
    grid = dict(lat=[0.0], lon=[10.0])
    fc = xr.DataArray([[5.0]], dims=["lat", "lon"], coords=grid)
    t33 = xr.DataArray([[3.0]], dims=["lat", "lon"], coords=grid)
    t67 = xr.DataArray([[7.0]], dims=["lat", "lon"], coords=grid)
    s2 = xr.DataArray([[4.0]], dims=["lat", "lon"], coords=grid)
    p = cpt_tercile_forecast(fc, t33, t67, s2, dofr=10, leverage=0.0)
    np.testing.assert_allclose(p.sum("tercile").values, 1.0, atol=1e-12)
    assert 0.05 < float(p.values[0, 0, 0]) < 0.5


# ── climatology_period (seasonal_mme) ────────────────────────────────────────

def test_seasonal_mme_climatology_period_changes_boundaries():
    from africas2s import seasonal_mme
    hind, obs, fcst = _data(n=24, seed=5)
    fyr = int(obs.year[-1]) + 1
    fcst = fcst.expand_dims(year=[fyr])
    kw = dict(method="cca", probability_aggregation="cpt_per_model",
              cpt_args=dict(x_eof_modes=3, y_eof_modes=3, cca_modes=2,
                            crossvalidation_window=3, standardize=True),
              forecast_year=fyr, verbose=False)
    r_full = seasonal_mme({"prcp": {"m": (hind, fcst)}}, obs, **kw)
    r_clim = seasonal_mme({"prcp": {"m": (hind, fcst)}}, obs,
                          climatology_period=(2000, 2011), **kw)
    a = r_full.tercile_forecast.values
    b = r_clim.tercile_forecast.values
    ok = np.isfinite(a) & np.isfinite(b)
    assert not np.allclose(a[ok], b[ok])
