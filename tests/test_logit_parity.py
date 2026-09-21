"""Unit tests for the ICPAC-parity knobs on the logit stream.

Each knob mirrors one line of ICPAC's zFcstSST1LeadDriver.R; defaults must
leave legacy behavior untouched (covered by test_logistic.py, plus the
default-equivalence checks here).
"""
import numpy as np
import pytest
import xarray as xr

from africas2s.indices import Index
from africas2s.logistic import _labels_from_obs, logistic_forecast
from africas2s.calibrate import calibrate


def _obs(values, n_lat=1, n_lon=1):
    values = np.asarray(values, dtype=float).reshape(-1, n_lat, n_lon)
    ny = values.shape[0]
    return xr.DataArray(
        values, dims=["year", "lat", "lon"],
        coords={"year": np.arange(2000, 2000 + ny),
                "lat": np.arange(n_lat), "lon": np.arange(n_lon)},
    )


@pytest.fixture
def index():
    return xr.DataArray(np.linspace(-2.0, 2.0, 30),
                        dims=["year"], coords={"year": np.arange(2000, 2030)})


def _rng_obs(index, seed=0):
    rng = np.random.default_rng(seed)
    vals = 120.0 - 30.0 * np.asarray(index)[:, None] + rng.normal(0, 10, (len(index), 4))
    return _obs(vals, n_lat=2, n_lon=2)


# ── obs_threshold / obs_rounding ─────────────────────────────────────────────

def test_obs_threshold_masks_dry_years(index):
    """yobs[yobs < rthr] <- NA: sub-threshold years leave the fit entirely."""
    vals = 120.0 - 30.0 * np.asarray(index)
    vals[:5] = 0.4                                     # 5 'dry' years under rthr=1
    p_masked = logistic_forecast(index, _obs(vals), 1.0, obs_threshold=1.0)
    manual = vals.copy()
    manual[:5] = np.nan
    p_manual = logistic_forecast(index, _obs(manual), 1.0)
    np.testing.assert_allclose(p_masked.values, p_manual.values, atol=1e-12)


def test_obs_threshold_applies_before_rounding(index):
    """R masks on the raw total, then rounds: 0.96 is masked even though it
    would round to 1.0."""
    vals = 120.0 - 30.0 * np.asarray(index)
    vals[0] = 0.96
    p = logistic_forecast(index, _obs(vals), 1.0,
                          obs_threshold=1.0, obs_rounding=1)
    manual = np.round(vals, 1)
    manual[0] = np.nan
    p_manual = logistic_forecast(index, _obs(manual), 1.0)
    np.testing.assert_allclose(p.values, p_manual.values, atol=1e-12)


def test_obs_rounding_equals_prerounded_obs(index):
    obs = _rng_obs(index)
    p_knob = logistic_forecast(index, obs, 1.0, obs_rounding=1)
    p_manual = logistic_forecast(index, obs.round(1), 1.0)
    np.testing.assert_allclose(p_knob.values, p_manual.values, atol=1e-12)


# ── threshold_rounding ───────────────────────────────────────────────────────

def test_threshold_rounding_moves_boundary_ties_like_R():
    """With boundaries rounded to 0.1, a value tied to the rounded boundary
    classifies as normal (exclusive edges), as in the R's <, > comparisons."""
    # n=7 -> t33 is exactly the 3rd sorted value (100.44), which rounds down
    # to 100.4: the year at 100.42 is below the raw boundary but above the
    # rounded one, so its label flips from below to normal.
    vals = np.array([99.0, 100.42, 100.44, 101.0, 102.0, 103.0, 104.0])
    obs_vals = vals[:, None]
    labels_raw, t33, _ = _labels_from_obs(obs_vals)
    labels_rnd, t33_r, _ = _labels_from_obs(obs_vals, threshold_rounding=1)
    assert t33[0] == pytest.approx(100.44)
    assert t33_r[0] == pytest.approx(100.4)
    assert labels_raw[1, 0] == 0 and labels_rnd[1, 0] == 1


# ── min_valid_each ───────────────────────────────────────────────────────────

def test_min_valid_each_masks_sparse_series(index):
    """ICPAC nmiss=15: a cell whose obs has only 12 finite years fails the
    per-series guard even though the overlap passes min_years."""
    vals = 120.0 - 30.0 * np.asarray(index)
    vals[12:] = np.nan                                  # 12 finite obs years
    p = logistic_forecast(index, _obs(vals), 1.0, min_years=11, min_valid_each=15)
    assert np.isnan(p.values).all()
    p_ok = logistic_forecast(index, _obs(vals), 1.0, min_years=11)
    assert np.isfinite(p_ok.values).all()


# ── renormalize modes ────────────────────────────────────────────────────────

def test_renormalize_cap_above_keeps_fitted_values(index):
    """cap_above must not rescale a triple that is at/under the cap, and must
    reset above (only) on an overshoot: sum is then exactly 1."""
    obs = _rng_obs(index)
    p = logistic_forecast(index, obs, 1.0, renormalize="cap_above",
                          backend="statsmodels")
    s = p.sum("tercile").values
    assert (s <= 1.0001 + 1e-12).all()
    p_sum = logistic_forecast(index, obs, 1.0, renormalize="sum",
                              backend="statsmodels")
    # The three fitted binomials almost never sum to exactly 1, so the two
    # modes must differ (that difference IS the ICPAC residual source).
    assert not np.allclose(p.values, p_sum.values, atol=1e-6)


def test_renormalize_none_keeps_raw_fitted_sums(index):
    """renormalize=None leaves the three independent binomials untouched, so
    somewhere on a noisy grid their sum deviates from 1."""
    p_none = logistic_forecast(index, _rng_obs(index), 1.0,
                               renormalize=None, backend="statsmodels")
    assert float(abs(p_none.sum("tercile") - 1.0).max()) > 1e-9


def test_default_renormalize_unchanged(index):
    obs = _rng_obs(index)
    p_default = logistic_forecast(index, obs, 1.0)
    p_sum = logistic_forecast(index, obs, 1.0, renormalize="sum")
    np.testing.assert_allclose(p_default.values, p_sum.values, atol=1e-12)
    np.testing.assert_allclose(p_default.sum("tercile").values, 1.0, atol=1e-9)


# ── combine_renormalize (calibrate level) ────────────────────────────────────

def test_combine_renormalize_sums_to_one(index):
    # Weak-signal obs: the fitted triples land under the cap (cap_above leaves
    # them alone, sums != 1), so the combine-level renormalization is what
    # brings the combined map to 1 — the ICPAC combine_components.py step.
    rng = np.random.default_rng(3)
    obs = _obs(rng.normal(120.0, 10.0, (len(index), 4)), n_lat=2, n_lon=2)
    hind = {"m/a": index, "m/b": index * 0.5}
    fc = {"m/a": 1.0, "m/b": 0.5}
    p_raw = calibrate(hind, obs=obs, method="logit", forecast=fc,
                      backend="statsmodels", renormalize="cap_above",
                      forecast_year=2030)
    assert float(abs(p_raw.sum("tercile") - 1.0).max()) > 1e-6
    p = calibrate(hind, obs=obs, method="logit", forecast=fc,
                  backend="statsmodels", renormalize="cap_above",
                  combine_renormalize=True, forecast_year=2030)
    np.testing.assert_allclose(p.sum("tercile").values, 1.0, atol=1e-9)


# ── Index.reduce(pixel_standardize=True) ─────────────────────────────────────

def test_pixel_standardize_matches_manual_ncl_construction():
    rng = np.random.default_rng(1)
    field = xr.DataArray(
        rng.normal(25, 2, (10, 4, 8)), dims=["year", "lat", "lon"],
        coords={"year": np.arange(2000, 2010),
                "lat": np.linspace(-6, 6, 4), "lon": np.linspace(100, 240, 8)},
    )
    idx = Index.custom(name="box", regions={"box": [-10, 10, 120, 200]},
                       combine=lambda z: z["box"], transform="raw")
    got = idx.reduce(field, pixel_standardize=True)
    z = (field - field.mean("year")) / field.std("year", ddof=1)
    sub = z.sel(lat=slice(-10, 10), lon=slice(120, 200))
    np.testing.assert_allclose(got.values, sub.mean(("lat", "lon")).values,
                               atol=1e-10)


def test_pixel_standardize_uses_climatology_reference():
    rng = np.random.default_rng(2)
    hind = xr.DataArray(
        rng.normal(25, 2, (10, 3, 3)), dims=["year", "lat", "lon"],
        coords={"year": np.arange(2000, 2010),
                "lat": [0.0, 1.0, 2.0], "lon": [150.0, 151.0, 152.0]},
    )
    fcst = hind.isel(year=[0]) + 1.0
    idx = Index.custom(name="box", regions={"box": [-5, 5, 145, 155]},
                       combine=lambda z: z["box"], transform="raw")
    got = idx.reduce(fcst, climatology=hind, pixel_standardize=True)
    z = (fcst - hind.mean("year")) / hind.std("year", ddof=1)
    np.testing.assert_allclose(got.values, z.mean(("lat", "lon")).values,
                               atol=1e-10)
