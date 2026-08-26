"""§7 CPT_ARGS parity tests (issue #23).

Each of the six knobs has a test that toggling it changes the output in the
expected way:
  - drymask_threshold  : removes low-climatology predictand cells before the fit
  - transform_predictand="Empirical" : rank-transform round-trip (bounded output)
  - transform_predictand="Gamma"     : deferred -> NotImplementedError
  - tailoring="Anomaly": output is the anomaly (climatological mean not added back)
  - synchronous_predictors : accepted (documented; no behaviour change)
  - skillmask_threshold: low-CV-skill cells replaced with climatology (post-CV)
  - crossvalidation_window: threaded into the CV scheme's window
"""
import numpy as np
import pytest
import xarray as xr

from africas2s.methods.cca import CCAMethod
from africas2s.pipelines.seasonal import seasonal_mme, _METHOD_PARAMS


def _tracks(hcst):
    return {"prcp": {"m1": (hcst, None)}}


# ---- C1: whitelist ----
def test_method_params_whitelist_includes_cca_cpt_args():
    for k in ("transform_predictand", "tailoring", "drymask_threshold"):
        assert k in _METHOD_PARAMS


# ---- C2: drymask_threshold (before the fit) ----
def test_drymask_excludes_low_climatology_cells(synthetic_gcm_hindcast, synthetic_obs):
    obs = synthetic_obs.copy()
    obs[:, 0, 0] = 1e-3                      # near-zero climatology corner cell
    m = CCAMethod(drymask_threshold=0.1)
    m.fit(synthetic_gcm_hindcast, obs)
    pred = m.predict(synthetic_gcm_hindcast.isel(year=0))   # forecast keeps member dim
    assert np.isnan(pred.isel(lat=0, lon=0).values).all()
    assert np.isfinite(pred.isel(lat=10, lon=10).values).any()


def test_no_drymask_keeps_all_cells(synthetic_gcm_hindcast, synthetic_obs):
    obs = synthetic_obs.copy()
    obs[:, 0, 0] = 1e-3
    m = CCAMethod(drymask_threshold=None)
    m.fit(synthetic_gcm_hindcast, obs)
    pred = m.predict(synthetic_gcm_hindcast.isel(year=0))
    assert np.isfinite(pred.isel(lat=0, lon=0).values).any()


def test_cca_fit_records_y_eof_modes(synthetic_gcm_hindcast, synthetic_obs):
    m = CCAMethod(x_eof_modes=2, y_eof_modes=3, cca_modes=1)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    assert m.x_eof_modes_ == 2
    assert m.y_eof_modes_ == 3
    assert m.ncc_ == 1


# ---- C3: transform_predictand ----
def test_transform_empirical_predictions_bounded(synthetic_gcm_hindcast, synthetic_obs):
    m = CCAMethod(transform_predictand="Empirical")
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    pred = m.predict(synthetic_gcm_hindcast.isel(year=0)).mean("member")
    vals = pred.values[np.isfinite(pred.values)]
    # empirical inverse-CDF maps back within the observed range (no blow-ups)
    assert vals.min() >= float(synthetic_obs.min()) - 1e-6
    assert vals.max() <= float(synthetic_obs.max()) + 1e-6


def test_transform_gamma_not_implemented(synthetic_gcm_hindcast, synthetic_obs):
    m = CCAMethod(transform_predictand="Gamma")
    with pytest.raises(NotImplementedError):
        m.fit(synthetic_gcm_hindcast, synthetic_obs)


# ---- C4: tailoring + synchronous_predictors ----
def test_tailoring_anomaly_changes_output(synthetic_gcm_hindcast, synthetic_obs):
    full = CCAMethod(tailoring=None)
    full.fit(synthetic_gcm_hindcast, synthetic_obs)
    pred_full = full.predict(synthetic_gcm_hindcast.isel(year=0)).mean("member")

    anom = CCAMethod(tailoring="Anomaly")
    anom.fit(synthetic_gcm_hindcast, synthetic_obs)
    pred_anom = anom.predict(synthetic_gcm_hindcast.isel(year=0)).mean("member")

    # full field sits near the obs climatology (~5); the anomaly sits near 0.
    assert abs(float(pred_anom.mean())) < 1.0 < abs(float(pred_full.mean()))


def test_synchronous_predictors_accepted():
    m = CCAMethod(synchronous_predictors=False)
    assert m.synchronous_predictors is False


# ---- C5: skillmask_threshold (post-CV, orchestrator) ----
def test_skillmask_replaces_low_skill_with_climatology(synthetic_gcm_hindcast, synthetic_obs):
    years = synthetic_obs.year.values
    climo = synthetic_obs.sel(year=years).mean("year")
    # threshold 2.0 > max possible Pearson (1.0) => every cell replaced by climo
    masked = seasonal_mme(_tracks(synthetic_gcm_hindcast), synthetic_obs,
                          cpt_args={"skillmask_threshold": 2.0}, verbose=False)
    np.testing.assert_allclose(masked.forecast.values, climo.values, atol=1e-6)

    plain = seasonal_mme(_tracks(synthetic_gcm_hindcast), synthetic_obs, verbose=False)
    assert not np.allclose(plain.forecast.values, climo.values, atol=1e-6)


# ---- C6: crossvalidation_window threading ----
def test_crossvalidation_window_threaded(monkeypatch, synthetic_gcm_hindcast, synthetic_obs):
    import africas2s.pipelines.seasonal as S
    captured = {}

    def fake_cv(years, window=1):
        captured["window"] = window
        for y in years:
            yield [yy for yy in years if yy != y], y

    monkeypatch.setattr(S, "get_cv", lambda name: fake_cv)
    seasonal_mme(_tracks(synthetic_gcm_hindcast), synthetic_obs,
                 cpt_args={"crossvalidation_window": 3}, verbose=False)
    assert captured["window"] == 3


def test_cca_mode_selection_auto_applies_selected_modes(
    monkeypatch, synthetic_gcm_hindcast, synthetic_obs
):
    import africas2s.methods.cca as C

    captured = {}

    def fake_select_modes(
        gcm, obs, years, window, x_eof_range, y_eof_range, cca_range,
        fallback_modes=None,
    ):
        captured.update(
            window=window,
            x_eof_range=x_eof_range,
            y_eof_range=y_eof_range,
            cca_range=cca_range,
            fallback_modes=fallback_modes,
        )
        return 2, 3, 1, 0.42, None, None

    monkeypatch.setattr(C, "select_modes", fake_select_modes)

    result = seasonal_mme(
        _tracks(synthetic_gcm_hindcast),
        synthetic_obs,
        method="cca",
        cv="loyo",
        cpt_args={
            "mode_selection": "auto",
            "crossvalidation_window": 5,
            "x_eof_range": (1, 4),
            "y_eof_range": (1, 5),
            "cca_range": (1, 2),
            "mode_selection_fallback": (2, 3, 1),
        },
        verbose=False,
    )

    method = result.per_model_methods[("prcp", "m1")]
    assert captured == {
        "window": 5,
        "x_eof_range": (1, 4),
        "y_eof_range": (1, 5),
        "cca_range": (1, 2),
        "fallback_modes": (2, 3, 1),
    }
    assert method.x_eof_modes_ == 2
    assert method.y_eof_modes_ == 3
    assert method.ncc_ == 1
    assert method.mode_selection_goodness_ == 0.42


def test_cca_mode_selection_auto_requires_loyo(synthetic_gcm_hindcast, synthetic_obs):
    with pytest.raises(ValueError, match="requires cv='loyo'"):
        seasonal_mme(
            _tracks(synthetic_gcm_hindcast),
            synthetic_obs,
            method="cca",
            cv="expanding",
            cpt_args={"mode_selection": "auto"},
            verbose=False,
        )


def test_capped_mode_ranges_bounds_by_sample_size():
    """Mode search ranges are capped by what the sample supports, so the search
    can't pick mode counts that drive the downstream Student-t dof (n-modes-1)
    to <= 1 on short hindcasts."""
    from africas2s.methods.cca import _capped_mode_ranges

    # 11 years, window=1 -> cap = 11 - 1 - 2 = 8
    x, y, c = _capped_mode_ranges((1, 10), (1, 10), (1, 10), n_years=11, window=1)
    assert x == (1, 8) and y == (1, 8) and c == (1, 8)
    # Long hindcast: the (1, 10) default is unaffected (cap is generous).
    x2, _, _ = _capped_mode_ranges((1, 10), (1, 6), (1, 3), n_years=40, window=1)
    assert x2 == (1, 10)
    # Very short hindcast: never collapses below a single mode.
    x3, y3, c3 = _capped_mode_ranges((2, 10), (1, 10), (1, 10), n_years=4, window=1)
    assert x3 == (1, 1) and y3 == (1, 1) and c3 == (1, 1)


def test_select_modes_does_not_pick_degenerate_high_modes_on_short_hindcast(
    synthetic_gcm_hindcast, synthetic_obs
):
    """On a short hindcast the selected modes must keep the full-data CPT dof
    (n - x_eof - 1) >= 2, so cpt_per_model never silently drops the model."""
    from africas2s.methods.cca import select_modes

    years = [int(y) for y in synthetic_obs.year.values][:8]
    gcm = synthetic_gcm_hindcast.sel(year=years)
    obs = synthetic_obs.sel(year=years)
    xe, ye, cc, *_ = select_modes(
        gcm, obs, years, window=1,
        x_eof_range=(1, 10), y_eof_range=(1, 10), cca_range=(1, 10),
    )
    n = len(years)
    assert n - xe - 1 >= 2
    assert xe <= n - 1 - 2 and ye <= n - 1 - 2 and cc <= n - 1 - 2


def test_cca_mode_selection_provides_default_fallback(
    monkeypatch, synthetic_gcm_hindcast, synthetic_obs
):
    """When the caller doesn't pass mode_selection_fallback, the pipeline must
    still supply one so a model whose CV goodness is never finite falls back to
    minimal modes instead of aborting the whole MME run."""
    import africas2s.methods.cca as C

    captured = {}

    def fake_select_modes(gcm, obs, years, window, x_eof_range, y_eof_range,
                          cca_range, fallback_modes=None):
        captured["fallback_modes"] = fallback_modes
        return 1, 1, 1, 0.1, None, None

    monkeypatch.setattr(C, "select_modes", fake_select_modes)
    seasonal_mme(
        _tracks(synthetic_gcm_hindcast), synthetic_obs,
        method="cca", cv="loyo",
        cpt_args={"mode_selection": "auto"},   # no mode_selection_fallback
        verbose=False,
    )
    assert captured["fallback_modes"] is not None


def test_select_modes_raises_when_goodness_is_never_finite(
    synthetic_gcm_hindcast, synthetic_obs
):
    from africas2s.methods.cca import select_modes

    obs = synthetic_obs * np.nan
    years = [int(y) for y in obs.year.values]
    with pytest.raises(ValueError, match="no finite Kendall goodness"):
        select_modes(
            synthetic_gcm_hindcast,
            obs,
            years,
            window=1,
            x_eof_range=(1, 1),
            y_eof_range=(1, 1),
            cca_range=(1, 1),
        )
