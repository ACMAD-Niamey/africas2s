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


# ── mode search uses the same settings as the fit ────────────────────────────

_PARITY = dict(standardize=True, lat_weights="cpt", sign_convention="cpt",
               y_reconstruction="cpt")


def test_select_modes_builds_trial_models_with_method_kwargs(
    monkeypatch, synthetic_gcm_hindcast, synthetic_obs
):
    """Every trial CCA in the search is built with the caller's model settings,
    and the folds come from the caller's CV scheme."""
    import africas2s.methods.cca as C
    from africas2s.cv import loyo_cyclic

    seen_kwargs, seen_windows = [], []
    real = C.CCAMethod

    def spy_method(**kwargs):
        seen_kwargs.append(kwargs)
        return real(**kwargs)

    def spy_cv(years, window=1):
        seen_windows.append(window)
        return loyo_cyclic(years, window=window)

    monkeypatch.setattr(C, "CCAMethod", spy_method)
    years = [int(y) for y in synthetic_obs.year.values]
    C.select_modes(
        synthetic_gcm_hindcast, synthetic_obs, years, window=3,
        x_eof_range=(1, 2), y_eof_range=(1, 2), cca_range=(1, 1),
        method_kwargs={**_PARITY, "x_eof_modes": 99, "tailoring": "Anomaly"},
        cv=spy_cv,
    )
    assert seen_windows == [3]
    assert seen_kwargs
    for kwargs in seen_kwargs:
        assert {k: kwargs[k] for k in _PARITY} == _PARITY
        assert kwargs["x_eof_modes"] in (1, 2)      # the searched value, not the caller's 99
        assert "tailoring" not in kwargs            # only model-shaping keys are forwarded


def test_select_modes_defaults_unchanged(synthetic_gcm_hindcast, synthetic_obs):
    """No method_kwargs / cv is the same as passing the defaults explicitly."""
    from africas2s.cv import loyo
    from africas2s.methods.cca import select_modes

    years = [int(y) for y in synthetic_obs.year.values]
    kw = dict(window=1, x_eof_range=(1, 3), y_eof_range=(1, 3), cca_range=(1, 2))
    a = select_modes(synthetic_gcm_hindcast, synthetic_obs, years, **kw)
    b = select_modes(synthetic_gcm_hindcast, synthetic_obs, years, **kw,
                     method_kwargs={}, cv=loyo)
    assert a[:3] == b[:3]
    assert a[3] == b[3]
    np.testing.assert_array_equal(a[4].values, b[4].values)


def test_select_modes_settings_change_the_ranking_input(synthetic_gcm_hindcast, synthetic_obs):
    """The parity settings reach the trial fit: its CV predictions differ."""
    from africas2s.methods.cca import select_modes

    years = [int(y) for y in synthetic_obs.year.values]
    kw = dict(window=1, x_eof_range=(2, 2), y_eof_range=(2, 2), cca_range=(1, 1))
    plain = select_modes(synthetic_gcm_hindcast, synthetic_obs, years, **kw)
    parity = select_modes(synthetic_gcm_hindcast, synthetic_obs, years, **kw,
                          method_kwargs=_PARITY)
    assert not np.allclose(plain[4].values, parity[4].values, equal_nan=True)


def test_seasonal_mme_forwards_fit_settings_to_mode_search(
    monkeypatch, synthetic_gcm_hindcast, synthetic_obs
):
    """seasonal_mme hands the search the model settings and CV scheme of the fit."""
    import africas2s.methods.cca as C
    from africas2s.cv import loyo_cyclic

    captured = {}

    def fake_select_modes(gcm, obs, years, window, x_eof_range, y_eof_range,
                          cca_range, fallback_modes=None, *, method_kwargs=None, cv=None):
        captured.update(method_kwargs=method_kwargs, cv=cv)
        return 2, 2, 1, 0.3, None, None

    monkeypatch.setattr(C, "select_modes", fake_select_modes)
    seasonal_mme(
        _tracks(synthetic_gcm_hindcast), synthetic_obs,
        method="cca", cv="loyo_cyclic",
        cpt_args={"mode_selection": "cpt", "crossvalidation_window": 3, **_PARITY},
        verbose=False,
    )
    assert captured["method_kwargs"] == _PARITY
    assert captured["cv"] is loyo_cyclic


# ── CPT's over-fitting exclusion and CPT's Kendall tau in the mode search ────

def test_kendall_tau_cpt_equals_tau_b_without_ties():
    from scipy.stats import kendalltau
    from africas2s.methods.cca import _kendall_tau_cpt

    rng = np.random.default_rng(0)
    for _ in range(5):
        p, o = rng.normal(size=20), rng.normal(size=20)
        assert _kendall_tau_cpt(p, o) == pytest.approx(kendalltau(p, o).statistic, abs=1e-12)


def test_kendall_tau_cpt_tie_rules_match_cpt_two_afc_cc():
    """Literal port of CPT scores.F95 two_afc_cc as the reference."""
    from africas2s.methods.cca import _kendall_tau_cpt

    def fortran_sign_one(x):          # Fortran Sign(one, x): +1 for x >= 0
        return 1.0 if x >= 0 else -1.0

    def reference(pred, obs):
        n, afc = len(obs), 0.0
        for k in range(n - 1):
            for l in range(k + 1, n):
                if obs[k] != obs[l] or pred[k] != pred[l]:
                    afc += abs(fortran_sign_one(obs[k] - obs[l])
                               + fortran_sign_one(pred[k] - pred[l])) / 2.0
                else:
                    afc += 0.5
        return 2.0 * afc / (n * (n - 1) / 2) - 1.0

    obs = np.array([0.0, 0.0, 0.0, 1.2, 0.0, 3.4, 0.5, 0.0, 0.5])    # ties, as in a dry cell
    pred = np.array([0.1, 0.4, 0.1, 0.9, 0.3, 0.2, 0.8, 0.0, 0.7])   # one tie shared with obs
    assert _kendall_tau_cpt(pred, obs) == pytest.approx(reference(pred, obs), abs=1e-12)

    from scipy.stats import kendalltau
    assert _kendall_tau_cpt(pred, obs) != pytest.approx(kendalltau(pred, obs).statistic, abs=1e-6)


def _short_case(synthetic_gcm_hindcast, synthetic_obs, n=9):
    years = [int(y) for y in synthetic_obs.year.values][:n]
    return synthetic_gcm_hindcast.sel(year=years), synthetic_obs.sel(year=years), years


def test_select_modes_excludes_overfitted_pairs(monkeypatch, synthetic_gcm_hindcast, synthetic_obs):
    """A pair whose modes fit a fold's training years exactly (canonical
    correlation 1) cannot win, however good its CV goodness looks."""
    import africas2s.methods.cca as C

    gcm, obs, years = _short_case(synthetic_gcm_hindcast, synthetic_obs)
    # 8 training years; (4, 4) spans the 7 centred degrees of freedom -> mu == 1
    monkeypatch.setattr(C, "_capped_mode_ranges", lambda x, y, c, n, w: (tuple(x), tuple(y), tuple(c)))
    kw = dict(window=1, x_eof_range=(4, 4), y_eof_range=(4, 4), cca_range=(1, 1))

    m = C.CCAMethod(x_eof_modes=4, y_eof_modes=4, cca_modes=1)
    m.fit(gcm.sel(year=years[1:]), obs.sel(year=years[1:]))
    assert m.mu_full_.max() > 1 - C._OVERFIT_TOL          # the premise: this pair over-fits
    assert len(m.mu_full_) == 4 and len(m.mu_) == 1

    off = C.select_modes(gcm, obs, years, **kw, overfit_check=False)
    assert off[:3] == (4, 4, 1) and np.isfinite(off[3])
    with pytest.raises(ValueError, match="over-fitted"):
        C.select_modes(gcm, obs, years, **kw)
    wide = C.select_modes(gcm, obs, years, window=1, x_eof_range=(1, 4), y_eof_range=(1, 4),
                          cca_range=(1, 1))
    assert wide[:2] != (4, 4)


def test_select_modes_overfit_check_inactive_when_nothing_overfits(
    synthetic_gcm_hindcast, synthetic_obs
):
    from africas2s.methods.cca import select_modes

    years = [int(y) for y in synthetic_obs.year.values]
    kw = dict(window=1, x_eof_range=(1, 3), y_eof_range=(1, 3), cca_range=(1, 2))
    a = select_modes(synthetic_gcm_hindcast, synthetic_obs, years, **kw)
    b = select_modes(synthetic_gcm_hindcast, synthetic_obs, years, **kw, overfit_check=False)
    assert a[:4] == b[:4]


def test_select_modes_tau_cpt_matches_tau_b_without_ties(synthetic_gcm_hindcast, synthetic_obs):
    from africas2s.methods.cca import select_modes

    years = [int(y) for y in synthetic_obs.year.values]
    kw = dict(window=1, x_eof_range=(1, 3), y_eof_range=(1, 3), cca_range=(1, 2))
    a = select_modes(synthetic_gcm_hindcast, synthetic_obs, years, **kw)
    b = select_modes(synthetic_gcm_hindcast, synthetic_obs, years, **kw, tau="cpt")
    assert a[:3] == b[:3]
    assert a[3] == pytest.approx(b[3], abs=1e-12)
    with pytest.raises(ValueError, match="tau must be"):
        select_modes(synthetic_gcm_hindcast, synthetic_obs, years, **kw, tau="c")


def test_seasonal_mme_forwards_overfit_and_tau_options(
    monkeypatch, synthetic_gcm_hindcast, synthetic_obs
):
    import africas2s.methods.cca as C

    seen = []
    real = C.select_modes

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(C, "select_modes", spy)
    base = dict(mode_selection="cpt", x_eof_range=(1, 2), y_eof_range=(1, 2), cca_range=(1, 1))
    run = lambda extra: seasonal_mme(
        _tracks(synthetic_gcm_hindcast), synthetic_obs, method="cca",
        cv="loyo", cpt_args={**base, **extra}, verbose=False)
    run({})
    assert "overfit_check" not in seen[-1] and "tau" not in seen[-1]
    run({"mode_selection_tau": "cpt", "mode_selection_overfit_check": False})
    assert seen[-1]["tau"] == "cpt" and seen[-1]["overfit_check"] is False
