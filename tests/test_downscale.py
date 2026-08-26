import numpy as np
import pytest
import xarray as xr


# ===================================================================
# 10. skill()
# ===================================================================

def test_skill_returns_report(climatology_forecast, synthetic_obs):
    from africas2s.skill import skill, SkillReport
    report = skill(climatology_forecast, synthetic_obs, metrics=["rpss"])
    assert isinstance(report, SkillReport)
    assert "rpss" in report.scores
    assert isinstance(report.scores["rpss"], float)


def test_skill_spatial_maps(climatology_forecast, synthetic_obs):
    from africas2s.skill import skill
    report = skill(climatology_forecast, synthetic_obs, metrics=["rpss"], spatial=True)
    assert "rpss" in report.spatial
    assert "lat" in report.spatial["rpss"].dims
    assert "lon" in report.spatial["rpss"].dims


def test_skill_multiple_metrics(climatology_forecast, synthetic_obs):
    from africas2s.skill import skill
    report = skill(climatology_forecast, synthetic_obs, metrics=["rpss", "roc"])
    assert "rpss" in report.scores
    assert "roc_bn" in report.scores


# ===================================================================
# 11. downscale()
# ===================================================================

def test_downscale_bcsd(synthetic_gcm_hindcast, synthetic_obs):
    import africas2s
    result = africas2s.downscale(synthetic_gcm_hindcast, synthetic_obs, method="bcsd")
    assert result.dims == ("member", "lat", "lon")
    assert len(result.lat) == len(synthetic_obs.lat)


def test_downscale_cca(synthetic_gcm_hindcast, synthetic_obs):
    import africas2s
    result = africas2s.downscale(synthetic_gcm_hindcast, synthetic_obs, method="cca")
    assert result.dims == ("member", "lat", "lon")


def test_downscale_tercile_output(synthetic_gcm_hindcast, synthetic_obs):
    import africas2s
    result = africas2s.downscale(
        synthetic_gcm_hindcast, synthetic_obs, method="bcsd", output_type="tercile"
    )
    assert "tercile" in result.dims
    sums = result.sum("tercile")
    np.testing.assert_allclose(sums.values, 1.0, atol=1e-10)


def test_downscale_accepts_predictor_hindcast_keyword(synthetic_gcm_hindcast, synthetic_obs):
    """The new canonical kwarg works without warnings."""
    import warnings
    import africas2s
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = africas2s.downscale(
            predictor_hindcast=synthetic_gcm_hindcast, obs=synthetic_obs, method="bcsd",
        )
    assert result.dims == ("member", "lat", "lon")


def test_downscale_legacy_gcm_keyword_emits_deprecation(synthetic_gcm_hindcast, synthetic_obs):
    """The old `gcm=` kwarg still works but emits a DeprecationWarning."""
    import africas2s
    with pytest.warns(DeprecationWarning, match="gcm"):
        result = africas2s.downscale(
            gcm=synthetic_gcm_hindcast, obs=synthetic_obs, method="bcsd",
        )
    assert result.dims == ("member", "lat", "lon")


def test_downscale_rejects_both_names(synthetic_gcm_hindcast, synthetic_obs):
    """Passing both `predictor_hindcast` and `gcm` is an explicit error."""
    import africas2s
    with pytest.raises(TypeError, match="both"):
        africas2s.downscale(
            predictor_hindcast=synthetic_gcm_hindcast,
            gcm=synthetic_gcm_hindcast,
            obs=synthetic_obs,
            method="bcsd",
        )


# ===================================================================
# 12. optimize()
# ===================================================================

def test_optimize_single_gcm(synthetic_gcm_hindcast, synthetic_obs):
    import africas2s
    best = africas2s.optimize(
        synthetic_gcm_hindcast, synthetic_obs,
        methods=["bcsd", "cca"], cv="loyo", primary_metric="rpss",
    )
    assert best.method in ["bcsd", "cca"]
    assert np.isfinite(best.score)
    assert best.forecast is not None
    assert "lat" in best.forecast.dims


def test_optimize_with_blocked_cv(synthetic_gcm_hindcast, synthetic_obs):
    """`optimize(cv="blocked")` works end-to-end with multi-year test folds."""
    import africas2s
    best = africas2s.optimize(
        synthetic_gcm_hindcast, synthetic_obs,
        methods=["cca"], cv="blocked", primary_metric="rpss",
        verbose=False, progress=False,
    )
    assert best.method == "cca"
    assert np.isfinite(best.score)


def test_optimize_with_lko_cv(synthetic_gcm_hindcast, synthetic_obs):
    """`optimize(cv="lko")` works with sliding multi-year test folds."""
    import africas2s
    best = africas2s.optimize(
        synthetic_gcm_hindcast, synthetic_obs,
        methods=["cca"], cv="lko", primary_metric="rpss",
        verbose=False, progress=False,
    )
    assert best.method == "cca"
    assert np.isfinite(best.score)


def test_optimize_with_expanding_cv(synthetic_gcm_hindcast, synthetic_obs):
    """`optimize(cv=callable)` works with realtime-style folds.

    `expanding`'s default min_train=10 leaves no eval years on a 10-year
    fixture, so we pass a pre-configured callable. This also exercises
    optimize()'s callable CV path.
    """
    from functools import partial
    import africas2s
    from africas2s.cv import expanding
    best = africas2s.optimize(
        synthetic_gcm_hindcast, synthetic_obs,
        methods=["cca"],
        cv=partial(expanding, min_train=4),
        primary_metric="rpss",
        verbose=False, progress=False,
    )
    assert best.method == "cca"
    assert np.isfinite(best.score)


# ===================================================================
# 13. E2E pipeline
# ===================================================================

def test_e2e_single_gcm_single_method(synthetic_gcm_hindcast, synthetic_obs):
    import africas2s
    result = africas2s.downscale(synthetic_gcm_hindcast, synthetic_obs, method="bcsd")
    assert result.dims == ("member", "lat", "lon")
    assert len(result.lat) == len(synthetic_obs.lat)


def test_e2e_multi_gcm_ensemble(synthetic_gcm_hindcast, synthetic_gcm_hindcast2, synthetic_obs):
    import africas2s
    best1 = africas2s.optimize(synthetic_gcm_hindcast, synthetic_obs,
                                methods=["bcsd"], cv="loyo", primary_metric="rpss")
    best2 = africas2s.optimize(synthetic_gcm_hindcast2, synthetic_obs,
                                methods=["bcsd"], cv="loyo", primary_metric="rpss")
    mme = africas2s.ensemble([best1, best2], synthetic_obs, strategy="uniform")
    assert "lat" in mme.forecast.dims
    assert "lon" in mme.forecast.dims


def test_e2e_drop_worst_and_skill_weighted_through_optimize(
    synthetic_gcm_hindcast, synthetic_gcm_hindcast2, synthetic_obs,
):
    """Integration: optimize→ensemble using the new skill-aware strategies.

    Both `drop_worst` and `skill_weighted` should consume the OptimizeResult
    objects directly (using their `.score` field) and produce a valid MME.
    """
    import africas2s
    best1 = africas2s.optimize(synthetic_gcm_hindcast, synthetic_obs,
                                methods=["bcsd"], cv="loyo", primary_metric="rpss",
                                verbose=False, progress=False)
    best2 = africas2s.optimize(synthetic_gcm_hindcast2, synthetic_obs,
                                methods=["bcsd"], cv="loyo", primary_metric="rpss",
                                verbose=False, progress=False)

    sw_mme = africas2s.ensemble([best1, best2], synthetic_obs, strategy="skill_weighted")
    assert sw_mme.forecast.dims == ("member", "lat", "lon")
    assert not np.all(np.isnan(sw_mme.forecast.values))

    # drop_worst with two members and n_drop=1 reduces to "keep best member only".
    dw_mme = africas2s.ensemble([best1, best2], synthetic_obs, strategy="drop_worst")
    assert dw_mme.forecast.dims == ("member", "lat", "lon")
    # Result equals whichever single forecast had the higher score.
    winner = best1.forecast if best1.score >= best2.score else best2.forecast
    np.testing.assert_allclose(dw_mme.forecast.values, winner.values, equal_nan=True)


def test_e2e_bma_through_optimize(
    synthetic_gcm_hindcast, synthetic_gcm_hindcast2, synthetic_obs,
):
    """Integration: BMA strategy with hindcasts pulled from the actual fits."""
    import africas2s
    from africas2s.methods.bcsd import BCSDMethod

    # Generate per-member hindcasts on the obs grid.
    hindcasts = []
    for gcm in (synthetic_gcm_hindcast, synthetic_gcm_hindcast2):
        m = BCSDMethod()
        m.fit(gcm, synthetic_obs)
        # Predict on each year of the GCM hindcast, ensemble-mean across members.
        preds = []
        for yr in synthetic_obs.year.values:
            pred = m.predict(gcm.sel(year=yr)).mean("member")
            preds.append(pred.expand_dims(year=[yr]))
        hindcasts.append(xr.concat(preds, dim="year"))

    best1 = africas2s.optimize(synthetic_gcm_hindcast, synthetic_obs,
                                methods=["bcsd"], cv="loyo", primary_metric="rpss",
                                verbose=False, progress=False)
    best2 = africas2s.optimize(synthetic_gcm_hindcast2, synthetic_obs,
                                methods=["bcsd"], cv="loyo", primary_metric="rpss",
                                verbose=False, progress=False)

    mme = africas2s.ensemble(
        [best1, best2], synthetic_obs, strategy="bma", hindcasts=hindcasts,
    )
    assert mme.forecast.dims == ("member", "lat", "lon")
    assert not np.all(np.isnan(mme.forecast.values))


def test_e2e_climatology_baseline(climatology_forecast, synthetic_obs):
    import africas2s
    report = africas2s.skill(climatology_forecast, synthetic_obs, metrics=["rpss"])
    np.testing.assert_allclose(report.scores["rpss"], 0.0, atol=0.05)


# ===================================================================
# 14. Plugin contract tests
# ===================================================================

def test_plugin_method_contract(synthetic_gcm_hindcast, synthetic_gcm_forecast, synthetic_obs):
    from africas2s.methods.base import MethodBase
    from africas2s.registry import register_method
    import africas2s

    @register_method("plugin_dummy")
    class DummyMethod(MethodBase):
        def fit(self, hindcast, obs, **kwargs):
            self.obs_mean_ = obs.mean("year")
            self.obs_coords_ = {"lat": obs.lat, "lon": obs.lon}

        def predict(self, forecast, **kwargs):
            result = self.obs_mean_.expand_dims(member=forecast.member)
            return result

    result = africas2s.downscale(synthetic_gcm_hindcast, synthetic_obs, method="plugin_dummy")
    assert result.dims == ("member", "lat", "lon")


def test_plugin_metric_contract(climatology_forecast, synthetic_obs):
    from africas2s.metrics.base import MetricBase
    from africas2s.registry import register_metric
    import africas2s

    @register_metric("always_half")
    class AlwaysHalf(MetricBase):
        def compute(self, forecast, obs, **kwargs):
            return 0.5

    report = africas2s.skill(climatology_forecast, synthetic_obs, metrics=["always_half"])
    assert report.scores["always_half"] == 0.5
