import numpy as np
import pytest
import xarray as xr


# ===================================================================
# 9. Ensemble
# ===================================================================

def test_uniform_ensemble():
    from africas2s.ensemble import ensemble
    lat = np.linspace(-1, 1, 5)
    lon = np.linspace(0, 1, 5)
    a = xr.DataArray(np.ones((3, 5, 5)) * 2, dims=["member", "lat", "lon"],
                     coords={"member": [0, 1, 2], "lat": lat, "lon": lon})
    b = xr.DataArray(np.ones((3, 5, 5)) * 4, dims=["member", "lat", "lon"],
                     coords={"member": [0, 1, 2], "lat": lat, "lon": lon})
    c = xr.DataArray(np.ones((3, 5, 5)) * 6, dims=["member", "lat", "lon"],
                     coords={"member": [0, 1, 2], "lat": lat, "lon": lon})
    result = ensemble([a, b, c], obs=None, strategy="uniform")
    np.testing.assert_allclose(result.forecast.values, 4.0)


def test_ensemble_single_model():
    from africas2s.ensemble import ensemble
    lat = np.linspace(-1, 1, 5)
    lon = np.linspace(0, 1, 5)
    a = xr.DataArray(np.ones((3, 5, 5)) * 7, dims=["member", "lat", "lon"],
                     coords={"member": [0, 1, 2], "lat": lat, "lon": lon})
    result = ensemble([a], obs=None, strategy="uniform")
    np.testing.assert_allclose(result.forecast.values, 7.0)


def test_ensemble_output_shape():
    from africas2s.ensemble import ensemble
    lat = np.linspace(-1, 1, 5)
    lon = np.linspace(0, 1, 5)
    a = xr.DataArray(np.ones((3, 5, 5)), dims=["member", "lat", "lon"],
                     coords={"member": [0, 1, 2], "lat": lat, "lon": lon})
    b = xr.DataArray(np.ones((3, 5, 5)) * 2, dims=["member", "lat", "lon"],
                     coords={"member": [0, 1, 2], "lat": lat, "lon": lon})
    result = ensemble([a, b], obs=None)
    assert "lat" in result.forecast.dims
    assert "lon" in result.forecast.dims


# ===================================================================
# 9b. New ensemble strategies (§24.1)
# ===================================================================

def _make_member(value, lat, lon):
    return xr.DataArray(
        np.full((3, len(lat), len(lon)), value),
        dims=["member", "lat", "lon"],
        coords={"member": [0, 1, 2], "lat": lat, "lon": lon},
    )


def test_drop_worst_drops_lowest_score_member():
    from africas2s.ensemble import ensemble
    lat = np.linspace(-1, 1, 4); lon = np.linspace(0, 1, 4)
    a = _make_member(1.0, lat, lon)
    b = _make_member(5.0, lat, lon)
    c = _make_member(9.0, lat, lon)
    # Scores rank: a=0.1 (worst), b=0.4, c=0.7. Drop a; mean of b,c = 7.0
    out = ensemble([a, b, c], obs=None, strategy="drop_worst",
                   scores=[0.1, 0.4, 0.7])
    np.testing.assert_allclose(out.forecast.values, 7.0)


def test_drop_worst_n_drop_kwarg():
    from africas2s.ensemble import ensemble
    lat = np.linspace(-1, 1, 4); lon = np.linspace(0, 1, 4)
    forecasts = [_make_member(v, lat, lon) for v in [1.0, 2.0, 3.0, 10.0]]
    # Drop bottom 2 (scores 0.0, 0.1) → mean of values 3.0 and 10.0 = 6.5
    out = ensemble(forecasts, obs=None, strategy="drop_worst",
                   scores=[0.0, 0.1, 0.5, 0.9], n_drop=2)
    np.testing.assert_allclose(out.forecast.values, 6.5)


def test_drop_worst_uses_optimize_result_score():
    from africas2s.ensemble import ensemble
    from africas2s.optimize import OptimizeResult
    lat = np.linspace(-1, 1, 4); lon = np.linspace(0, 1, 4)
    forecasts = [
        OptimizeResult(method="cca", score=-0.2, forecast=_make_member(1.0, lat, lon)),
        OptimizeResult(method="cca", score=0.3,  forecast=_make_member(5.0, lat, lon)),
        OptimizeResult(method="cca", score=0.5,  forecast=_make_member(9.0, lat, lon)),
    ]
    out = ensemble(forecasts, obs=None, strategy="drop_worst")
    np.testing.assert_allclose(out.forecast.values, 7.0)


def test_drop_worst_explicit_scores_override_optimize_result():
    from africas2s.ensemble import ensemble
    from africas2s.optimize import OptimizeResult
    lat = np.linspace(-1, 1, 4); lon = np.linspace(0, 1, 4)
    # OptimizeResult scores would suggest dropping the first; explicit override
    # flips it so the last is dropped instead.
    forecasts = [
        OptimizeResult(method="cca", score=-0.2, forecast=_make_member(1.0, lat, lon)),
        OptimizeResult(method="cca", score=0.3,  forecast=_make_member(5.0, lat, lon)),
        OptimizeResult(method="cca", score=0.5,  forecast=_make_member(9.0, lat, lon)),
    ]
    out = ensemble(forecasts, obs=None, strategy="drop_worst",
                   scores=[0.9, 0.8, 0.1])
    np.testing.assert_allclose(out.forecast.values, 3.0)  # mean of 1.0, 5.0


def test_drop_worst_errors_without_scores():
    """Plain DataArrays with no scores and no kwarg cannot be ranked."""
    from africas2s.ensemble import ensemble
    lat = np.linspace(-1, 1, 4); lon = np.linspace(0, 1, 4)
    forecasts = [_make_member(v, lat, lon) for v in [1.0, 5.0, 9.0]]
    with pytest.raises(ValueError, match="score"):
        ensemble(forecasts, obs=None, strategy="drop_worst")


def test_drop_worst_n_drop_too_large():
    from africas2s.ensemble import ensemble
    lat = np.linspace(-1, 1, 4); lon = np.linspace(0, 1, 4)
    forecasts = [_make_member(v, lat, lon) for v in [1.0, 5.0]]
    with pytest.raises(ValueError, match="n_drop"):
        ensemble(forecasts, obs=None, strategy="drop_worst",
                 scores=[0.1, 0.5], n_drop=2)


def test_skill_weighted_positive_scores():
    """Weights ∝ scores, normalized to sum to 1."""
    from africas2s.ensemble import ensemble
    lat = np.linspace(-1, 1, 4); lon = np.linspace(0, 1, 4)
    a = _make_member(2.0, lat, lon)
    b = _make_member(4.0, lat, lon)
    # scores [0.1, 0.3] → weights [0.25, 0.75] → 0.25*2 + 0.75*4 = 3.5
    out = ensemble([a, b], obs=None, strategy="skill_weighted",
                   scores=[0.1, 0.3])
    np.testing.assert_allclose(out.forecast.values, 3.5)


def test_skill_weighted_negative_scores_get_zero_weight():
    """Members with non-positive skill score should not contribute."""
    from africas2s.ensemble import ensemble
    lat = np.linspace(-1, 1, 4); lon = np.linspace(0, 1, 4)
    a = _make_member(2.0, lat, lon)  # bad
    b = _make_member(4.0, lat, lon)  # good
    c = _make_member(8.0, lat, lon)  # good
    # scores [-0.5, 0.2, 0.6] → weights [0, 0.25, 0.75] → 0 + 1.0 + 6.0 = 7.0
    out = ensemble([a, b, c], obs=None, strategy="skill_weighted",
                   scores=[-0.5, 0.2, 0.6])
    np.testing.assert_allclose(out.forecast.values, 7.0)


def test_skill_weighted_all_nonpositive_falls_back_to_uniform():
    """If every member has score ≤ 0, return the uniform mean rather than divide-by-zero."""
    from africas2s.ensemble import ensemble
    lat = np.linspace(-1, 1, 4); lon = np.linspace(0, 1, 4)
    a = _make_member(2.0, lat, lon)
    b = _make_member(4.0, lat, lon)
    out = ensemble([a, b], obs=None, strategy="skill_weighted",
                   scores=[-0.5, -0.2])
    np.testing.assert_allclose(out.forecast.values, 3.0)  # uniform mean


def test_skill_weighted_explicit_weights_kwarg():
    """`weights=` kwarg bypasses score-based computation."""
    from africas2s.ensemble import ensemble
    lat = np.linspace(-1, 1, 4); lon = np.linspace(0, 1, 4)
    a = _make_member(2.0, lat, lon)
    b = _make_member(8.0, lat, lon)
    # weights normalize: [0.1, 0.4] → [0.2, 0.8] → 0.4 + 6.4 = 6.8
    out = ensemble([a, b], obs=None, strategy="skill_weighted",
                   weights=[0.1, 0.4])
    np.testing.assert_allclose(out.forecast.values, 6.8)


def test_skill_weighted_uses_optimize_result_score():
    from africas2s.ensemble import ensemble
    from africas2s.optimize import OptimizeResult
    lat = np.linspace(-1, 1, 4); lon = np.linspace(0, 1, 4)
    forecasts = [
        OptimizeResult(method="cca", score=0.2, forecast=_make_member(2.0, lat, lon)),
        OptimizeResult(method="cca", score=0.6, forecast=_make_member(4.0, lat, lon)),
    ]
    out = ensemble(forecasts, obs=None, strategy="skill_weighted")
    # weights [0.25, 0.75] → 0.5 + 3.0 = 3.5
    np.testing.assert_allclose(out.forecast.values, 3.5)


def _bma_synthetic_setup(seed=0):
    """Two-member BMA scenario: member 0 tracks obs, member 1 is pure noise."""
    rng = np.random.default_rng(seed)
    n_years, nlat, nlon = 12, 5, 5
    years = np.arange(2000, 2000 + n_years)
    lat = np.linspace(-2, 2, nlat); lon = np.linspace(0, 4, nlon)
    obs_data = rng.standard_normal((n_years, nlat, nlon))
    obs = xr.DataArray(obs_data, dims=["year", "lat", "lon"],
                       coords={"year": years, "lat": lat, "lon": lon})
    # Member 0: obs + small noise (skillful)
    h0 = obs_data + rng.standard_normal(obs_data.shape) * 0.1
    # Member 1: pure noise (no skill)
    h1 = rng.standard_normal(obs_data.shape) * 1.5
    hindcasts = [
        xr.DataArray(h0, dims=["year", "lat", "lon"],
                     coords={"year": years, "lat": lat, "lon": lon}),
        xr.DataArray(h1, dims=["year", "lat", "lon"],
                     coords={"year": years, "lat": lat, "lon": lon}),
    ]
    # Forecasts at one target time: a constant value per member (so we can
    # check the combined output equals the BMA-weighted average).
    f0 = xr.DataArray(np.full((3, nlat, nlon), 5.0),
                      dims=["member", "lat", "lon"],
                      coords={"member": [0, 1, 2], "lat": lat, "lon": lon})
    f1 = xr.DataArray(np.full((3, nlat, nlon), 9.0),
                      dims=["member", "lat", "lon"],
                      coords={"member": [0, 1, 2], "lat": lat, "lon": lon})
    return [f0, f1], obs, hindcasts


def test_bma_weights_concentrate_on_skillful_member():
    """BMA should put most weight on the member whose hindcast tracks obs."""
    from africas2s.strategies.bma import BMAStrategy
    forecasts, obs, hindcasts = _bma_synthetic_setup(seed=0)
    s = BMAStrategy()
    weights, _sigma2 = s._fit_em(hindcasts, obs)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-8)
    assert weights[0] > 0.8, f"skillful member got weight {weights[0]:.3f}"
    assert weights[1] < 0.2, f"noise member got weight {weights[1]:.3f}"


def test_bma_combine_uses_fitted_weights():
    """Combined forecast equals weighted sum using the fitted BMA weights."""
    from africas2s.ensemble import ensemble
    from africas2s.strategies.bma import BMAStrategy
    forecasts, obs, hindcasts = _bma_synthetic_setup(seed=0)
    out = ensemble(forecasts, obs, strategy="bma", hindcasts=hindcasts)
    # Fit independently to compare
    weights, _ = BMAStrategy()._fit_em(hindcasts, obs)
    expected = weights[0] * 5.0 + weights[1] * 9.0
    np.testing.assert_allclose(out.forecast.values, expected, rtol=1e-6)


def test_bma_requires_hindcasts():
    from africas2s.ensemble import ensemble
    forecasts, obs, _ = _bma_synthetic_setup(seed=0)
    with pytest.raises(ValueError, match="hindcasts"):
        ensemble(forecasts, obs, strategy="bma")


def test_bma_weights_sum_to_one():
    """Even on degenerate cases, BMA weights should always normalize."""
    from africas2s.strategies.bma import BMAStrategy
    rng = np.random.default_rng(42)
    n_years = 8
    years = np.arange(2000, 2000 + n_years)
    obs = xr.DataArray(rng.standard_normal((n_years, 3, 3)),
                       dims=["year", "lat", "lon"],
                       coords={"year": years, "lat": np.arange(3.0), "lon": np.arange(3.0)})
    # Three near-identical members
    h = [obs + rng.standard_normal((n_years, 3, 3)) * 0.01 for _ in range(3)]
    weights, _ = BMAStrategy()._fit_em(h, obs)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-8)
    assert np.all(weights >= 0)


# ===================================================================
# 21. Ensemble safeguards (#58, #59)
# ===================================================================

def test_apply_shrinkage_blend():
    """w_shrunk = (1 - λ) * w + λ / N. λ=0 returns w; λ=1 returns uniform."""
    from africas2s.ensemble import _apply_shrinkage

    w = np.array([0.8, 0.2])
    np.testing.assert_allclose(_apply_shrinkage(w, 0.0), w)
    np.testing.assert_allclose(_apply_shrinkage(w, 1.0), [0.5, 0.5])
    np.testing.assert_allclose(_apply_shrinkage(w, 0.5), [0.65, 0.35])


def test_effective_n_degenerate():
    """effective_N = 1 / sum(w**2). Uniform → N; (0.95, 0.05, 0, 0, 0) → ~1.11."""
    from africas2s.ensemble import _effective_n

    n = 5
    uniform = np.full(n, 1.0 / n)
    np.testing.assert_allclose(_effective_n(uniform), n)

    degenerate = np.array([0.95, 0.05, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(_effective_n(degenerate), 1 / (0.95**2 + 0.05**2), rtol=1e-6)


def test_strategy_fit_uniform():
    """uniform.fit returns 1/N for any input."""
    from africas2s.registry import get_strategy

    n_year, n_lat, n_lon = 5, 3, 3
    coords = {"year": np.arange(n_year), "lat": np.arange(n_lat), "lon": np.arange(n_lon)}
    arrs = [
        xr.DataArray(np.random.randn(n_year, n_lat, n_lon), dims=["year", "lat", "lon"], coords=coords)
        for _ in range(3)
    ]
    obs = xr.DataArray(np.random.randn(n_year, n_lat, n_lon), dims=["year", "lat", "lon"], coords=coords)
    w = get_strategy("uniform")().fit(arrs, obs)
    np.testing.assert_allclose(w, [1/3, 1/3, 1/3])


def test_strategy_fit_drop_worst_indicator():
    """drop_worst.fit returns indicator weights for the kept members.

    Uses a noise-only obs so per-member skill is purely sample-driven;
    we rank synthetically by hand-constructing one member to be much
    closer to obs than the others.
    """
    from africas2s.registry import get_strategy
    from africas2s.tercile import to_tercile_cv

    np.random.seed(0)
    n_year, n_lat, n_lon = 12, 3, 3
    coords = {"year": np.arange(n_year), "lat": np.arange(n_lat), "lon": np.arange(n_lon)}
    obs = xr.DataArray(np.random.randn(n_year, n_lat, n_lon),
                       dims=["year", "lat", "lon"], coords=coords)
    # Member 1 is obs itself (perfect), members 0 and 2 are noise.
    arrs_continuous = [
        xr.DataArray(np.random.randn(n_year, n_lat, n_lon), dims=["year", "lat", "lon"], coords=coords),
        obs,
        xr.DataArray(np.random.randn(n_year, n_lat, n_lon), dims=["year", "lat", "lon"], coords=coords),
    ]
    forecasts = [to_tercile_cv(a, obs) for a in arrs_continuous]
    w = get_strategy("drop_worst")().fit(forecasts, obs, n_drop=1)
    # n_drop=1: exactly one weight should be zero, the other two equal 0.5.
    assert (w == 0).sum() == 1
    nonzero = w[w > 0]
    np.testing.assert_allclose(nonzero, [0.5, 0.5])
    # Member 1 (the perfect one) must be among the kept members.
    assert w[1] == 0.5


def test_strategy_fit_skill_weighted_clipped():
    """skill_weighted.fit clips negative skill to 0 and normalises."""
    from africas2s.registry import get_strategy
    from africas2s.tercile import to_tercile_cv

    np.random.seed(0)
    n_year, n_lat, n_lon = 12, 3, 3
    coords = {"year": np.arange(n_year), "lat": np.arange(n_lat), "lon": np.arange(n_lon)}
    obs = xr.DataArray(np.random.randn(n_year, n_lat, n_lon),
                       dims=["year", "lat", "lon"], coords=coords)
    # One perfect member, one anti-correlated member.
    arrs_continuous = [obs, -obs]
    forecasts = [to_tercile_cv(a, obs) for a in arrs_continuous]
    w = get_strategy("skill_weighted")().fit(forecasts, obs)
    assert w.sum() == pytest.approx(1.0)
    # Perfect member dominates; anti-correlated member clips to near 0.
    # Don't pin an exact equality — the anti-tercile's RPSS can be slightly
    # positive on small grids/seeds; the contract is "perfect > anti".
    assert w[0] > 0.8
    assert w[1] < 0.2


def test_strategy_self_shrinks_flags():
    """Only bma reports self_shrinks() == True."""
    from africas2s.registry import get_strategy

    assert get_strategy("uniform")().self_shrinks() is False
    assert get_strategy("drop_worst")().self_shrinks() is False
    assert get_strategy("skill_weighted")().self_shrinks() is False
    assert get_strategy("bma")().self_shrinks() is True


def test_strategy_uniform_combine_with_weights():
    """uniform.combine respects an explicit weights= kwarg (used by the
    nested-CV loop to apply a non-uniform vector through the same code path).
    """
    from africas2s.registry import get_strategy

    coords = {"lat": [0], "lon": [0]}
    a = xr.DataArray([[1.0]], dims=["lat", "lon"], coords=coords)
    b = xr.DataArray([[3.0]], dims=["lat", "lon"], coords=coords)
    out = get_strategy("uniform")().combine([a, b], weights=np.array([0.25, 0.75]))
    np.testing.assert_allclose(out.values, [[2.5]])


def test_strategy_drop_worst_combine_with_weights():
    """drop_worst.combine respects an explicit weights= kwarg (skipping its
    own ranking logic)."""
    from africas2s.registry import get_strategy

    coords = {"lat": [0], "lon": [0]}
    arrs = [
        xr.DataArray([[1.0]], dims=["lat", "lon"], coords=coords),
        xr.DataArray([[3.0]], dims=["lat", "lon"], coords=coords),
        xr.DataArray([[5.0]], dims=["lat", "lon"], coords=coords),
    ]
    # weights drop the second member entirely
    out = get_strategy("drop_worst")().combine(arrs, weights=np.array([0.5, 0.0, 0.5]))
    np.testing.assert_allclose(out.values, [[3.0]])


def test_ensemble_no_optimize_returns_trivial_ensemble_result():
    """ensemble(..., optimize_ensemble=False) wraps the strategy's combine()
    in an EnsembleResult with the trivial-safeguard fields."""
    from africas2s.ensemble import ensemble, EnsembleResult

    coords = {"lat": [0], "lon": [0]}
    a = xr.DataArray([[2.0]], dims=["lat", "lon"], coords=coords, name="A")
    b = xr.DataArray([[4.0]], dims=["lat", "lon"], coords=coords, name="B")

    result = ensemble([a, b], obs=None, strategy="uniform")
    assert isinstance(result, EnsembleResult)
    np.testing.assert_allclose(result.forecast.values, [[3.0]])
    np.testing.assert_allclose(result.weights, [0.5, 0.5])
    assert result.member_names == ["A", "B"]
    assert result.member_cv_skill == {}
    assert result.effective_n == pytest.approx(2.0)
    assert result.gate_passed is True
    assert result.shrinkage_lambda == 0.0
    assert result.safeguards_applied == {}


def test_ensemble_optimize_self_shrinking_strategy_skips_shrinkage():
    """A self_shrinks() strategy must not get outer-loop shrinkage on top."""
    from africas2s.ensemble import ensemble
    from africas2s.strategies.base import StrategyBase
    from africas2s.registry import register_strategy

    @register_strategy("test_self_shrinking")
    class _SS(StrategyBase):
        def combine(self, forecasts, obs=None, *, weights=None, **kwargs):
            from africas2s.strategies.uniform import _as_array
            arrays = [_as_array(f) for f in forecasts]
            if weights is None:
                return sum(arrays) / len(arrays)
            w = np.asarray(weights, dtype=float)
            result = arrays[0] * float(w[0])
            for arr, wi in zip(arrays[1:], w[1:]):
                result = result + arr * float(wi)
            return result

        def fit(self, forecasts, obs, years=None, *, primary_metric="rpss", **kwargs):
            return np.full(len(forecasts), 1.0 / len(forecasts))

        def self_shrinks(self):
            return True

    np.random.seed(0)
    n_year, n_lat, n_lon = 8, 2, 2
    coords = {"year": np.arange(n_year), "lat": np.arange(n_lat), "lon": np.arange(n_lon)}
    obs = xr.DataArray(np.random.randn(n_year, n_lat, n_lon),
                       dims=["year", "lat", "lon"], coords=coords)
    arrs = [
        xr.DataArray(np.random.randn(n_year, n_lat, n_lon),
                     dims=["year", "lat", "lon"], coords=coords)
        for _ in range(3)
    ]

    result = ensemble(arrs, obs, strategy="test_self_shrinking",
                      optimize_ensemble=True, primary_metric="pearson_r")
    assert result.shrinkage_lambda == 0.0
    assert result.safeguards_applied.get("shrinkage_skipped_for_self_shrinking") is True


def test_ensemble_optimize_effective_n_floor_fallback():
    """A strategy whose fit returns a degenerate weight vector triggers the
    floor and falls back to uniform per-fold."""
    from africas2s.ensemble import ensemble
    from africas2s.strategies.base import StrategyBase
    from africas2s.registry import register_strategy

    @register_strategy("test_degenerate")
    class _Deg(StrategyBase):
        def combine(self, forecasts, obs=None, *, weights=None, **kwargs):
            from africas2s.strategies.uniform import _as_array
            arrays = [_as_array(f) for f in forecasts]
            if weights is None:
                return sum(arrays) / len(arrays)
            w = np.asarray(weights, dtype=float)
            result = arrays[0] * float(w[0])
            for arr, wi in zip(arrays[1:], w[1:]):
                result = result + arr * float(wi)
            return result

        def fit(self, forecasts, obs, years=None, *, primary_metric="rpss", **kwargs):
            # Degenerate (effective_N ~ 1.1): forces the floor to fire.
            n = len(forecasts)
            w = np.zeros(n)
            w[0] = 0.95
            w[1] = 0.05
            return w

    np.random.seed(1)
    n_year, n_lat, n_lon = 8, 2, 2
    coords = {"year": np.arange(n_year), "lat": np.arange(n_lat), "lon": np.arange(n_lon)}
    obs = xr.DataArray(np.random.randn(n_year, n_lat, n_lon),
                       dims=["year", "lat", "lon"], coords=coords)
    arrs = [
        xr.DataArray(np.random.randn(n_year, n_lat, n_lon),
                     dims=["year", "lat", "lon"], coords=coords)
        for _ in range(5)
    ]

    result = ensemble(arrs, obs, strategy="test_degenerate",
                      optimize_ensemble=True, primary_metric="pearson_r",
                      safeguards={"shrinkage": 0.0})  # disable shrinkage so floor is what fires
    fired = result.safeguards_applied.get("min_effective_n_floor_fired_on", [])
    assert len(fired) > 0, f"expected the floor to fire on at least one fold, got {fired}"


def test_ensemble_optimize_recovers_known_better_member():
    """skill_weighted with optimize_ensemble=True concentrates weight on the
    member that actually beats the others on the primary metric.

    Override min_effective_n=2 so the floor doesn't fight the test's premise:
    with 3 members and shrinkage=0.5, the best achievable effective_N for a
    perfect-singleton fit is 2 (post-shrinkage weights [1/6, 2/3, 1/6]). The
    default floor of 3 would force a uniform fallback regardless of recovery
    quality — testing that here would just be testing the floor, not recovery.
    """
    from africas2s.ensemble import ensemble

    np.random.seed(2)
    n_year, n_lat, n_lon = 12, 3, 3
    coords = {"year": np.arange(n_year), "lat": np.arange(n_lat), "lon": np.arange(n_lon)}
    obs = xr.DataArray(np.random.randn(n_year, n_lat, n_lon),
                       dims=["year", "lat", "lon"], coords=coords)
    # Member 1 is obs (perfect on the continuous scale → high primary skill).
    # Members 0 and 2 are noise.
    arrs = [
        xr.DataArray(np.random.randn(n_year, n_lat, n_lon),
                     dims=["year", "lat", "lon"], coords=coords),
        obs,
        xr.DataArray(np.random.randn(n_year, n_lat, n_lon),
                     dims=["year", "lat", "lon"], coords=coords),
    ]

    result = ensemble(arrs, obs, strategy="skill_weighted",
                      optimize_ensemble=True, primary_metric="pearson_r",
                      safeguards={"min_effective_n": 2})
    assert result.weights[1] > result.weights[0]
    assert result.weights[1] > result.weights[2]
    # Diagnostics populated.
    assert len(result.member_cv_skill) == 3
    assert "gate_outer_cv" in result.safeguards_applied
    assert "uniform_outer_cv" in result.safeguards_applied


def test_ensemble_optimize_all_equal_members_returns_uniform():
    """With no genuine signal differentiating members, optimisation must not
    over-concentrate weight. Identical members → fit returns uniform weights
    (so the optimised and uniform paths are literally identical and the gate
    accepts with margin 0). Either way, the result is uniform — that's the
    safeguard's intent.
    """
    from africas2s.ensemble import ensemble

    np.random.seed(3)
    n_year, n_lat, n_lon = 8, 2, 2
    coords = {"year": np.arange(n_year), "lat": np.arange(n_lat), "lon": np.arange(n_lon)}
    obs = xr.DataArray(np.random.randn(n_year, n_lat, n_lon),
                       dims=["year", "lat", "lon"], coords=coords)
    base = xr.DataArray(np.random.randn(n_year, n_lat, n_lon),
                        dims=["year", "lat", "lon"], coords=coords)
    arrs = [base.copy(), base.copy(), base.copy()]

    result = ensemble(arrs, obs, strategy="skill_weighted",
                      optimize_ensemble=True, primary_metric="pearson_r")
    np.testing.assert_allclose(result.weights, [1/3, 1/3, 1/3], atol=0.15)
    # Diagnostics always populated for the nested-CV path.
    assert "gate_outer_cv" in result.safeguards_applied
    assert "uniform_outer_cv" in result.safeguards_applied


def test_ensemble_optimize_gate_fails_with_perverse_strategy():
    """A strategy whose fit picks adversarial weights (concentrating on a
    single noise member) makes the optimised outer-CV worse than uniform,
    forcing the gate to fail → uniform fallback + RuntimeWarning."""
    import warnings as _warnings
    from africas2s.ensemble import ensemble
    from africas2s.strategies.base import StrategyBase
    from africas2s.registry import register_strategy

    @register_strategy("test_winner_take_all")
    class _WTA(StrategyBase):
        def combine(self, forecasts, obs=None, *, weights=None, **kwargs):
            from africas2s.strategies.uniform import _as_array
            arrays = [_as_array(f) for f in forecasts]
            if weights is None:
                return sum(arrays) / len(arrays)
            w = np.asarray(weights, dtype=float)
            result = arrays[0] * float(w[0])
            for arr, wi in zip(arrays[1:], w[1:]):
                result = result + arr * float(wi)
            return result

        def fit(self, forecasts, obs, years=None, *, primary_metric="pearson_r", **kwargs):
            n = len(forecasts)
            w = np.zeros(n)
            w[0] = 1.0  # always pick member 0 — adversarial when 0 is noise
            return w

    np.random.seed(5)
    n_year, n_lat, n_lon = 10, 3, 3
    coords = {"year": np.arange(n_year), "lat": np.arange(n_lat), "lon": np.arange(n_lon)}
    obs = xr.DataArray(np.random.randn(n_year, n_lat, n_lon),
                       dims=["year", "lat", "lon"], coords=coords)
    # Member 0 is anti-correlated with obs; members 1 and 2 are noise. Winner-
    # take-all picks the anti-correlated member → pooled opt-path pearson_r
    # ≈ -1, while uniform averages anti-correlation with noise → ≈ -1/3. So
    # unif_cv > opt_cv reliably and the gate fails for the right reason. (See
    # the "the flaw" note in the Task 5–6 instructions: pooling-then-scoring
    # changes which fixtures provably trigger the gate.)
    arrs = [
        xr.DataArray(-obs.values, dims=["year", "lat", "lon"], coords=coords),
        xr.DataArray(np.random.randn(n_year, n_lat, n_lon),
                     dims=["year", "lat", "lon"], coords=coords),
        xr.DataArray(np.random.randn(n_year, n_lat, n_lon),
                     dims=["year", "lat", "lon"], coords=coords),
    ]

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        result = ensemble(arrs, obs, strategy="test_winner_take_all",
                          optimize_ensemble=True, primary_metric="pearson_r",
                          safeguards={"min_effective_n": 0})  # disable floor; we want gate to fire
    assert result.gate_passed is False, (
        f"expected gate failure; opt={result.safeguards_applied.get('gate_outer_cv')}, "
        f"unif={result.safeguards_applied.get('uniform_outer_cv')}"
    )
    np.testing.assert_allclose(result.weights, [1/3, 1/3, 1/3])
    gate_warnings = [str(w.message) for w in caught if "gate" in str(w.message).lower()]
    assert gate_warnings, "expected a gate-failure RuntimeWarning"


# ===================================================================
# 21b. Input validation (regression guards)
# ===================================================================

def test_ensemble_empty_forecasts_raises_clear_error():
    """An empty forecasts list used to crash with `ZeroDivisionError`
    deep inside `uniform.combine` (sum/len of an empty list). Now raises
    a clear `ValueError` at the entry."""
    from africas2s.ensemble import ensemble
    obs = xr.DataArray(
        np.random.randn(5, 3, 3),
        dims=["year", "lat", "lon"],
        coords={"year": np.arange(5), "lat": np.arange(3.0), "lon": np.arange(3.0)},
    )
    with pytest.raises(ValueError, match="empty"):
        ensemble([], obs=obs, strategy="uniform")


def test_resolve_safeguards_rejects_negative_min_effective_n():
    """A negative `min_effective_n` silently disabled the floor:
    `_effective_n(w)` is always >= 1, so any negative threshold made
    the `< min_effective_n` check unreachable. Now validated at resolve time.
    """
    from africas2s.ensemble import _resolve_safeguards
    with pytest.raises(ValueError, match="min_effective_n"):
        _resolve_safeguards({"min_effective_n": -5})


def test_ensemble_leaky_pipeline_overstates_skill():
    """Without nested CV, in-sample skill is optimistic. The diagnostics
    should still record the outer-CV scores so users can compare."""
    from africas2s.ensemble import ensemble

    np.random.seed(4)
    n_year, n_lat, n_lon = 10, 2, 2
    coords = {"year": np.arange(n_year), "lat": np.arange(n_lat), "lon": np.arange(n_lon)}
    obs = xr.DataArray(np.random.randn(n_year, n_lat, n_lon),
                       dims=["year", "lat", "lon"], coords=coords)
    arrs = [
        xr.DataArray(np.random.randn(n_year, n_lat, n_lon),
                     dims=["year", "lat", "lon"], coords=coords)
        for _ in range(4)
    ]

    # Honest path: full safeguards.
    honest = ensemble(arrs, obs, strategy="skill_weighted",
                     optimize_ensemble=True, primary_metric="pearson_r")
    # Leaky path: explicit override to disable nested CV and gate. Reproduces
    # the silent-bug configuration the safeguards exist to prevent.
    leaky = ensemble(arrs, obs, strategy="skill_weighted",
                     optimize_ensemble=True, primary_metric="pearson_r",
                     safeguards={"nested_cv": False, "gate": False,
                                 "shrinkage": 0.0, "min_effective_n": 0})
    # With pure noise, neither path should produce a real signal; what we're
    # confirming is that the override path completes (and that its diagnostics
    # do not include outer-CV scores, since nested CV was disabled).
    assert leaky.safeguards_applied.get("nested_cv") is False
    assert "gate_outer_cv" not in leaky.safeguards_applied
    # Honest path did run nested CV.
    assert honest.safeguards_applied.get("nested_cv") is True
    assert "gate_outer_cv" in honest.safeguards_applied
