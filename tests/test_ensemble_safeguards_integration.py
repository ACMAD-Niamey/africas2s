"""End-to-end check for ensemble safeguards on a planted-signal fixture.

Exercises the realistic call path (`ensemble(optimize_ensemble=True)` with
three members, one of which has a planted signal) so composition regressions
surface alongside unit-level changes.
"""
import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def planted_signal_ensemble():
    np.random.seed(13)
    n_year, n_lat, n_lon = 12, 3, 3
    coords = {
        "year": np.arange(n_year),
        "lat": np.arange(n_lat),
        "lon": np.arange(n_lon),
    }
    obs = xr.DataArray(
        np.random.randn(n_year, n_lat, n_lon),
        dims=["year", "lat", "lon"], coords=coords,
    )
    # Three "members":
    # - member 0: pure noise.
    # - member 1: 0.8 * obs + 0.2 * noise -- well-correlated with obs.
    # - member 2: pure noise.
    def noise():
        return np.random.randn(n_year, n_lat, n_lon)

    forecasts = [
        xr.DataArray(noise(), dims=["year", "lat", "lon"], coords=coords, name="noise_a"),
        xr.DataArray(0.8 * obs.values + 0.2 * noise(),
                     dims=["year", "lat", "lon"], coords=coords, name="signal"),
        xr.DataArray(noise(), dims=["year", "lat", "lon"], coords=coords, name="noise_b"),
    ]
    return forecasts, obs


def test_ensemble_safeguards_end_to_end(planted_signal_ensemble):
    """ensemble(optimize_ensemble=True, strategy='skill_weighted',
    primary_metric='pearson_r') recovers the planted-signal member and beats
    the uniform baseline.

    Overrides min_effective_n=2 because with three members and the default
    shrinkage of 0.5, a perfect-singleton fit shrinks to weights
    [1/6, 2/3, 1/6] which gives effective_N = 2 -- below the default floor of
    3. With three members the floor and a meaningful concentration on the
    best member are in tension; tightening to 2 lets recovery happen while
    still rejecting wholly-degenerate winner-take-all fits.
    """
    from africas2s.ensemble import ensemble, EnsembleResult

    forecasts, obs = planted_signal_ensemble
    result = ensemble(
        forecasts, obs,
        strategy="skill_weighted",
        optimize_ensemble=True,
        primary_metric="pearson_r",
        safeguards={"min_effective_n": 2},
    )
    assert isinstance(result, EnsembleResult)
    assert result.gate_passed is True, (
        f"expected gate to pass with a true planted signal; got "
        f"opt={result.safeguards_applied.get('gate_outer_cv')}, "
        f"unif={result.safeguards_applied.get('uniform_outer_cv')}"
    )
    assert result.weights[1] > 0.4, (
        f"expected signal member dominant, got {result.weights}"
    )
    assert len(result.member_cv_skill) == 3
    # pearson_r is higher-is-better -> signal member has the highest score.
    skills = result.member_cv_skill
    assert skills["signal"] == max(skills.values()), (
        f"signal member should have highest pearson_r; got {skills}"
    )
    # Diagnostics: optimised outer-CV should beat uniform outer-CV.
    opt_cv = result.safeguards_applied["gate_outer_cv"]
    unif_cv = result.safeguards_applied["uniform_outer_cv"]
    assert opt_cv > unif_cv, (
        f"expected optimised OOS to beat uniform OOS on the planted signal; "
        f"got opt={opt_cv:.4f}, unif={unif_cv:.4f}"
    )


def test_safeguards_path_populates_member_contributions():
    """The default-safeguards optimize path (nested_cv=True) populates
    member_contributions with one entry per member."""
    import numpy as np
    import xarray as xr
    from africas2s import ensemble

    rng = np.random.default_rng(99)
    n_year = 25
    years = list(range(2000, 2000 + n_year))
    coords = {
        "year": years,
        "lat": np.linspace(-5, 5, 4),
        "lon": np.linspace(30, 40, 4),
    }

    def _da(vals, name):
        return xr.DataArray(vals, dims=("year", "lat", "lon"),
                            coords=coords, name=name)

    obs_vals = rng.standard_normal((n_year, 4, 4))
    obs = _da(obs_vals, "obs")
    A = _da(obs_vals * 0.4 + rng.standard_normal((n_year, 4, 4)) * 0.6, "A")
    B = _da(obs_vals * 0.2 + rng.standard_normal((n_year, 4, 4)) * 0.8, "B")

    result = ensemble([A, B], obs, optimize_ensemble=True,
                      primary_metric="pearson_r")

    mc = result.member_contributions
    assert mc is not None
    assert set(mc.keys()) == {"A", "B"}
    for name in ("A", "B"):
        assert "correlation_with_mme_mean" in mc[name]
        assert "skill_delta" in mc[name]
        assert mc[name]["skill_delta"].dims == ("lat", "lon")
