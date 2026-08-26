"""Unit tests for `EnsembleResult.member_contributions`."""
import numpy as np
import pytest
import xarray as xr

from africas2s import ensemble


def _gridded(values, *, years, name):
    """Wrap (year, lat, lon) values into a DataArray with named coords."""
    n_year, n_lat, n_lon = values.shape
    return xr.DataArray(
        values,
        dims=("year", "lat", "lon"),
        coords={
            "year": list(years),
            "lat": np.linspace(-5.0, 5.0, n_lat),
            "lon": np.linspace(30.0, 40.0, n_lon),
        },
        name=name,
    )


def test_identical_members_have_perfect_correlation_and_zero_delta():
    """When every member is identical to obs, MME = obs = each member.
    correlation_with_mme_mean = 1.0, skill_delta = 0 everywhere."""
    rng = np.random.default_rng(0)
    years = list(range(2000, 2020))
    obs_vals = rng.standard_normal((20, 4, 4))
    obs = _gridded(obs_vals, years=years, name="obs")
    m1 = _gridded(obs_vals.copy(), years=years, name="m1")
    m2 = _gridded(obs_vals.copy(), years=years, name="m2")

    result = ensemble([m1, m2], obs, strategy="uniform")
    mc = result.member_contributions
    assert mc is not None
    assert set(mc.keys()) == {"m1", "m2"}
    for name in ("m1", "m2"):
        assert mc[name]["correlation_with_mme_mean"] == pytest.approx(1.0, abs=1e-10)
        sd = mc[name]["skill_delta"]
        assert sd.dims == ("lat", "lon")
        # Member skill = MME skill = 1.0 perfectly correlated; delta ≈ 0.
        np.testing.assert_allclose(sd.values, 0.0, atol=1e-10)


def test_member_equals_mme_has_correlation_one():
    """In a 3-member ensemble where member C = (A + B) / 2, C exactly
    equals the MME mean (since (A+B+C)/3 = (A+B+(A+B)/2)/3 = (A+B)/2 = C).
    C's correlation_with_mme_mean must be 1.0."""
    rng = np.random.default_rng(1)
    years = list(range(2000, 2020))
    obs = _gridded(rng.standard_normal((20, 4, 4)), years=years, name="obs")
    a_vals = rng.standard_normal((20, 4, 4))
    b_vals = rng.standard_normal((20, 4, 4))
    c_vals = (a_vals + b_vals) / 2.0
    A = _gridded(a_vals, years=years, name="A")
    B = _gridded(b_vals, years=years, name="B")
    C = _gridded(c_vals, years=years, name="C")

    result = ensemble([A, B, C], obs, strategy="uniform")
    mc = result.member_contributions
    assert mc["C"]["correlation_with_mme_mean"] == pytest.approx(1.0, abs=1e-10)


def test_skill_delta_sign_convention():
    """High-skill member should have negative skill_delta (it beats the MME);
    low-skill member should have positive skill_delta (MME beats it)."""
    rng = np.random.default_rng(2)
    years = list(range(2000, 2030))
    n = len(years)
    obs_vals = rng.standard_normal((n, 4, 4))
    obs = _gridded(obs_vals, years=years, name="obs")
    # A: nearly perfect (small noise) → high skill.
    A = _gridded(obs_vals + 0.05 * rng.standard_normal((n, 4, 4)),
                 years=years, name="A")
    # B: pure noise → near-zero skill.
    B = _gridded(rng.standard_normal((n, 4, 4)), years=years, name="B")

    result = ensemble([A, B], obs, strategy="uniform")
    mc = result.member_contributions

    # A is better than the MME (which is dragged down by B); skill_delta < 0 on most cells.
    assert float(mc["A"]["skill_delta"].mean()) < 0
    # B is worse than the MME; skill_delta > 0 on most cells.
    assert float(mc["B"]["skill_delta"].mean()) > 0


def test_member_contributions_none_when_obs_is_none():
    """No obs → no honest CV alignment → member_contributions is None."""
    rng = np.random.default_rng(3)
    years = list(range(2000, 2015))
    A = _gridded(rng.standard_normal((15, 4, 4)), years=years, name="A")
    B = _gridded(rng.standard_normal((15, 4, 4)), years=years, name="B")

    result = ensemble([A, B], None, strategy="uniform")
    assert result.member_contributions is None


def test_member_contributions_none_when_year_dim_absent():
    """Single-year forecasts (no year dim) → member_contributions is None."""
    rng = np.random.default_rng(4)
    obs = xr.DataArray(
        rng.standard_normal((4, 4)),
        dims=("lat", "lon"),
        coords={"lat": np.linspace(-5, 5, 4), "lon": np.linspace(30, 40, 4)},
        name="obs",
    )
    A = obs.rename("A").copy()
    B = obs.rename("B").copy()

    result = ensemble([A, B], obs, strategy="uniform")
    assert result.member_contributions is None


def test_member_contributions_none_on_nested_cv_false_path():
    """The optimistic optimize_ensemble path has no honest CV; mirror pev=None."""
    rng = np.random.default_rng(5)
    years = list(range(2000, 2020))
    obs = _gridded(rng.standard_normal((20, 4, 4)), years=years, name="obs")
    A = _gridded(rng.standard_normal((20, 4, 4)), years=years, name="A")
    B = _gridded(rng.standard_normal((20, 4, 4)), years=years, name="B")

    with pytest.warns(RuntimeWarning, match="optimistic"):
        result = ensemble(
            [A, B], obs,
            optimize_ensemble=True, primary_metric="pearson_r",
            safeguards={"nested_cv": False},
        )
    assert result.member_contributions is None


def test_member_contributions_populated_on_nested_cv_path():
    """The nested-CV optimize path uses opt_pooled (honest CV) as the MME
    forecast. Field is populated; each correlation_with_mme_mean is in
    [-1, 1] (or NaN for degenerate fixtures)."""
    rng = np.random.default_rng(6)
    years = list(range(2000, 2025))
    obs = _gridded(rng.standard_normal((25, 4, 4)), years=years, name="obs")
    A = _gridded(
        rng.standard_normal((25, 4, 4)) * 0.6 + obs.values * 0.4,
        years=years, name="A",
    )
    B = _gridded(
        rng.standard_normal((25, 4, 4)) * 0.6 + obs.values * 0.3,
        years=years, name="B",
    )

    result = ensemble([A, B], obs, optimize_ensemble=True,
                      primary_metric="pearson_r")
    mc = result.member_contributions
    assert mc is not None
    assert set(mc.keys()) == {"A", "B"}
    for name in ("A", "B"):
        corr = mc[name]["correlation_with_mme_mean"]
        # In [-1, 1] or NaN; this fixture should give a real value.
        assert np.isnan(corr) or -1.0 <= corr <= 1.0
        assert mc[name]["skill_delta"].dims == ("lat", "lon")
