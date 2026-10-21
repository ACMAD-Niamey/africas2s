import numpy as np
import pytest
import xarray as xr


# ===================================================================
# 8. Tercile conversion
# ===================================================================

def test_to_tercile_cv_default_is_leakage_disciplined():
    """`to_tercile_cv()` defaults to cpt_boundaries=True (CPT reference convention).

    Regression guard against accidentally flipping back to the leaky default —
    the issue (§6.5 / #22) made this the disciplined default.
    """
    import inspect
    from africas2s.tercile import to_tercile_cv
    sig = inspect.signature(to_tercile_cv)
    assert sig.parameters["cpt_boundaries"].default is True


def test_to_tercile_cv_disciplined_and_leaky_paths_diverge():
    """The disciplined (cpt_boundaries=True) and leaky (False) paths must
    produce different tercile probabilities on synthetic data.

    If they ever produce identical results, the boundary path has silently
    been short-circuited and the leakage discipline has lost its teeth.
    """
    from africas2s.tercile import to_tercile_cv
    rng = np.random.default_rng(0)
    n_years = 14
    years = np.arange(2000, 2000 + n_years)
    lat = np.linspace(-2, 2, 5)
    lon = np.linspace(0, 4, 5)
    obs_data = rng.standard_normal((n_years, len(lat), len(lon)))
    cv_data = obs_data * 0.6 + rng.standard_normal((n_years, len(lat), len(lon))) * 0.4
    obs = xr.DataArray(obs_data, dims=["year", "lat", "lon"],
                       coords={"year": years, "lat": lat, "lon": lon})
    cv = xr.DataArray(cv_data, dims=["year", "lat", "lon"],
                      coords={"year": years, "lat": lat, "lon": lon})
    leverages = np.full(n_years, 0.1)

    disciplined = to_tercile_cv(cv, obs, method="cpt", leverages=leverages,
                                cpt_boundaries=True)
    leaky = to_tercile_cv(cv, obs, method="cpt", leverages=leverages,
                          cpt_boundaries=False)

    # Probabilities must differ somewhere (boundaries computed differently).
    assert not np.allclose(disciplined.values, leaky.values, equal_nan=True), (
        "disciplined and leaky paths produced identical probabilities — "
        "the cpt_boundaries flag is no longer affecting behaviour."
    )


def test_to_tercile_cv_leaky_path_still_available():
    """Opt-in leaky behaviour stays accessible for legacy/comparison runs."""
    from africas2s.tercile import to_tercile_cv
    rng = np.random.default_rng(1)
    n_years = 10
    years = np.arange(2000, 2000 + n_years)
    obs = xr.DataArray(
        rng.standard_normal((n_years, 3, 3)),
        dims=["year", "lat", "lon"],
        coords={"year": years, "lat": np.arange(3.0), "lon": np.arange(3.0)},
    )
    cv = xr.DataArray(
        rng.standard_normal((n_years, 3, 3)),
        dims=["year", "lat", "lon"],
        coords={"year": years, "lat": np.arange(3.0), "lon": np.arange(3.0)},
    )
    leverages = np.full(n_years, 0.1)
    out = to_tercile_cv(cv, obs, method="cpt", leverages=leverages,
                        cpt_boundaries=False)
    np.testing.assert_allclose(out.sum("tercile").values, 1.0, atol=1e-6)


def test_continuous_to_tercile(synthetic_gcm_forecast, synthetic_obs):
    from africas2s.tercile import to_tercile
    result = to_tercile(synthetic_gcm_forecast, synthetic_obs)
    assert "tercile" in result.dims
    assert result.dims == ("tercile", "lat", "lon")
    # Probabilities sum to 1
    sums = result.sum("tercile")
    np.testing.assert_allclose(sums.values, 1.0, atol=1e-10)
    # Values in [0, 1]
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


def test_tercile_uniform_from_climatology(synthetic_obs):
    """If forecast matches obs climatology, tercile probs ≈ 1/3."""
    from africas2s.tercile import to_tercile
    # Use obs mean as every member's "forecast"
    clim = synthetic_obs.mean("year")
    members = np.arange(20)  # many members all at climatology
    fcst = clim.expand_dims(member=members)
    result = to_tercile(fcst, synthetic_obs)
    # Since all members equal climatology mean, some pixels will be near 1/3
    mean_probs = result.mean(dim=["lat", "lon"])
    # Each tercile should get roughly 1/3 of the probability
    for i in range(3):
        assert 0.0 <= float(mean_probs.isel(tercile=i)) <= 1.0


# ===================================================================
# Probabilities are dimensionless, whatever the predictand's units
# ===================================================================
#
# xarray >= 2026.2 keeps attrs through comparisons and reductions, so a
# counting step such as ``(forecast < t33).sum("member") / n`` silently
# inherits the predictand's ``units`` ("mm") unless the output is labelled.
# Every tercile-probability constructor carries ``units="1"`` (CF for a
# dimensionless quantity) so agents and unit-aware tools never read a
# probability "in mm".

def _in_mm(da):
    return da.assign_attrs(units="mm", long_name="precipitation")


def test_to_tercile_probabilities_are_dimensionless(synthetic_obs):
    from africas2s.tercile import to_tercile
    obs = _in_mm(synthetic_obs)
    fc = obs.isel(year=slice(0, 5)).rename(year="member")   # ensemble on the obs grid
    for method in ("counting", "gaussian"):
        out = to_tercile(fc, obs, method=method)
        assert out.attrs.get("units") == "1", (method, out.attrs)


def test_to_tercile_cv_probabilities_are_dimensionless(synthetic_obs):
    from africas2s.tercile import to_tercile_cv
    obs = _in_mm(synthetic_obs)
    rng = np.random.default_rng(0)
    cv = _in_mm(obs + rng.normal(0, 5, obs.shape))
    n = obs.sizes["year"]
    for method, kw in (("bootstrap", {}), ("gaussian_loo", {}), ("gaussian_pev", {}),
                       ("cpt", {"leverages": np.full(n, 0.1)})):
        out = to_tercile_cv(cv, obs, method=method, **kw)
        assert out.attrs.get("units") == "1", (method, out.attrs)


def test_cpt_tercile_forecast_is_dimensionless(synthetic_obs):
    from africas2s.tercile import cpt_tercile_forecast
    obs = _in_mm(synthetic_obs)
    fc = obs.isel(year=-1, drop=True)                        # one map on the obs grid
    t33 = obs.quantile(1 / 3, dim="year").drop_vars("quantile")
    t67 = obs.quantile(2 / 3, dim="year").drop_vars("quantile")
    out = cpt_tercile_forecast(fc, t33, t67, obs.var("year"), dofr=obs.sizes["year"] - 2)
    assert out.attrs.get("units") == "1", out.attrs
