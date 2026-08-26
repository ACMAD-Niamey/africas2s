"""Unit + equivalence tests for `seasonal_mme(native_years=True)`.

`native_years=True` lets each model in a CCA `cpt_per_model` MME calibrate on
its own `hcst.year ∩ obs.year` overlap instead of the single global
intersection. See docs/superpowers/specs/2026-05-15-seasonal-mme-orchestrator-
design.md for the base contract this extends.
"""
import numpy as np
import pytest
import xarray as xr

import africas2s
from africas2s import seasonal_mme


def _grid(values, *, year_coords, name="x"):
    """Wrap (year, lat, lon) values into a DataArray with named coords."""
    n_year, n_lat, n_lon = values.shape
    return xr.DataArray(
        values,
        dims=("year", "lat", "lon"),
        coords={
            "year": list(year_coords),
            "lat": np.linspace(-5.0, 5.0, n_lat),
            "lon": np.linspace(30.0, 40.0, n_lon),
        },
        name=name,
    )


def _make_predictor(values, *, member, year_coords, name="m"):
    """Wrap (year, member, lat, lon) values into a DataArray."""
    n_year, n_mem, n_lat, n_lon = values.shape
    return xr.DataArray(
        values,
        dims=("year", "member", "lat", "lon"),
        coords={
            "year": list(year_coords),
            "member": list(range(n_mem)),
            "lat": np.linspace(-5.0, 5.0, n_lat),
            "lon": np.linspace(30.0, 40.0, n_lon),
        },
        name=name,
    )


def _minimal_obs(years):
    n_year = len(years)
    rng = np.random.default_rng(0)
    return _grid(rng.standard_normal((n_year, 4, 4)), year_coords=years, name="obs")


def _minimal_hcst(years, seed=1):
    n_year = len(years)
    rng = np.random.default_rng(seed)
    return _make_predictor(
        rng.standard_normal((n_year, 3, 4, 4)),
        member=3, year_coords=years, name="m",
    )


def test_native_years_requires_cpt_per_model():
    """native_years=True + probability_aggregation='pooled' is undefined
    (pooling members with different year sets) and must raise early."""
    with pytest.raises(ValueError, match="cpt_per_model"):
        africas2s.seasonal_mme(
            {"prcp": {}}, xr.DataArray([0]),
            method="cca", native_years=True, probability_aggregation="pooled",
        )


def test_native_years_rejects_skillmask_threshold():
    """native_years=True + cpt_args['skillmask_threshold'] is incoherent: the
    skill mask needs a single shared obs baseline (pearson(forecast, obs) +
    climo_mean), but native_years rebinds obs_sliced to the (possibly empty)
    union/intersection of per-model years, not the coherent baseline that
    produced tercile_forecast. Must raise early, before any calibration."""
    with pytest.raises(ValueError, match="skillmask"):
        africas2s.seasonal_mme(
            {"prcp": {}}, xr.DataArray([0]),
            method="cca", native_years=True, probability_aggregation="cpt_per_model",
            cpt_args={"skillmask_threshold": 0.1},
        )


def test_native_years_default_false_unchanged():
    """native_years is opt-in: omitting it must behave exactly like
    native_years=False (smoke test that the new kwarg doesn't change the
    default call shape)."""
    years = list(range(2000, 2015))
    obs = _minimal_obs(years)
    hcst = _minimal_hcst(years)
    result_default = seasonal_mme(
        {"prcp": {"A": (hcst, None)}}, obs,
        method="cca", probability_aggregation="cpt_per_model",
        forecast_year=2014, verbose=False,
    )
    result_explicit_false = seasonal_mme(
        {"prcp": {"A": (hcst, None)}}, obs,
        method="cca", probability_aggregation="cpt_per_model",
        forecast_year=2014, verbose=False, native_years=False,
    )
    xr.testing.assert_identical(
        result_default.tercile_forecast, result_explicit_false.tercile_forecast
    )


def _fcst_slice(hcst, year, seed):
    """A single-year forecast slice (year, member, lat, lon), independent of
    hcst's own year range, so `forecast_year` resolution (Rule 1) doesn't
    require the forecast year to be inside every model's hindcast range."""
    template = hcst.isel(year=0, drop=True)
    rng = np.random.default_rng(seed)
    fc = template.copy(data=rng.standard_normal(template.shape))
    return fc.expand_dims(year=[year])


def test_native_years_per_model_overlap_used():
    """Two models with different (but each individually >=5-year) overlaps
    with obs must not raise the global 'year intersection' error that
    native_years=False would raise for a <5-year global overlap."""
    obs = _minimal_obs(list(range(2000, 2020)))
    # A: overlaps obs on 2000-2009 (10 years). B: overlaps obs on 2015-2019
    # only (5 years). Global intersection of A,B,obs is empty.
    h_a = _minimal_hcst(list(range(2000, 2010)), seed=1)
    h_b = _minimal_hcst(list(range(2015, 2020)), seed=2)
    f_a = _fcst_slice(h_a, 2025, seed=11)
    f_b = _fcst_slice(h_b, 2025, seed=12)
    # Keep EOF modes small and fixed (rather than the CCA default of
    # auto-scaling up to n_years-1) so dof = n_years - n_modes - 1 stays
    # positive for both the 10-year and the 5-year model.
    cpt = dict(x_eof_modes=2, y_eof_modes=2, cca_modes=1)

    with pytest.raises(ValueError, match="year intersection"):
        seasonal_mme(
            {"prcp": {"A": (h_a, f_a), "B": (h_b, f_b)}}, obs,
            method="cca", probability_aggregation="cpt_per_model",
            native_years=False, forecast_year=2025, verbose=False, cpt_args=cpt,
        )

    # native_years=True: each model uses its own overlap, so this succeeds.
    result = seasonal_mme(
        {"prcp": {"A": (h_a, f_a), "B": (h_b, f_b)}}, obs,
        method="cca", probability_aggregation="cpt_per_model",
        native_years=True, forecast_year=2025, verbose=False, cpt_args=cpt,
    )
    assert result.tercile_forecast is not None
    assert set(result.tercile_forecast.dims) >= {"tercile", "lat", "lon"}


def test_native_years_per_model_floor_enforced():
    """Each model's own overlap must still meet the >=5-year floor under
    native_years=True; a model with too little overlap raises."""
    obs = _minimal_obs(list(range(2000, 2020)))
    h_a = _minimal_hcst(list(range(2000, 2010)), seed=1)
    # B overlaps obs on only 2018-2019 (2 years) -> below the 5-year floor.
    h_b = _minimal_hcst(list(range(2018, 2020)), seed=2)
    f_a = _fcst_slice(h_a, 2025, seed=11)
    f_b = _fcst_slice(h_b, 2025, seed=12)

    with pytest.raises(ValueError, match="year intersection"):
        seasonal_mme(
            {"prcp": {"A": (h_a, f_a), "B": (h_b, f_b)}}, obs,
            method="cca", probability_aggregation="cpt_per_model",
            native_years=True, forecast_year=2025, verbose=False,
        )
