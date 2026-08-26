"""Unit tests for `africas2s.pipelines.seasonal.seasonal_mme`."""
import numpy as np
import pytest
import xarray as xr

from africas2s import seasonal_mme, SeasonalMMEResult


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


def test_function_and_result_importable():
    """Smoke test: the function and the result type are exposed from
    `africas2s` and `africas2s.pipelines`."""
    from africas2s.pipelines import seasonal_mme as f, SeasonalMMEResult as R
    assert f is seasonal_mme
    assert R is SeasonalMMEResult


def _minimal_obs(years):
    n_year = len(years)
    rng = np.random.default_rng(0)
    return _grid(rng.standard_normal((n_year, 4, 4)), year_coords=years, name="obs")


def _minimal_hcst(years):
    n_year = len(years)
    rng = np.random.default_rng(1)
    return _make_predictor(
        rng.standard_normal((n_year, 3, 4, 4)),
        member=3, year_coords=years, name="m",
    )


def test_empty_predictor_tracks_raises():
    obs = _minimal_obs(list(range(2000, 2015)))
    with pytest.raises(ValueError, match="predictor_tracks"):
        seasonal_mme({}, obs)


def test_empty_track_raises():
    obs = _minimal_obs(list(range(2000, 2015)))
    with pytest.raises(ValueError, match="prcp"):
        seasonal_mme({"prcp": {}, "sst": {"ECMWF": (_minimal_hcst(list(range(2000, 2015))), None)}}, obs)


def test_probabilistic_method_raises():
    obs = _minimal_obs(list(range(2000, 2015)))
    hcst = _minimal_hcst(list(range(2000, 2015)))
    with pytest.raises(NotImplementedError, match="corrdiff"):
        seasonal_mme(
            {"prcp": {"ECMWF": (hcst, None)}},
            obs,
            method="corrdiff",
        )


def test_year_intersection_too_small_raises():
    """Two models with only 3 years of overlap → ValueError with model names."""
    obs = _minimal_obs(list(range(2000, 2020)))
    h1 = _minimal_hcst(list(range(2000, 2005)))   # 2000-2004
    h2 = _minimal_hcst(list(range(2003, 2008)))   # 2003-2007 → overlap = 2003,2004 (only 2 years)
    with pytest.raises(ValueError, match="year intersection"):
        seasonal_mme(
            {"prcp": {"A": (h1, None), "B": (h2, None)}},
            obs,
        )


def test_year_intersection_no_overlap_with_obs_raises():
    """Obs and hindcast share no years → ValueError."""
    obs = _minimal_obs(list(range(1980, 1990)))
    hcst = _minimal_hcst(list(range(2000, 2010)))
    with pytest.raises(ValueError, match="year intersection"):
        seasonal_mme({"prcp": {"A": (hcst, None)}}, obs)


def test_forecast_year_explicit_not_available_raises():
    """`forecast_year=2020` but no model has that year → ValueError."""
    obs = _minimal_obs(list(range(2000, 2015)))
    h1 = _minimal_hcst(list(range(2000, 2015)))   # last hindcast year = 2014
    with pytest.raises(ValueError, match="2020"):
        seasonal_mme(
            {"prcp": {"A": (h1, None)}},
            obs,
            forecast_year=2020,
        )


def test_forecast_year_mismatched_forecast_slices_raises():
    """`forecast_year=None` with two models having different fcst years → ValueError."""
    obs = _minimal_obs(list(range(2000, 2015)))
    h = _minimal_hcst(list(range(2000, 2015)))
    f_2015 = _make_predictor(
        np.random.default_rng(0).standard_normal((1, 3, 4, 4)),
        member=3, year_coords=[2015], name="m",
    )
    f_2016 = _make_predictor(
        np.random.default_rng(1).standard_normal((1, 3, 4, 4)),
        member=3, year_coords=[2016], name="m",
    )
    with pytest.raises(ValueError, match="forecast_year"):
        seasonal_mme(
            {"prcp": {"A": (h, f_2015), "B": (h, f_2016)}},
            obs,
        )


def test_forecast_year_mixed_none_and_provided_raises():
    """`forecast_year=None`, mixed fcst=None vs fcst=DataArray → ValueError."""
    obs = _minimal_obs(list(range(2000, 2015)))
    h = _minimal_hcst(list(range(2000, 2015)))
    f_2015 = _make_predictor(
        np.random.default_rng(0).standard_normal((1, 3, 4, 4)),
        member=3, year_coords=[2015], name="m",
    )
    with pytest.raises(ValueError, match="forecast_year"):
        seasonal_mme(
            {"prcp": {"A": (h, f_2015), "B": (h, None)}},
            obs,
        )


def test_tercile_method_explicit_cpt_with_non_cca_raises():
    """`tercile_method='cpt'` requires `method='cca'` (for leverages). Asking
    for CPT terciles with BCSD must raise a clear ValueError."""
    obs = _minimal_obs(list(range(2000, 2015)))
    h = _minimal_hcst(list(range(2000, 2015)))
    with pytest.raises(ValueError, match="cpt"):
        seasonal_mme(
            {"prcp": {"A": (h, None)}},
            obs,
            method="bcsd",
            tercile_method="cpt",
        )


def _two_track_fixture(n_year=15, seed=42):
    """2 tracks × 2 models × n_year years × 3 members × 4×4 grid.

    Each (track, model) hindcast is a noisy copy of the obs signal so that
    CCA can learn something but the test isn't statistically demanding.
    """
    rng = np.random.default_rng(seed)
    years = list(range(2000, 2000 + n_year))
    obs_vals = rng.standard_normal((n_year, 4, 4))
    obs = _grid(obs_vals, year_coords=years, name="obs")

    def _noisy_predictor(name, noise_seed):
        rr = np.random.default_rng(noise_seed)
        # Predictor "signal" weakly tied to obs + member spread + global trend.
        signal = obs_vals[:, None, :, :] * 0.6 + rr.standard_normal((n_year, 3, 4, 4)) * 0.4
        return _make_predictor(signal, member=3, year_coords=years, name=name)

    tracks = {
        "prcp": {
            "A": (_noisy_predictor("A_prcp", 1), None),
            "B": (_noisy_predictor("B_prcp", 2), None),
        },
        "sst": {
            "A": (_noisy_predictor("A_sst", 3), None),
            "B": (_noisy_predictor("B_sst", 4), None),
        },
    }
    return tracks, obs


def test_happy_path_populates_result():
    tracks, obs = _two_track_fixture()
    result = seasonal_mme(tracks, obs, method="cca", cv="loyo",
                          cpt_args={"n_modes": 2}, verbose=False)
    # Type
    assert isinstance(result, SeasonalMMEResult)
    # Per-model dicts: 4 entries = 2 tracks × 2 models
    assert len(result.per_model_cv_hindcasts) == 4
    assert len(result.per_model_forecasts) == 4
    assert len(result.per_model_methods) == 4
    assert ("prcp", "A") in result.per_model_methods
    assert ("sst", "B") in result.per_model_methods
    # Forecast year defaulted to last intersected year.
    assert result.metadata["forecast_year"] == 2014
    assert result.metadata["tercile_method"] == "cpt"
    assert result.metadata["method"] == "cca"
    assert result.metadata["years_used"][0] == 2000
    assert result.metadata["years_used"][-1] == 2014
    assert result.metadata["n_members"] == 4
    # Headline outputs
    assert result.tercile_forecast.dims == ("tercile", "lat", "lon")
    assert "year" in result.tercile_cv.dims and "tercile" in result.tercile_cv.dims
    # Tercile probabilities sum to ~1 (ignore NaN; should be no NaN on this fixture).
    np.testing.assert_allclose(
        result.tercile_forecast.sum("tercile").values, 1.0, atol=1e-6,
    )
    np.testing.assert_allclose(
        result.tercile_cv.sum("tercile").values, 1.0, atol=1e-6,
    )
    # PEV alias matches the ensemble result's PEV
    assert result.pev is result.ensemble_result.pev
    assert result.pev is not None
    # Ensemble has weights for the 4 pooled members
    assert len(result.ensemble_result.weights) == 4


def test_year_intersection_trims_correctly():
    """Models with different year ranges → fitting/scoring uses the intersection."""
    rng = np.random.default_rng(0)
    obs = _grid(rng.standard_normal((20, 4, 4)),
                year_coords=list(range(1995, 2015)), name="obs")
    h1 = _make_predictor(rng.standard_normal((20, 3, 4, 4)),
                         member=3, year_coords=list(range(1991, 2011)), name="m1")
    h2 = _make_predictor(rng.standard_normal((20, 3, 4, 4)),
                         member=3, year_coords=list(range(1995, 2015)), name="m2")
    result = seasonal_mme(
        {"prcp": {"A": (h1, None), "B": (h2, None)}},
        obs, method="cca", cpt_args={"n_modes": 2}, verbose=False,
    )
    # Intersection should be 1995-2010 (16 years).
    assert result.metadata["years_used"][0] == 1995
    assert result.metadata["years_used"][-1] == 2010
    assert len(result.metadata["years_used"]) == 16
    # Forecast year defaults to max(intersection) = 2010.
    assert result.metadata["forecast_year"] == 2010


def test_non_cca_uses_bootstrap_tercile():
    """method='bcsd' → tercile_method auto-resolves to 'bootstrap'."""
    tracks, obs = _two_track_fixture()
    result = seasonal_mme(tracks, obs, method="bcsd", verbose=False)
    assert result.metadata["tercile_method"] == "bootstrap"


def test_cpt_per_model_aggregation_normalized_and_recorded():
    tracks, obs = _two_track_fixture()
    result = seasonal_mme(
        tracks, obs, method="cca", cv="loyo",
        cpt_args={"n_modes": 2},
        probability_aggregation="cpt_per_model",
        verbose=False,
    )
    assert result.metadata["probability_aggregation"] == "cpt_per_model"
    assert result.tercile_forecast.dims == ("tercile", "lat", "lon")
    np.testing.assert_allclose(
        result.tercile_forecast.sum("tercile").values, 1.0, atol=1e-6,
    )


def test_cpt_per_model_scores_against_matching_cv_terciles():
    """The probabilistic skill report must be scored against CV terciles built
    the same way as the published forecast. Under cpt_per_model the CV terciles
    are the per-model-averaged CPT maps, not the pooled construction, so they
    differ from the pooled-aggregation CV terciles."""
    tracks, obs = _two_track_fixture()
    pooled = seasonal_mme(
        tracks, obs, method="cca", cv="loyo", cpt_args={"n_modes": 2},
        probability_aggregation="pooled", verbose=False)
    per_model = seasonal_mme(
        tracks, obs, method="cca", cv="loyo", cpt_args={"n_modes": 2},
        probability_aggregation="cpt_per_model", verbose=False)

    assert "tercile" in per_model.tercile_cv.dims and "year" in per_model.tercile_cv.dims
    np.testing.assert_allclose(
        per_model.tercile_cv.sum("tercile").values, 1.0, atol=1e-6)
    # The CV terciles reflect the chosen aggregation rather than always the
    # pooled path: per-model-averaged != pooled-ensemble-mean construction.
    assert not np.allclose(
        pooled.tercile_cv.values, per_model.tercile_cv.values, atol=1e-9)


def test_default_aggregation_is_pooled_and_unchanged():
    tracks, obs = _two_track_fixture()
    base = seasonal_mme(tracks, obs, method="cca", cv="loyo",
                        cpt_args={"n_modes": 2}, verbose=False)
    explicit = seasonal_mme(tracks, obs, method="cca", cv="loyo",
                            cpt_args={"n_modes": 2},
                            probability_aggregation="pooled", verbose=False)
    assert base.metadata["probability_aggregation"] == "pooled"
    np.testing.assert_allclose(
        base.tercile_forecast.values, explicit.tercile_forecast.values, atol=1e-9,
    )


def test_invalid_aggregation_raises():
    tracks, obs = _two_track_fixture()
    with pytest.raises(ValueError, match="probability_aggregation"):
        seasonal_mme(tracks, obs, method="cca",
                     probability_aggregation="bogus", verbose=False)
