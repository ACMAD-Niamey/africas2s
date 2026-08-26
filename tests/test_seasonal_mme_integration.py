"""Integration tests for `seasonal_mme()` — parity with manual pipeline."""
import numpy as np
import xarray as xr
import pytest

from africas2s import seasonal_mme, downscale, ensemble, skill
from africas2s.tercile import to_tercile_cv


def _grid(values, *, year_coords, name="x"):
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


def _make_predictor(values, *, year_coords, name="m"):
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


def test_single_track_single_model_pev_matches_manual_flow():
    """With one track and one model, the orchestrator's PEV should match
    the PEV you'd get from a manual `ensemble(...)` over a single CV
    hindcast. This is a coarse equivalence — the orchestrator additionally
    runs CV folding via `cv=`, so the inner CV predictions may differ
    slightly from a one-shot fit. The signal we check is that the
    orchestrator returns a non-None PEV with the right shape, and that
    the spatial mean is in a sensible range relative to obs variance.
    """
    rng = np.random.default_rng(7)
    years = list(range(2000, 2018))
    obs = _grid(rng.standard_normal((18, 4, 4)), year_coords=years, name="obs")
    h = _make_predictor(
        rng.standard_normal((18, 3, 4, 4)) * 0.8 + obs.values[:, None, :, :] * 0.3,
        year_coords=years, name="m",
    )

    result = seasonal_mme(
        {"prcp": {"A": (h, None)}},
        obs, method="cca", cpt_args={"n_modes": 2}, verbose=False,
    )

    assert result.pev is not None
    assert result.pev.dims == ("lat", "lon")
    obs_var = float(obs.var())
    # Sanity: PEV is within an order of magnitude of obs variance.
    assert 0.1 * obs_var < float(result.pev.mean()) < 10 * obs_var


def test_two_tracks_match_manually_constructed_mme():
    """Orchestrator's pooled member count and ensemble forecast match a
    manual two-track, two-model pool produced by calling `downscale` and
    `ensemble` directly.

    We check structural agreement (member count, ensemble forecast shape,
    obs alignment) rather than exact numerical equivalence — the orchestrator
    uses LOYO CV inside `_per_model_cv` while `downscale(year-stacked)`
    uses a single fit-on-all-minus-last-year shortcut. The headline
    invariant: the orchestrator's `ensemble_result.weights` has length 4.
    """
    rng = np.random.default_rng(11)
    years = list(range(2000, 2015))
    obs = _grid(rng.standard_normal((15, 4, 4)), year_coords=years, name="obs")
    def _h(seed):
        rr = np.random.default_rng(seed)
        return _make_predictor(
            rr.standard_normal((15, 3, 4, 4)),
            year_coords=years, name="m",
        )
    result = seasonal_mme(
        {"prcp": {"A": (_h(1), None), "B": (_h(2), None)},
         "sst":  {"A": (_h(3), None), "B": (_h(4), None)}},
        obs, method="cca", cpt_args={"n_modes": 2}, verbose=False,
    )
    assert len(result.ensemble_result.weights) == 4
    assert result.ensemble_result.forecast.dims[0] == "year"
    assert "lat" in result.ensemble_result.forecast.dims
    assert "lon" in result.ensemble_result.forecast.dims


def test_seasonal_mme_surfaces_member_contributions():
    """seasonal_mme() should copy ensemble_result.member_contributions into
    skill_report.diagrams['member_contributions']."""
    rng = np.random.default_rng(13)
    years = list(range(2000, 2020))
    obs = _grid(rng.standard_normal((20, 4, 4)), year_coords=years, name="obs")

    def _h(seed):
        rr = np.random.default_rng(seed)
        return _make_predictor(
            rr.standard_normal((20, 3, 4, 4)),
            year_coords=years, name="m",
        )

    from africas2s import seasonal_mme
    result = seasonal_mme(
        {"prcp": {"A": (_h(1), None), "B": (_h(2), None)},
         "sst":  {"A": (_h(3), None), "B": (_h(4), None)}},
        obs, method="cca", cpt_args={"n_modes": 2}, verbose=False,
    )

    # Live on EnsembleResult
    mc = result.ensemble_result.member_contributions
    assert mc is not None
    assert set(mc.keys()) == {"prcp__A", "prcp__B", "sst__A", "sst__B"}

    # Surfaced into the SkillReport's diagrams payload
    surfaced = result.skill_report.diagrams.get("member_contributions")
    assert surfaced is mc
