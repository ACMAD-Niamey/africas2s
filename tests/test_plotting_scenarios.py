"""Scenario-completion and index-scatter plots.

Plot tests can only check structure, so they check the structure that carries
meaning: that each segment is drawn in its own colour, that the analog spaghetti
has one line per analog, and that an error bar is drawn around the forecast
rather than through it.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from africas2s.analog import analogs_from_years  # noqa: E402
from africas2s.climate import seasonal_stack  # noqa: E402
from africas2s.completion import complete  # noqa: E402
from africas2s.plotting.scenarios import (  # noqa: E402
    _contiguous_runs,
    _half_widths,
    plot_accumulation_scenarios,
    plot_index_scatter,
)
from africas2s.series import error_bounds  # noqa: E402

SEASON = "JJAS"
YEARS = np.arange(1991, 2027)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _dekads(years):
    return pd.DatetimeIndex(
        [pd.Timestamp(y, m, d) for y in years for m in range(1, 13) for d in (1, 11, 21)]
    )


@pytest.fixture
def result():
    stamps = _dekads(YEARS)
    series = xr.DataArray(
        (stamps.year.to_numpy() - 1990).astype(float), dims="time",
        coords={"time": stamps},
    )
    clim = seasonal_stack(series, SEASON, years=range(1991, 2026))
    obs = series.sel(time=slice("2026-06-01", "2026-07-20"))   # 5 dekads
    fcst = series.sel(time=slice("2026-07-21", "2026-08-10")) * 0 + 2.0  # 2 dekads
    return complete(obs, analogs_from_years([1997, 2005, 2015]),
                    climatology=clim, season=SEASON, forecast=fcst), clim


# --- run splitting ---------------------------------------------------------


def test_contiguous_runs_extends_each_run_one_step_left_to_close_visual_gaps():
    """A line drawn only over its own segment leaves a gap at every handover."""
    mask = np.array([False, False, True, True, False])
    assert _contiguous_runs(mask) == [(1, 4)]


def test_contiguous_runs_handles_a_leading_run_without_going_negative():
    assert _contiguous_runs(np.array([True, True, False])) == [(0, 2)]


def test_contiguous_runs_handles_a_trailing_run():
    assert _contiguous_runs(np.array([False, True, True])) == [(0, 3)]


def test_contiguous_runs_finds_several_runs():
    mask = np.array([True, False, False, True, True])
    assert _contiguous_runs(mask) == [(0, 1), (2, 5)]


# --- accumulation plot -----------------------------------------------------


def test_accumulation_plot_draws_one_line_per_analog_plus_the_shared_segments(result):
    completion, _ = result
    fig = plot_accumulation_scenarios(completion, show_consensus=False)
    ax = fig.axes[0]
    colors = [line.get_color() for line in ax.lines]
    assert colors.count("#8a8a8a") == 3      # one analog spaghetti per scenario
    assert colors.count("#1f5fa9") == 1      # observed, drawn once
    assert colors.count("#e0559b") == 1      # forecast, drawn once


def test_accumulation_plot_observed_curve_ends_where_the_forecast_begins(result):
    """The observed and forecast lines must meet, not leave a gap."""
    completion, _ = result
    fig = plot_accumulation_scenarios(completion, show_consensus=False)
    lines = {line.get_color(): line for line in fig.axes[0].lines}
    observed = lines["#1f5fa9"].get_xydata()
    forecast = lines["#e0559b"].get_xydata()
    np.testing.assert_allclose(observed[-1], forecast[0])


def test_accumulation_plot_adds_a_consensus_line_and_a_legend_entry(result):
    completion, _ = result
    fig = plot_accumulation_scenarios(completion)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "Observed" in labels and "Forecast" in labels
    assert any("Analog median (n=3)" == label for label in labels)


def test_accumulation_plot_can_overlay_a_climatological_reference(result):
    completion, clim = result
    fig = plot_accumulation_scenarios(completion, climatology=clim)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "Climatological median" in labels


def test_accumulation_plot_uses_calendar_stamps_when_the_completion_carried_them(result):
    completion, _ = result
    fig = plot_accumulation_scenarios(completion)
    assert fig.axes[0].get_xlabel() == ""  # dates, not "season step"
    assert "JJAS 2026" in fig.axes[0].get_title()


def test_accumulation_plot_refuses_an_unreduced_grid():
    stamps = _dekads(YEARS)
    grid = xr.DataArray(
        np.ones((len(stamps), 2, 2)), dims=("time", "lat", "lon"),
        coords={"time": stamps, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    clim = seasonal_stack(grid, SEASON, years=range(1991, 2026))
    obs = grid.sel(time=slice("2026-06-01", "2026-07-20"))
    completion = complete(obs, analogs_from_years([1997]), climatology=clim,
                          season=SEASON)
    with pytest.raises(ValueError, match="select a single pixel/region"):
        plot_accumulation_scenarios(completion)


def test_accumulation_plot_accepts_a_reduce_callable():
    stamps = _dekads(YEARS)
    grid = xr.DataArray(
        np.ones((len(stamps), 2, 2)), dims=("time", "lat", "lon"),
        coords={"time": stamps, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    clim = seasonal_stack(grid, SEASON, years=range(1991, 2026))
    obs = grid.sel(time=slice("2026-06-01", "2026-07-20"))
    completion = complete(obs, analogs_from_years([1997]), climatology=clim,
                          season=SEASON)
    fig = plot_accumulation_scenarios(completion, reduce=lambda d: d.mean(["lat", "lon"]))
    assert fig.axes[0].lines


def test_accumulation_plot_draws_onto_a_supplied_axis(result):
    completion, _ = result
    _, ax = plt.subplots()
    fig = plot_accumulation_scenarios(completion, ax=ax)
    assert fig is ax.figure


# --- index scatter ---------------------------------------------------------


def _series(values, name, years=YEARS):
    return xr.DataArray(np.asarray(values, float), dims="year",
                        coords={"year": years}, name=name)


def _categorical(values, name, years=YEARS):
    """A non-float series, as a real tercile or class label would be."""
    return xr.DataArray(np.asarray(values), dims="year",
                        coords={"year": years}, name=name)


@pytest.fixture
def indices():
    rng = np.random.default_rng(0)
    return (_series(rng.normal(size=len(YEARS)), "roni"),
            _series(rng.normal(size=len(YEARS)), "dmi"))


def test_index_scatter_plots_every_shared_year(indices):
    x, y = indices
    fig = plot_index_scatter(x, y)
    assert sum(c.get_offsets().shape[0] for c in fig.axes[0].collections) == len(YEARS)


def test_index_scatter_aligns_on_the_intersection_of_years(indices):
    x, y = indices
    fig = plot_index_scatter(x, y.sel(year=slice(2000, 2010)))
    assert sum(c.get_offsets().shape[0] for c in fig.axes[0].collections) == 11


def test_index_scatter_colours_by_an_observed_category(indices):
    """The move that turns a scatter of ocean states into a statement about
    rainfall: one colour per observed tercile."""
    x, y = indices
    terciles = _categorical(np.tile([0, 1, 2], len(YEARS) // 3), "tercile")
    fig = plot_index_scatter(x, y, color_by=terciles)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert sorted(labels) == ["0", "1", "2"]
    assert len(fig.axes[0].collections) == 3


def test_index_scatter_accepts_string_categories(indices):
    x, y = indices
    labels = np.array(["dry", "normal", "wet"] * (len(YEARS) // 3))
    fig = plot_index_scatter(x, y, color_by=_categorical(labels, "cat"),
                             categories=["dry", "normal", "wet"])
    got = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert got == ["dry", "normal", "wet"]


def test_index_scatter_annotates_the_highlighted_analog_years(indices):
    x, y = indices
    fig = plot_index_scatter(x, y, highlight=[1997, 2015])
    annotations = [a.get_text() for a in fig.axes[0].texts]
    assert sorted(annotations) == ["1997", "2015"]


def test_index_scatter_marks_the_forecast_with_a_star(indices):
    x, y = indices
    fig = plot_index_scatter(x, y, forecast=(1.5, 0.8))
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "Forecast" in labels


def test_index_scatter_brackets_the_forecast_with_error_bars(indices):
    x, y = indices
    hindcast_pred = _series(np.linspace(-1, 1, len(YEARS)), "pred")
    obs = _series(np.zeros(len(YEARS)), "obs")
    bounds = error_bounds(hindcast_pred, obs, 1.5, level=0.8)
    fig = plot_index_scatter(x, y, forecast=(1.5, 0.8), error_bars=(bounds, bounds))
    # errorbar draws its caps and bars as extra artists on the axis.
    assert len(fig.axes[0].containers) >= 1


def test_half_widths_are_offsets_from_the_centre_not_absolute_bounds():
    """matplotlib's errorbar wants distances. Passing absolute bounds would draw
    the interval in the wrong place entirely."""
    got = _half_widths(10.0, (8.0, 14.0))
    np.testing.assert_allclose(got, [[2.0], [4.0]])


def test_half_widths_accepts_an_errorbounds_object():
    obs = _series(np.zeros(len(YEARS)), "obs")
    bounds = error_bounds(_series(np.linspace(-1, 1, len(YEARS)), "p"), obs, 0.0)
    got = _half_widths(0.0, bounds)
    assert got.shape == (2, 1) and (got >= 0).all()


def test_index_scatter_labels_axes_from_the_series_names(indices):
    x, y = indices
    ax = plot_index_scatter(x, y).axes[0]
    assert ax.get_xlabel() == "roni" and ax.get_ylabel() == "dmi"


def test_index_scatter_axis_labels_can_be_overridden(indices):
    x, y = indices
    ax = plot_index_scatter(x, y, xlabel="RONI (C)", ylabel="IOD (C)",
                            title="OND 2026").axes[0]
    assert ax.get_xlabel() == "RONI (C)"
    assert ax.get_title() == "OND 2026"


# --- new styling options ---------------------------------------------------


def test_index_scatter_fills_highlighted_points_when_given_a_colour(indices):
    x, y = indices
    fig = plot_index_scatter(x, y, highlight=[1997, 2015], highlight_color="#d62728")
    # the highlight scatter is the collection with a non-'none' facecolor
    filled = [c for c in fig.axes[0].collections
              if c.get_facecolors().size and c.get_facecolors()[0][3] > 0]
    assert filled  # at least one collection is actually filled


def test_index_scatter_draws_a_trendline_and_r2(indices):
    x, y = indices
    fig = plot_index_scatter(x, y, trendline=True)
    dashed = [ln for ln in fig.axes[0].lines if ln.get_linestyle() == "--"]
    assert dashed, "expected an OLS trendline"
    assert any("R²" in t.get_text() for t in fig.axes[0].texts)


def test_index_scatter_trendline_annotation_can_be_suppressed(indices):
    x, y = indices
    fig = plot_index_scatter(x, y, trendline=True, trendline_annotate=False)
    assert not any("R²" in t.get_text() for t in fig.axes[0].texts)


def test_accumulation_colours_each_scenario_and_labels_it_by_year(result):
    res, clim = result
    fig = plot_accumulation_scenarios(res, color_by_scenario=True)
    legend = fig.axes[0].get_legend()
    labels = [t.get_text() for t in legend.get_texts()]
    for yr in ("1997", "2005", "2015"):
        assert yr in labels
