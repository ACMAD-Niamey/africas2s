"""Plots for scenario completion and index diagnostics.

Two figures that a forecaster reads rather than admires.

:func:`plot_accumulation_scenarios` draws the season-to-date accumulation: the
observed curve, the forecast segment continuing it, and one thin line per analog
fanning out to the season's end. Because each segment is drawn from the
:class:`~africas2s.completion.CompletionResult`'s own segment labels, the
picture cannot disagree with the numbers.

:func:`plot_index_scatter` draws one scalar index against another, optionally
coloured by an observed outcome category and bracketed by a forecast's
confidence interval. It is the shape of every "here is where this year falls
among the historical analogs" argument — a RONI-versus-IOD scatter coloured by
observed rainfall tercile, a West-Pacific-rainfall-versus-Kiremt-rainfall
scatter, a forecast-versus-observed verification plot.

:func:`plot_index_evolution` draws the "which past years looked like this
one" plume: every year's index curve in grey, the selected analogs in colour,
this year in black over the steps it has actually reached.

None of them knows what it is plotting. All take DataArrays and labels.
"""
from __future__ import annotations

import numpy as np

from .._optional import require_optional

_HINT = "pip install africas2s[plotting]"

# Observed, forecast, analog. Blue for what happened, pink for what a model
# says will happen, grey for what happened in comparable years — the CHC deck's
# convention, and a sensible one: the eye should not read an analog as data.
_SEGMENT_STYLE = {
    "observed": dict(color="#1f5fa9", linewidth=2.6, zorder=5),
    "forecast": dict(color="#e0559b", linewidth=2.6, zorder=5),
    "analog": dict(color="#8a8a8a", linewidth=0.9, alpha=0.65, zorder=2),
}


def _new_fig(ax, figsize):
    import importlib

    require_optional("matplotlib", _HINT)
    plt = importlib.import_module("matplotlib.pyplot")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return plt, fig, ax


def _x_axis(curves):
    """Calendar stamps when the completion carried them, else the step index."""
    if "time" in curves.coords:
        return curves.time.values, None
    return curves.step.values, "season step"


def _contiguous_runs(mask):
    """Index runs where ``mask`` is True, each extended one step to the left.

    A line drawn only over its own segment leaves a visual gap at every
    boundary. Extending each run backwards by one point makes the observed curve
    meet the forecast curve exactly at the handover, which is where they do in
    fact meet.
    """
    runs, start = [], None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((max(start - 1, 0), i))
            start = None
    if start is not None:
        runs.append((max(start - 1, 0), len(mask)))
    return runs


def plot_accumulation_scenarios(
    result,
    *,
    ax=None,
    reduce=None,
    show_consensus: bool = True,
    color_by_scenario: bool = False,
    climatology=None,
    title: str | None = None,
    ylabel: str = "accumulation",
    figsize=(9, 5.5),
):
    """Plot the accumulation curves of a :class:`CompletionResult`.

    Parameters
    ----------
    result : CompletionResult
        Its ``scenarios`` must reduce to a single series — select a pixel or a
        region first (``result.scenarios.sel(region="sebeta")``), or pass
        ``reduce``.
    reduce : callable, optional
        Applied to each of ``scenarios`` and ``climatology`` to collapse the
        non-``(scenario, step)`` dims, e.g.
        ``lambda da: da.mean(["lat", "lon"])``.
    show_consensus : bool
        Draw the median-across-analogs accumulation as a heavier dashed line.
    climatology : xr.DataArray, optional
        ``(year, step, ...)`` historical increments; its median accumulation is
        drawn as a reference. Pass ``result``'s own climatology to show how far
        the scenarios sit from a normal season.

    Returns
    -------
    matplotlib Figure
    """
    plt, fig, ax = _new_fig(ax, figsize)

    curves = result.accumulation()
    if reduce is not None:
        curves = reduce(curves)
    extra = [d for d in curves.dims if d not in ("scenario", "step")]
    if extra:
        raise ValueError(
            f"accumulation still has dims {extra}; select a single pixel/region "
            "or pass reduce= to collapse them."
        )

    x, x_label = _x_axis(curves)
    segments = np.asarray(result.segments.values)

    if color_by_scenario:
        # One colour per analog scenario, labelled by its scenario coordinate
        # (the analog year) — the deck's colour-by-year convention.
        import matplotlib.pyplot as _plt

        scenarios = list(curves.scenario.values)
        palette = _plt.get_cmap("turbo", max(len(scenarios), 1))
        for i, scenario in enumerate(scenarios):
            values = curves.sel(scenario=scenario).values
            labelled = False
            for start, stop in _contiguous_runs(segments == "analog"):
                ax.plot(x[start:stop], values[start:stop], color=palette(i),
                        linewidth=1.1, alpha=0.9, zorder=3,
                        label=(str(scenario) if not labelled else None))
                labelled = True
    else:
        for scenario in curves.scenario.values:
            values = curves.sel(scenario=scenario).values
            for start, stop in _contiguous_runs(segments == "analog"):
                ax.plot(x[start:stop], values[start:stop], **_SEGMENT_STYLE["analog"])

    # Observed and forecast are identical across scenarios by construction, so
    # draw them once from the first member.
    reference = curves.isel(scenario=0).values
    for name in ("observed", "forecast"):
        for start, stop in _contiguous_runs(segments == name):
            ax.plot(x[start:stop], reference[start:stop],
                    label=name.capitalize(), **_SEGMENT_STYLE[name])

    if show_consensus:
        consensus = curves.median("scenario")
        analog_mask = segments == "analog"
        if analog_mask.any():
            first = max(int(np.argmax(analog_mask)) - 1, 0)
            ax.plot(x[first:], consensus.values[first:], color="#1a1a1a",
                    linewidth=2.0, linestyle="--", zorder=6,
                    label=f"Analog median (n={curves.sizes['scenario']})")

    if climatology is not None:
        clim = reduce(climatology) if reduce is not None else climatology
        clim_curve = clim.cumsum("step").median("year")
        ax.plot(x, clim_curve.values, color="#2e7d32", linewidth=1.6,
                linestyle=":", zorder=4, label="Climatological median")

    # Deduplicate the per-run labels so the legend has one entry per segment.
    handles, labels = ax.get_legend_handles_labels()
    seen, unique = set(), []
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen.add(label)
            unique.append((handle, label))
    if unique:
        ax.legend(*zip(*unique), loc="upper left", frameon=False, fontsize=9)

    ax.set_ylabel(ylabel)
    if x_label:
        ax.set_xlabel(x_label)
    else:
        fig.autofmt_xdate()
    meta = result.metadata
    ax.set_title(title or f"{meta.get('season')} {meta.get('year')} scenario completion")
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    return fig


# Diverging three-class palette: dry / normal / wet. Readable in greyscale by
# lightness order, and safe for the common red-green colour blindness.
_TERCILE_COLORS = ("#b2582b", "#bdbdbd", "#2c6fac")


def plot_index_scatter(
    x,
    y,
    *,
    ax=None,
    color_by=None,
    categories=None,
    colors=None,
    highlight=None,
    highlight_color=None,
    forecast=None,
    forecast_marker="*",
    forecast_color="#f2c14e",
    forecast_label="Forecast",
    error_bars=None,
    trendline: bool = False,
    trendline_annotate: bool = True,
    labels: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    figsize=(7, 6),
):
    """Scatter two index series against each other, over a shared ``year`` dim.

    Parameters
    ----------
    x, y : xr.DataArray
        Series over ``year``. Aligned on the intersection of their years.
    color_by : xr.DataArray, optional
        A categorical series over ``year`` (e.g. the observed rainfall tercile).
        Points are coloured by it, which is what turns a scatter of ocean states
        into a statement about rainfall outcomes.
    categories : sequence, optional
        The categories in legend order. Defaults to the sorted unique values.
    highlight : sequence of year labels, optional
        Drawn with a heavier edge and annotated — the analog years.
    highlight_color : colour, optional
        Fill for the highlighted points. Defaults to unfilled (an open ring);
        set e.g. ``"#d62728"`` for the deck's solid-red-dot convention.
    trendline : bool
        Draw the ordinary-least-squares best-fit line through the points.
    trendline_annotate : bool
        With ``trendline``, print the R² of the fit in the corner.
    forecast : tuple (x, y), optional
        The year being forecast, drawn as a star.
    error_bars : tuple, optional
        ``(x_bounds, y_bounds)`` — each an :class:`~africas2s.series.ErrorBounds`
        or an ``(lower, upper)`` pair — bracketing ``forecast``.

    Returns
    -------
    matplotlib Figure
    """
    import xarray as xr

    plt, fig, ax = _new_fig(ax, figsize)

    x, y = xr.align(x, y, join="inner")
    years = x.year.values

    if color_by is None:
        ax.scatter(x.values, y.values, s=42, c="#4c72b0", edgecolor="white",
                   linewidth=0.7, zorder=3)
    else:
        color_by = color_by.sel(year=years)
        values = color_by.values
        if categories is None:
            categories = [c for c in np.unique(values[~_isnull(values)])]
        palette = colors or _TERCILE_COLORS
        for i, category in enumerate(categories):
            mask = values == category
            if not mask.any():
                continue
            ax.scatter(x.values[mask], y.values[mask], s=42,
                       c=palette[i % len(palette)], edgecolor="white",
                       linewidth=0.7, zorder=3, label=str(category))

    if trendline:
        xv, yv = np.asarray(x.values, float), np.asarray(y.values, float)
        ok = np.isfinite(xv) & np.isfinite(yv)
        if ok.sum() >= 2:
            slope, intercept = np.polyfit(xv[ok], yv[ok], 1)
            xs = np.array([xv[ok].min(), xv[ok].max()])
            ax.plot(xs, intercept + slope * xs, "--", color="#555555",
                    linewidth=1.4, zorder=2)
            if trendline_annotate:
                r2 = np.corrcoef(xv[ok], yv[ok])[0, 1] ** 2
                ax.annotate(f"R² = {r2:.2f}", xy=(0.05, 0.93),
                            xycoords="axes fraction", fontsize=9, color="#333333")

    if highlight is not None:
        mask = np.isin(years, np.asarray(highlight))
        ax.scatter(x.values[mask], y.values[mask], s=110,
                   facecolor=highlight_color if highlight_color is not None else "none",
                   edgecolor="#1a1a1a", linewidth=1.5, zorder=4)
        for year in years[mask]:
            ax.annotate(str(year), (float(x.sel(year=year)), float(y.sel(year=year))),
                        textcoords="offset points", xytext=(7, 5), fontsize=8)
    if labels:
        for year in years:
            ax.annotate(str(year), (float(x.sel(year=year)), float(y.sel(year=year))),
                        textcoords="offset points", xytext=(5, 3), fontsize=7,
                        color="#555555")

    if forecast is not None:
        fx, fy = float(forecast[0]), float(forecast[1])
        if error_bars is not None:
            x_bounds, y_bounds = error_bars
            ax.errorbar(
                fx, fy,
                xerr=_half_widths(fx, x_bounds), yerr=_half_widths(fy, y_bounds),
                fmt="none", ecolor="#1a1a1a", elinewidth=1.2, capsize=4, zorder=5,
            )
        s = 340 if forecast_marker == "*" else 150
        ax.scatter([fx], [fy], marker=forecast_marker, s=s, c=forecast_color,
                   edgecolor="#1a1a1a", linewidth=1.0, zorder=6, label=forecast_label)

    ax.axhline(0, color="#999999", linewidth=0.7, zorder=1)
    ax.axvline(0, color="#999999", linewidth=0.7, zorder=1)
    ax.set_xlabel(xlabel or (x.name or "x"))
    ax.set_ylabel(ylabel or (y.name or "y"))
    if title:
        ax.set_title(title)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc="best", frameon=False, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    return fig


# Highlight colours for analog curves, best analog first: a warm ramp that
# reads in order and stays distinct from the black target and grey plume.
_ANALOG_COLORS = ["#d7301f", "#fc8d59", "#fdcc8a", "#7f2704", "#b35806", "#e08214"]


def plot_index_evolution(curves, analogs=None, *, target_year=None, highlight=None,
                         labels=None, colors=None, ax=None, title=None,
                         ylabel=None, zero_line=True, figsize=(11, 4.6)):
    """Plume of an index's evolution by year, with the analog years picked out.

    Parameters
    ----------
    curves : xr.DataArray
        ``(year, step)`` — the same array :func:`africas2s.analogs_from_evolution`
        scored, e.g. ``seasonal_stack(oni, (1, 12), cadence="monthly")``.
    analogs : AnalogSet, optional
        The selection to highlight. Its ``metadata`` supplies ``target_year``
        and, when it came from ``analogs_from_evolution``, the per-year ``r``
        and ``mad`` that annotate the legend.
    target_year : int, optional
        The year drawn in black; defaults to ``analogs.metadata["target_year"]``.
        Its curve is drawn only over the steps it has (a partly observed year
        stops where the record stops, with markers on each observed step).
    highlight : sequence of int, optional
        Years to colour when there is no ``analogs`` (or in addition to it).
    labels : sequence of str, optional
        Tick labels for the step axis (e.g. the twelve overlapping-season
        codes ``DJF … NDJ``); defaults to the ``step`` coordinate values.
    colors : sequence of str, optional
        Highlight colours, best analog first; cycled if shorter.
    ax, title, ylabel, zero_line, figsize
        Usual matplotlib plumbing. ``zero_line`` draws a reference at 0 (an
        anomaly index); switch it off for an absolute index.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not {"year", "step"} <= set(curves.dims):
        raise ValueError(
            f"curves must have 'year' and 'step' dims, got {tuple(curves.dims)}")
    plt, fig, ax = _new_fig(ax, figsize)
    curves = curves.transpose("year", "step")
    years = [int(y) for y in curves.year.values]
    x = np.arange(curves.sizes["step"])

    if target_year is None and analogs is not None:
        target_year = analogs.metadata.get("target_year")
    picked = [int(y) for y in (analogs.years if analogs is not None else [])]
    picked += [int(y) for y in (highlight or []) if int(y) not in picked]
    colors = list(colors or _ANALOG_COLORS)
    meta = analogs.metadata if analogs is not None else {}
    r_by_year, mad_by_year = meta.get("r", {}), meta.get("mad", {})

    for y in years:
        if y == target_year or y in picked:
            continue
        ax.plot(x, curves.sel(year=y).values, color="0.85", linewidth=0.8, zorder=1)
    for k, y in enumerate(picked):
        if y not in years:
            raise ValueError(f"highlighted year {y} is not in curves.year")
        label = str(y)
        if y in r_by_year and y in mad_by_year:
            label += f" (r={r_by_year[y]:+.2f}, mad={mad_by_year[y]:.2f})"
        ax.plot(x, curves.sel(year=y).values, color=colors[k % len(colors)],
                linewidth=2.3, zorder=3, label=label)
    if target_year is not None:
        if target_year not in years:
            raise ValueError(f"target_year {target_year} is not in curves.year")
        t = np.asarray(curves.sel(year=target_year).values, dtype=float)
        have = np.isfinite(t)
        partial = bool(have.any()) and not bool(have.all())
        ax.plot(x[have], t[have], color="black", linewidth=3.0, marker="o",
                markersize=4, zorder=4,
                label=f"{target_year} observed" + (" (to date)" if partial else ""))

    if zero_line:
        ax.axhline(0, color="0.4", linewidth=0.8, zorder=0)
    ax.grid(alpha=0.25)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in (labels if labels is not None
                                         else curves.step.values)])
    if ylabel is None:
        ylabel = curves.name or "index"
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontweight="bold")
    if picked or target_year is not None:
        ax.legend(fontsize=9, loc="upper left")
    return fig


def _isnull(values):
    try:
        return np.isnan(values)
    except TypeError:  # non-numeric categories (e.g. "below"/"above")
        return np.array([v is None for v in values])


def _half_widths(centre, bounds):
    """(lower, upper) or ErrorBounds -> the 2x1 offsets matplotlib wants."""
    lower, upper = (bounds.lower, bounds.upper) if hasattr(bounds, "lower") else bounds
    return np.array([[centre - float(lower)], [float(upper) - centre]])
