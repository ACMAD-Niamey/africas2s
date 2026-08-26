"""Multi-panel forecast composites: matrices of maps, components + objective.

Every replication workflow so far has hand-rolled the same figure scaffolding:
a grid of per-model maps sharing one legend, an "all the component forecasts
plus the final objective" composite, panel-title chips that survive cartopy's
gridline labels, and a figure-level tercile legend built from the palette.
This module makes those one call each:

- :func:`plot_matrix` — a grid of tercile and/or continuous maps (mixed is
  fine) with shared masks, one shared tercile legend, and one shared colorbar.
- :func:`plot_components_objective` — the components-plus-final composite:
  each input forecast alongside the combined objective, objective emphasized.
- :func:`tercile_legend_handles` — the palette-derived legend handles, for
  composing custom figures.

Masks follow the rest of the package: the ``TercileStyle`` carries the dry
mask, the country/geometry clip, extent, and palette; ``skill_mask`` blanks
low-skill cells at display time (True = insufficient skill → nodata). For
mask-before-combine semantics use :func:`africas2s.mask_by_skill` on the data
instead — both conventions are in operational use.
"""
from __future__ import annotations

import importlib
import math

import numpy as np
import xarray as xr

from .._optional import require_optional
from .forecasts import (
    _HINT,
    _align_bool_mask,
    plot_field,
    plot_tercile_forecast,
    region_masks,
)


def _normalize_panels(panels, *, context="plot_matrix"):
    """Accept a dict {title: data} or an iterable of (title, data[, style])."""
    if hasattr(panels, "items"):
        entries = [(str(k), v, None) for k, v in panels.items()]
    else:
        entries = []
        for item in panels:
            if isinstance(item, (tuple, list)) and len(item) == 3:
                title, da, st = item
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                (title, da), st = item, None
            else:
                raise TypeError(
                    f"{context}: each panel must be (title, data) or "
                    f"(title, data, style), got {type(item).__name__}"
                )
            entries.append((str(title), da, st))
    if not entries:
        raise ValueError(f"{context}: no panels to draw")
    return entries


def _is_tercile(da):
    return "tercile" in getattr(da, "dims", ())


def _apply_skill_mask(da, skill_mask, *, context="plot_matrix"):
    """Blank (NaN) cells where ``skill_mask`` is True, aligned to ``da``'s grid."""
    from .._spatial import spatial_dims

    lat_dim, lon_dim = spatial_dims(da, context=f"{context} skill_mask target")
    aligned = _align_bool_mask(skill_mask, da[lat_dim].values, da[lon_dim].values,
                               name="skill_mask")
    mask_da = xr.DataArray(aligned, dims=(lat_dim, lon_dim),
                           coords={lat_dim: da[lat_dim], lon_dim: da[lon_dim]})
    return da.where(~mask_da)


def _panel_extent(da, style):
    from .._spatial import spatial_dims

    if style is not None and style.extent is not None:
        return tuple(style.extent)
    lat_dim, lon_dim = spatial_dims(da, context="plot_matrix")
    lat = np.asarray(da[lat_dim].values)
    lon = np.asarray(da[lon_dim].values)
    return (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))


def _grid_axes(plt, nrows, ncols, figsize):
    """Best-available grid of axes: cartopy GeoAxes if importable, else plain."""
    try:
        ccrs = importlib.import_module("cartopy.crs")
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                                 subplot_kw={"projection": ccrs.PlateCarree()},
                                 layout="constrained", squeeze=False)
        return fig, axes, True
    except Exception:
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                                 layout="constrained", squeeze=False)
        return fig, axes, False


def _panel_chip(ax, text, *, bold=False, fontsize=8):
    """Panel label as a white chip inside the axes — cartopy's labelled
    gridliner reserves the title band, so in-axes chips keep grids compact."""
    ax.text(0.02, 0.98, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=fontsize, fontweight="bold" if bold else "normal",
            zorder=10,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                      boxstyle="round,pad=0.25"))


def _declutter(ax, is_geo):
    """Strip per-panel tick/gridline labels; chips carry the identification."""
    if is_geo:
        # Cartopy keeps Gridliners in ax._gridliners (<0.25) or ax.artists
        # (>=0.25); duck-type on the label flags rather than the class.
        liners = list(getattr(ax, "_gridliners", [])) + [
            a for a in ax.artists if hasattr(a, "top_labels")]
        for gl in liners:
            gl.top_labels = gl.bottom_labels = False
            gl.left_labels = gl.right_labels = False
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")


def tercile_legend_handles(style=None, *, variable_kind="precip", detailed=False,
                           include_dry=None, dry_label="Dry-masked / no data"):
    """Legend handles for a tercile palette, for figure- or axes-level legends.

    With no ``style``, three saturated patches follow the ``variable_kind``
    convention (precip: below = red; temp: below = blue). With a ``style``,
    ``detailed=False`` gives one patch per category (strongest color) and
    ``detailed=True`` one patch per probability band per category, labelled
    with the band ("40–50%", ">70%"). ``include_dry`` appends a dry-mask patch
    (defaults to whether the style carries a ``dry_mask``). Returns a list of
    Matplotlib ``Patch`` handles.
    """
    require_optional("matplotlib", _HINT)
    Patch = importlib.import_module("matplotlib.patches").Patch

    if variable_kind == "precip":
        below_label, above_label = "Below normal (drier)", "Above normal (wetter)"
        below_sat, above_sat = "#d7301f", "#2166ac"
    elif variable_kind == "temp":
        below_label, above_label = "Below normal (cooler)", "Above normal (warmer)"
        below_sat, above_sat = "#2166ac", "#d7301f"
    else:
        raise ValueError(
            f"variable_kind must be 'precip' or 'temp', got {variable_kind!r}")

    if style is None:
        return [
            Patch(facecolor=below_sat, edgecolor="black", linewidth=0.5, label=below_label),
            Patch(facecolor="#999999", edgecolor="black", linewidth=0.5, label="Near normal"),
            Patch(facecolor=above_sat, edgecolor="black", linewidth=0.5, label=above_label),
        ]

    if include_dry is None:
        include_dry = style.dry_mask is not None

    if not detailed:
        handles = [
            Patch(facecolor=style.below_colors[-1], edgecolor="#666666",
                  linewidth=0.3, label=below_label),
            Patch(facecolor=style.normal_colors[len(style.normal_colors) // 2],
                  edgecolor="#666666", linewidth=0.3, label="Near normal"),
            Patch(facecolor=style.above_colors[-1], edgecolor="#666666",
                  linewidth=0.3, label=above_label),
        ]
        if include_dry:
            handles.append(Patch(facecolor=style.dry_color, edgecolor="#666666",
                                 linewidth=0.3, label=dry_label))
        return handles

    def _band(lo, hi):
        return f"{lo:.0f}–{hi:.0f}%" if hi <= 100 else f">{lo:.0f}%"

    handles = []
    for cat_name, colors in (("Below", style.below_colors),
                             ("Normal", style.normal_colors),
                             ("Above", style.above_colors)):
        handles.extend(
            Patch(facecolor=c, edgecolor="#666666", linewidth=0.3,
                  label=f"{cat_name} {_band(lo, hi)}")
            for c, lo, hi in zip(colors, style.prob_bins[:-1], style.prob_bins[1:]))
    if include_dry:
        handles.append(Patch(facecolor=style.dry_color, edgecolor="#666666",
                             linewidth=0.3, label=dry_label))
    return handles


def _plot_grid(entries, *, emphasize=None, style=None, ncols=3, skill_mask=None,
               smooth=False, variable_kind="precip", legend=True,
               legend_detailed=False, cmap="RdBu_r", levels=None, vmin=None,
               vmax=None, center=None, extend="both", cbar_label=None,
               suptitle=None, panel_size=(4.2, 3.9), figsize=None):
    require_optional("matplotlib", _HINT)
    plt = importlib.import_module("matplotlib.pyplot")
    Patch = importlib.import_module("matplotlib.patches").Patch

    if skill_mask is not None:
        entries = [(t, _apply_skill_mask(da, skill_mask), st)
                   for t, da, st in entries]

    n = len(entries)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    if figsize is None:
        figsize = (panel_size[0] * ncols, panel_size[1] * nrows)
    fig, axes, is_geo = _grid_axes(plt, nrows, ncols, figsize)
    flat = axes.ravel()

    # One scale across every continuous panel, so the shared colorbar is
    # honest. The range is computed over the cells that will actually show:
    # values under the style's clip/dry masks are excluded, else an extreme in
    # a to-be-blanked region would waste most of the colormap.
    field_entries = [(t, da, st) for t, da, st in entries if not _is_tercile(da)]
    eff_vmin, eff_vmax = vmin, vmax
    if field_entries and levels is None and (vmin is None or vmax is None):
        finite = []
        for _, da, entry_style in field_entries:
            pstyle = entry_style if entry_style is not None else style
            if pstyle is not None and (pstyle.dry_mask is not None
                                       or pstyle.clip_to is not None):
                dry_m, outside_m = region_masks(da, pstyle)
                da = da.where(~(dry_m | outside_m))
            v = np.asarray(da.values, dtype=float)
            v = v[np.isfinite(v)]
            if v.size:
                finite.append(v)
        if finite:
            lo = min(float(v.min()) for v in finite)
            hi = max(float(v.max()) for v in finite)
            if center is not None and vmin is None and vmax is None:
                half = max(hi - center, center - lo, 1e-12)
                eff_vmin, eff_vmax = center - half, center + half
            else:
                eff_vmin = vmin if vmin is not None else lo
                eff_vmax = vmax if vmax is not None else hi

    field_mappable, field_axes, any_tercile = None, [], False
    for i, (title, da, entry_style) in enumerate(entries):
        ax = flat[i]
        pstyle = entry_style if entry_style is not None else style
        if is_geo:
            ccrs = importlib.import_module("cartopy.crs")
            ax.set_extent(_panel_extent(da, pstyle), crs=ccrs.PlateCarree())
        if _is_tercile(da):
            any_tercile = True
            plot_tercile_forecast(da, style=pstyle, ax=ax, legend=False,
                                  variable_kind=variable_kind, smooth=smooth)
            ax.set_title("")
        else:
            im = plot_field(da, style=pstyle, ax=ax, cmap=cmap,
                            vmin=eff_vmin if levels is None else None,
                            vmax=eff_vmax if levels is None else None,
                            center=None if levels is not None else center,
                            levels=levels, extend=extend, smooth=smooth,
                            title="")
            field_axes.append(ax)
            if im is not None:
                field_mappable = im
        _declutter(ax, is_geo)
        bold = emphasize is not None and i == emphasize
        _panel_chip(ax, title, bold=bold)
        if bold:
            for spine in ax.spines.values():
                spine.set_linewidth(1.8)
                spine.set_edgecolor("#000000")
    for j in range(n, nrows * ncols):
        flat[j].set_visible(False)

    if field_mappable is not None and field_axes:
        # Anchor to every visible panel, not just the continuous ones:
        # constrained layout takes the colorbar's space from the axes it is
        # attached to, and in a mixed grid stealing from only the field panels
        # would shrink their column below the tercile panels beside it.
        cb = fig.colorbar(field_mappable, ax=list(flat[:n]),
                          orientation="horizontal",
                          shrink=0.8, pad=0.02, aspect=45, label=cbar_label)
        cb.ax.tick_params(labelsize=8)

    if legend and any_tercile:
        handles = tercile_legend_handles(style, variable_kind=variable_kind,
                                         detailed=legend_detailed)
        if legend_detailed and style is not None:
            # Column-per-category layout: legends fill top-to-bottom, so pad the
            # dry patch's column with invisible spacers to keep columns aligned.
            n_bins = len(style.prob_bins) - 1
            leg_ncol = 3
            if len(handles) > 3 * n_bins:            # dry patch appended
                spacer = Patch(facecolor="none", edgecolor="none", label="")
                handles.extend([spacer] * (n_bins - 1))
                leg_ncol = 4
        else:
            leg_ncol = min(len(handles), 4)
        fig.legend(handles=handles, loc="outside lower center", ncol=leg_ncol,
                   frameon=True, framealpha=0.9, fontsize=8,
                   title="Probability of dominant tercile"
                         if style is not None else
                         "Dominant tercile  (intensity = confidence)",
                   title_fontsize=8)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    return fig


def plot_matrix(panels, *, style=None, ncols=3, skill_mask=None, smooth=False,
                variable_kind="precip", legend=True, legend_detailed=False,
                cmap="RdBu_r", levels=None, vmin=None, vmax=None, center=None,
                extend="both", cbar_label=None, suptitle=None,
                panel_size=(4.2, 3.9), figsize=None):
    """Grid of forecast maps — tercile, continuous, or a mix — with shared styling.

    The one-call version of the per-model figure grids every replication
    workflow builds by hand: raw forecasts per model, the same models after
    CCA, ensemble members, ours-vs-reference rosters.

    Parameters
    ----------
    panels : dict or iterable
        ``{title: data}`` or an iterable of ``(title, data)`` /
        ``(title, data, style)``. Each ``data`` is a DataArray: dims
        ``(tercile, lat, lon)`` render as a styled dominant-tercile map, plain
        ``(lat, lon)`` as a continuous field. A per-panel style overrides the
        figure ``style`` (e.g. one panel with the dry mask switched off).
    style : TercileStyle or None
        Palette, probability bins, dry mask, country clip, lakes, and extent,
        applied to every panel that doesn't carry its own.
    ncols : int
        Grid columns; rows are ``ceil(n / ncols)``. Trailing cells are hidden.
    skill_mask : DataArray or ndarray or None
        Bool mask, True = insufficient skill: those cells are blanked (nodata)
        on every panel at display time. A coordinate-bearing (lat/lon)
        DataArray is aligned to each panel's grid; a bare array must match.
        For mask-before-combine semantics use :func:`africas2s.mask_by_skill`
        on the data instead.
    smooth : bool or int
        Cubic-refined contour rendering (GHACOF look) for every panel;
        continuous panels then require ``levels``. ``True`` = factor 4.
    variable_kind : {"precip", "temp"}
        Category color convention for tercile panels and the legend.
    legend, legend_detailed : bool
        Shared figure-level tercile legend below the grid (only when at least
        one panel is a tercile map); ``legend_detailed`` shows every
        probability band rather than one patch per category.
    cmap, levels, vmin, vmax, center, extend, cbar_label
        Continuous-panel scale, shared across all continuous panels (a single
        shared colorbar). With no explicit scale, vmin/vmax are computed over
        all continuous panels (symmetric about ``center`` when given).
    suptitle : str or None
        Bold figure title.
    panel_size : (float, float)
        Per-panel size in inches; ``figsize`` overrides the computed size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    entries = _normalize_panels(panels)
    return _plot_grid(entries, style=style, ncols=ncols, skill_mask=skill_mask,
                      smooth=smooth, variable_kind=variable_kind, legend=legend,
                      legend_detailed=legend_detailed, cmap=cmap, levels=levels,
                      vmin=vmin, vmax=vmax, center=center, extend=extend,
                      cbar_label=cbar_label, suptitle=suptitle,
                      panel_size=panel_size, figsize=figsize)


def plot_components_objective(components, objective, *, objective_label="Objective",
                              ncols=2, **kwargs):
    """Component forecasts alongside the final objective, objective emphasized.

    The standard "what went in, what came out" composite: every component
    forecast (per-model, per-method, or per-experiment) in a grid with the
    combined objective as the last panel, drawn with a bold label and a
    heavier frame so it reads as the headline.

    ``components`` is a dict or ``(title, data[, style])`` iterable as in
    :func:`plot_matrix`; ``objective`` is the combined forecast DataArray.
    All other keywords are :func:`plot_matrix`'s. Returns the Figure.
    """
    entries = _normalize_panels(components, context="plot_components_objective")
    entries.append((str(objective_label), objective, None))
    return _plot_grid(entries, emphasize=len(entries) - 1, ncols=ncols, **kwargs)
