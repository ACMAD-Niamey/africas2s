"""Spatial maps: gridded fields and per-region choropleths.

Two figures the monitoring decks lean on and africas2s could not yet draw:

:func:`plot_field_map` renders a gridded ``(lat, lon)`` field — a percentile map,
a rank-of-record map, a seasonal total. It carries a *highlight* mode for the
"driest / wettest on record" pixels the Early Estimates maps mark in red.

:func:`plot_choropleth` fills administrative polygons by a per-region value —
the SMPG woreda percentile maps. It takes the output of
:func:`rosetta.zonal` (a value per ``region``) plus the geometries, and needs no
knowledge of what admin level or country it is drawing.

Both are colormap-and-geometry helpers, not analyses: they render a DataArray
someone else computed. Coastlines/borders use cartopy when it is installed and
degrade to a plain lat/lon axis (with optional geometry outlines) when it is
not, so the pixel path has no hard cartopy dependency.
"""
from __future__ import annotations

import importlib

import numpy as np

from .._optional import require_optional

_HINT = "pip install africas2s[plotting]"

# Percentile / probability fields live in [0, 1]. A perceptually flat sequential
# map reads "low is bad" unambiguously for rainfall; the caller can override.
_PERCENTILE_CMAP = "YlGnBu"
# The deck marks the record-driest pixels in a saturated red. One colour, so it
# reads as a category ("on record") not a continuum.
_RECORD_COLOR = "#c0272d"


def _pyplot():
    require_optional("matplotlib", _HINT)
    return importlib.import_module("matplotlib.pyplot")


def _classified(classes):
    """Build a discrete colour scheme from ``(bounds, colors[, labels])``.

    ``bounds`` has one more entry than ``colors``: N colours fill the N
    intervals between N+1 boundaries. An optional ``labels`` (length N) names the
    classes on the colour bar; without it the intervals are labelled by their
    edges. Returns everything a classified `pcolormesh`/choropleth needs.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap

    if len(classes) == 3:
        bounds, colors, labels = classes
    else:
        bounds, colors = classes
        labels = None
    if len(bounds) != len(colors) + 1:
        raise ValueError(
            f"classes needs len(bounds) == len(colors) + 1; got "
            f"{len(bounds)} bounds and {len(colors)} colors."
        )
    cmap = ListedColormap(list(colors))
    norm = BoundaryNorm(list(bounds), cmap.N)
    centers = [(bounds[i] + bounds[i + 1]) / 2 for i in range(len(colors))]
    if labels is None:
        labels = [f"{bounds[i]:g}–{bounds[i + 1]:g}" for i in range(len(colors))]
    return cmap, norm, list(bounds), centers, list(labels)


def _try_cartopy():
    try:
        import cartopy.crs as ccrs  # noqa: F401
        import cartopy.feature as cfeature  # noqa: F401
        return importlib.import_module("cartopy.crs"), importlib.import_module(
            "cartopy.feature")
    except ImportError:
        return None, None


def _geo_axes(ax, figsize):
    """A (plt, fig, ax, ccrs-or-None) tuple; a cartopy GeoAxes if available."""
    plt = _pyplot()
    ccrs, _ = _try_cartopy()
    if ax is not None:
        return plt, ax.figure, ax, ccrs
    if ccrs is not None:
        fig, ax = plt.subplots(figsize=figsize,
                               subplot_kw={"projection": ccrs.PlateCarree()})
    else:
        fig, ax = plt.subplots(figsize=figsize)
    return plt, fig, ax, ccrs


def _add_basemap(ax, ccrs):
    if ccrs is None:
        return
    _, cfeature = _try_cartopy()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#666666")


# Natural Earth admin-0 polygons, as cached on disk by cartopy. Reused (rather
# than re-downloaded) so a plain-matplotlib install can still draw country
# outlines on a multi-country map without pulling in cartopy.
from pathlib import Path as _Path

_NE_COUNTRIES = {
    "10m": _Path.home() / ".local/share/cartopy/shapefiles/natural_earth/cultural/ne_10m_admin_0_countries.shp",
    "50m": _Path.home() / ".local/share/cartopy/shapefiles/natural_earth/cultural/ne_50m_admin_0_countries.shp",
}


def natural_earth_borders(region=None, *, scale="50m"):
    """Load Natural Earth admin-0 country polygons as a GeoDataFrame.

    A convenience for drawing country outlines on a multi-country field map
    (pass the result as ``plot_field_map(..., boundaries=...)``). It reuses the
    Natural Earth shapefiles cartopy caches on disk, so it works on a
    plain-matplotlib install with no cartopy import.

    Parameters
    ----------
    region : sequence ``[lat_s, lat_n, lon_w, lon_e]``, optional
        Keep only countries intersecting this box. ``None`` returns the world.
    scale : {"50m", "10m"}
        Natural Earth resolution. ``"50m"`` is the lighter default.

    Returns
    -------
    GeoDataFrame with a single ``geometry`` column, in EPSG:4326.

    Raises
    ------
    FileNotFoundError
        If the cached shapefile is absent — install cartopy once (or place the
        Natural Earth ``ne_<scale>_admin_0_countries`` files at the cached path)
        to populate it.
    """
    gpd = _require_geopandas()
    path = _NE_COUNTRIES.get(scale)
    if path is None:
        raise ValueError(f"scale must be one of {sorted(_NE_COUNTRIES)}, got {scale!r}")
    if not path.exists():
        raise FileNotFoundError(
            f"Natural Earth admin-0 shapefile not found at {path}. Install cartopy "
            "once to populate its shapefile cache, or drop the "
            f"ne_{scale}_admin_0_countries.* files there."
        )
    gdf = gpd.read_file(path)[["geometry"]]
    if region is not None:
        lat_s, lat_n, lon_w, lon_e = region
        gdf = gdf.cx[lon_w:lon_e, lat_s:lat_n]
    return gdf.reset_index(drop=True)


def _transform(ccrs):
    return {"transform": ccrs.PlateCarree()} if ccrs is not None else {}


def plot_field_map(
    da,
    *,
    ax=None,
    cmap=None,
    vmin=None,
    vmax=None,
    classes=None,
    clip=None,
    highlight=None,
    highlight_label="driest on record",
    boundaries=None,
    title=None,
    cbar_label=None,
    figsize=(8, 7),
):
    """Render a gridded ``(lat, lon)`` field as a map.

    Parameters
    ----------
    da : xr.DataArray
        Must have lat/lon dims (any of the usual aliases). Any extra dims must
        already be reduced to a single 2-D slice.
    clip : shapefile path / GeoDataFrame / GeoSeries / geometry, optional
        Mask cells outside this region to NaN, so the map shows the region's
        true outline rather than the fetch bounding box. Multi-feature inputs
        (e.g. all woredas) are dissolved to their union (the country).
    classes : tuple, optional
        A discrete classification ``(bounds, colors[, labels])`` — ``bounds`` is
        N+1 breakpoints, ``colors`` the N fill colours, ``labels`` their optional
        class names. Draws a stepped colour bar instead of a continuous ramp,
        matching how operational rank / percentile maps are shown. Overrides
        ``cmap``/``vmin``/``vmax``.
    highlight : float, optional
        Overpaint cells equal to this value in a single saturated colour — the
        "driest on record" convention (pass ``highlight=1`` over a
        :func:`africas2s.rank_of_record` field). NaN-safe.
    boundaries : GeoDataFrame / GeoSeries, optional
        Admin outlines to overlay (drawn as unfilled edges).
    cmap, vmin, vmax : optional
        Default to a sequential map on ``[0, 1]`` when the data looks like a
        percentile field, else matplotlib's autoscaling. Ignored if ``classes``
        is given.

    Returns
    -------
    matplotlib Figure
    """
    from .._spatial import spatial_dims

    lat, lon = spatial_dims(da, context="plot_field_map")
    if da.ndim != 2:
        raise ValueError(
            f"plot_field_map needs a 2-D (lat, lon) field; got dims {tuple(da.dims)}. "
            "Select or reduce the extra dimensions first."
        )

    plt, fig, ax, ccrs = _geo_axes(ax, figsize)
    da = da.sortby([lat, lon])
    if clip is not None:
        da = _mask_to_geometry(da, _resolve_geometry(clip), lat, lon)
    values = da.transpose(lat, lon).values

    # Match the colorbar height to the *drawn* map, which shrinks under equal
    # aspect: without this the bar is sized to the pre-aspect axes and towers
    # over a map that is wider than it is tall. 0.046 is the usual square-axes
    # fraction; scaling by the data's height/width ratio tracks any extent.
    cbar_fraction = 0.046 * (values.shape[0] / max(values.shape[1], 1))

    if classes is not None:
        cmap_obj, norm, bounds, centers, labels = _classified(classes)
        mesh = ax.pcolormesh(da[lon].values, da[lat].values, values,
                             cmap=cmap_obj, norm=norm, **_transform(ccrs))
        cb = fig.colorbar(mesh, ax=ax, fraction=cbar_fraction, pad=0.04,
                          boundaries=bounds, ticks=centers,
                          label=cbar_label or (da.name or "value"))
        cb.ax.set_yticklabels(labels)
    else:
        looks_like_fraction = np.nanmin(values) >= 0.0 and np.nanmax(values) <= 1.0
        if cmap is None:
            cmap = _PERCENTILE_CMAP if looks_like_fraction else "viridis"
        if vmin is None and looks_like_fraction:
            vmin, vmax = 0.0, 1.0
        mesh = ax.pcolormesh(da[lon].values, da[lat].values, values,
                             cmap=cmap, vmin=vmin, vmax=vmax, **_transform(ccrs))
        fig.colorbar(mesh, ax=ax, fraction=cbar_fraction, pad=0.04,
                     label=cbar_label or (da.name or "value"))

    if highlight is not None:
        mask = np.isclose(values, float(highlight))
        if mask.any():
            painted = np.where(mask, 1.0, np.nan)
            from matplotlib.colors import ListedColormap
            ax.pcolormesh(da[lon].values, da[lat].values, painted,
                          cmap=ListedColormap([_RECORD_COLOR]),
                          vmin=0, vmax=1, **_transform(ccrs))
            ax.plot([], [], "s", color=_RECORD_COLOR, label=highlight_label)
            ax.legend(loc="upper right", frameon=True, fontsize=8)

    _overlay_boundaries(ax, boundaries, ccrs)
    _add_basemap(ax, ccrs)
    if title:
        ax.set_title(title)
    if ccrs is None:
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


def plot_choropleth(
    values,
    geometries,
    *,
    by=None,
    ax=None,
    cmap=None,
    vmin=None,
    vmax=None,
    classes=None,
    missing_color="#e8e8e8",
    edgecolor="#ffffff",
    linewidth=0.2,
    title=None,
    cbar_label=None,
    figsize=(8, 8),
):
    """Fill admin polygons by a per-``region`` value.

    Parameters
    ----------
    values : xr.DataArray
        Indexed by a ``region`` dim — the output of :func:`rosetta.zonal`. Its
        ``region`` labels are matched to ``geometries`` on ``by``.
    geometries : GeoDataFrame
        One row per region.
    by : str, optional
        Column of ``geometries`` holding the region key that matches
        ``values``'s ``region`` coordinate. Defaults to the GeoDataFrame index.
    classes : tuple, optional
        Discrete classification ``(bounds, colors[, labels])`` — as in
        :func:`plot_field_map`. Draws a stepped legend; overrides
        ``cmap``/``vmin``/``vmax``.
    missing_color : colour
        Fill for regions with no value (NaN) — a district the grid never
        covered. Drawn, not dropped, so the map has no holes.

    Returns
    -------
    matplotlib Figure
    """
    require_optional("matplotlib", _HINT)
    gpd = _require_geopandas()
    plt = _pyplot()

    region_dim = "region" if "region" in values.dims else values.dims[0]
    keys = values[region_dim].values
    frame = geometries.copy()
    if by is None:
        lookup = {i: float(v) for i, v in zip(range(len(frame)), np.asarray(values.values, float))}
        frame["_value"] = [lookup.get(i, np.nan) for i in range(len(frame))]
    else:
        if by not in frame.columns:
            raise ValueError(
                f"column {by!r} not found in geometries; have {list(frame.columns)}"
            )
        lookup = {k: float(v) for k, v in zip(keys, np.asarray(values.values, float))}
        frame["_value"] = frame[by].map(lookup)

    if frame.crs is not None and frame.crs.to_epsg() != 4326:
        frame = frame.to_crs("EPSG:4326")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Size the colorbar to the drawn (equal-aspect) map -- see plot_field_map.
    minx, miny, maxx, maxy = frame.total_bounds
    cbar_fraction = 0.046 * ((maxy - miny) / max(maxx - minx, 1e-9))
    plot_kw = dict(column="_value", ax=ax, edgecolor=edgecolor, linewidth=linewidth,
                   legend=True,
                   legend_kwds={"label": cbar_label or (values.name or "value"),
                                "fraction": cbar_fraction, "pad": 0.04},
                   missing_kwds={"color": missing_color, "edgecolor": edgecolor,
                                 "linewidth": linewidth, "label": "no data"})
    if classes is not None:
        cmap_obj, norm, bounds, centers, labels = _classified(classes)
        plot_kw["legend_kwds"].update(boundaries=bounds, ticks=centers)
        frame.plot(cmap=cmap_obj, norm=norm, **plot_kw)
        # geopandas returns no handle to the colorbar; relabel via the last axis.
        cb_ax = fig.axes[-1]
        cb_ax.set_yticklabels(labels)
    else:
        finite = frame["_value"].to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        looks_like_fraction = finite.size and finite.min() >= 0.0 and finite.max() <= 1.0
        if cmap is None:
            cmap = _PERCENTILE_CMAP if looks_like_fraction else "viridis"
        if vmin is None and looks_like_fraction:
            vmin, vmax = 0.0, 1.0
        frame.plot(cmap=cmap, vmin=vmin, vmax=vmax, **plot_kw)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def _overlay_boundaries(ax, boundaries, ccrs):
    if boundaries is None:
        return
    gpd = _require_geopandas()
    frame = boundaries
    if hasattr(frame, "geometry"):
        if getattr(frame, "crs", None) is not None and frame.crs.to_epsg() != 4326:
            frame = frame.to_crs("EPSG:4326")
        kw = {"ax": ax, "facecolor": "none", "edgecolor": "#333333", "linewidth": 0.4}
        if ccrs is not None:
            kw["transform"] = ccrs.PlateCarree()
        frame.boundary.plot(**kw)


def _require_geopandas():
    try:
        return importlib.import_module("geopandas")
    except ImportError as e:
        raise ImportError(
            "Choropleths and boundary overlays need geopandas. "
            "Install the plotting + geo extras."
        ) from e


def _resolve_geometry(clip):
    """A single EPSG:4326 shapely geometry from a shapefile / gdf / geoseries / geom."""
    gpd = _require_geopandas()
    if isinstance(clip, str) or hasattr(clip, "__fspath__"):
        gdf = gpd.read_file(clip)
    elif isinstance(clip, gpd.GeoDataFrame):
        gdf = clip
    elif isinstance(clip, gpd.GeoSeries):
        gdf = gpd.GeoDataFrame(geometry=clip)
    else:
        return clip  # assume a bare shapely geometry, already EPSG:4326
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf.union_all()   # dissolve to one boundary (e.g. woredas -> country)


def _mask_to_geometry(da, geometry, lat, lon):
    """Set cells whose centre falls outside ``geometry`` to NaN.

    The map then shows the region's true shape instead of the fetch bounding
    box, matching how operational products are drawn.
    """
    import xarray as xr
    from rasterio.features import geometry_mask
    from rasterio.transform import from_bounds

    da = da.sortby([lat, lon]).transpose(lat, lon)
    latv, lonv = da[lat].values, da[lon].values
    if latv.size < 2 or lonv.size < 2:
        return da
    dlat, dlon = float(latv[1] - latv[0]), float(lonv[1] - lonv[0])
    west, east = lonv[0] - dlon / 2, lonv[-1] + dlon / 2
    south, north = latv[0] - dlat / 2, latv[-1] + dlat / 2
    transform = from_bounds(west, south, east, north, len(lonv), len(latv))
    # geometry_mask: True OUTSIDE the shapes. from_bounds is north-up (row 0 =
    # north); our latitude is ascending (row 0 = south), so flip to align.
    outside = np.flipud(geometry_mask([geometry], out_shape=(len(latv), len(lonv)),
                                      transform=transform, invert=False))
    inside = xr.DataArray(~outside, dims=(lat, lon), coords={lat: latv, lon: lonv})
    return da.where(inside)
