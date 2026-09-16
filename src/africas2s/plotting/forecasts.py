"""Forecast output plots: tercile, deterministic, exceedance, flex-PDF."""

from pathlib import Path

import numpy as np
import xarray as xr

from .._optional import require_optional


_HINT = "pip install africas2s[plotting]"

# Dominant-tercile color saturation: probability above 1/3 at which
# the color reaches full intensity. 0.37 = 70% probability cap.
_TERCILE_SAT = 0.37

# Natural Earth coastline/border shapefiles, as cached by cartopy. Used to draw
# a basemap via geopandas when cartopy itself is not installed.
_NE_ROOT = Path.home() / ".local" / "share" / "cartopy" / "shapefiles" / "natural_earth"
_NE_COAST = _NE_ROOT / "physical" / "ne_50m_coastline.shp"
_NE_BORDERS = _NE_ROOT / "cultural" / "ne_50m_admin_0_boundary_lines_land.shp"


def _no_dominant(values, valid, prob_bins, secondary_max=None):
    """Cells whose leading category does not qualify as dominant.

    Two rules, both expressed in percent. The leading probability must reach
    the first bin edge ``prob_bins[0]`` — with the GHACOF/ACMAD 33.3 edge that
    is always true (the largest of three probabilities is at least a third),
    with ICPAC's 40 or NMME's 38 it is the "no dominant category" threshold.
    With ``secondary_max`` set, the OTHER OUTER tercile(s) must also be under
    that value. NOAA CPC's NMME rule names the opposite class, not every other
    category: "A and B contours show when one class has >38% of ensemble
    members, and the opposite class is below 33%. In the case that A is >38%
    and N is >33%, A will be shown." So a near-normal runner-up never blanks an
    outer tercile; when normal leads, both outer terciles must clear the bar.
    That also reproduces NOAA's white cases exactly: every tercile under the
    first bin edge, or both outer terciles above it.

    ``values`` is (tercile, ...) in (below, normal, above) order. Returns a
    bool array; cells that are not a valid triple are reported False here
    (they are excluded elsewhere).
    """
    finite = np.where(np.isfinite(values), values, -np.inf)
    pct = finite * 100.0
    leading = np.max(pct, axis=0)
    with np.errstate(invalid="ignore"):
        weak = valid & (leading < prob_bins[0])
        if secondary_max is not None:
            lead_i = np.argmax(pct, axis=0)
            below, above = pct[0], pct[2]
            # below leads -> above constrains; above leads -> below constrains;
            # normal leads -> both must be under the bar.
            rival = np.where(lead_i == 0, above,
                             np.where(lead_i == 2, below, np.maximum(below, above)))
            weak = weak | (valid & (rival >= secondary_max))
    return weak


def _no_dominant_label(style):
    """Legend text for the white "no dominant category" cells, or None when
    the style's rules never leave a valid cell unfilled."""
    if style is None:
        return None
    threshold = float(style.prob_bins[0]) > 100.0 / 3.0 + 1e-6
    secondary = getattr(style, "secondary_max", None) is not None
    if not (threshold or secondary):
        return None
    parts = []
    if threshold:
        parts.append(f"leading < {float(style.prob_bins[0]):g}%")
    if secondary:
        parts.append(f"opposing tercile ≥ {float(style.secondary_max):g}%")
    return "No dominant category (" + " or ".join(parts) + ")"


def _tercile_codes(probs, prob_bins, secondary_max=None):
    """Integer class code per cell for a discrete tercile palette.

    ``probs`` is (tercile=3, lat, lon), tercile order (below, normal, above),
    fractional (0-1). Returns (code, valid). n = len(prob_bins)-1 bins:
    above -> 0..n-1, normal -> n..2n-1, below -> 2n..3n-1; -1 = no valid triple
    or no dominant category (see ``_no_dominant``).
    """
    valid = np.isfinite(probs).all(axis=0)
    valid = valid & ~_no_dominant(probs, valid, prob_bins, secondary_max)
    dom = np.argmax(np.where(np.isfinite(probs), probs, -1.0), axis=0)  # 0=below 1=normal 2=above
    prob_pct = np.where(valid, np.max(probs, axis=0) * 100.0, np.nan)
    n = len(prob_bins) - 1
    pbin = np.clip(np.digitize(prob_pct, prob_bins) - 1, 0, n - 1)
    base = {2: 0, 1: n, 0: 2 * n}
    code = np.full(prob_pct.shape, -1, dtype=int)
    for d, b in base.items():
        sel = valid & (dom == d)
        code[sel] = b + pbin[sel]
    return code, valid


def _new_fig(ax, figsize=(8, 5)):
    import importlib
    require_optional("matplotlib", _HINT)
    plt = importlib.import_module("matplotlib.pyplot")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return plt, fig, ax


def _tercile_rgb(probs, red_cat, blue_cat):
    """Dominant-tercile RGB image. `probs` is (tercile=3, lat, lon).

    Cells that are not a complete finite tercile triple (significance-masked or
    uncalibratable cells arrive as all-NaN) are left blank (white) rather than
    painted into a category. Without this, ``argmax``/``max`` over an all-NaN
    cell return category 0 / NaN, so masked cells would render as a confident
    below/above colour (or NaN pixels) instead of as "no valid forecast".
    """
    valid = np.isfinite(probs).all(axis=0)
    dom_cat = probs.argmax(axis=0)
    dom_prob = probs.max(axis=0)
    intensity = np.clip((dom_prob - 1 / 3) / _TERCILE_SAT, 0.0, 1.0)

    rgb = np.ones(dom_cat.shape + (3,))
    is_red = (dom_cat == red_cat) & valid
    is_blue = (dom_cat == blue_cat) & valid
    is_normal = (dom_cat == 1) & valid
    rgb[is_red] = np.stack(
        [np.ones(is_red.sum()), 1 - intensity[is_red], 1 - intensity[is_red]], axis=-1
    )
    rgb[is_blue] = np.stack(
        [1 - intensity[is_blue], 1 - intensity[is_blue], np.ones(is_blue.sum())], axis=-1
    )
    rgb[is_normal] = np.stack(
        [1 - 0.4 * intensity[is_normal]] * 3, axis=-1
    )
    return rgb


def _make_tercile_axes(plt, extent, figsize):
    """Best-available single-map axes: a cartopy GeoAxes if cartopy is
    importable, otherwise a plain axes. Returns (fig, ax, is_geo)."""
    try:
        import importlib
        ccrs = importlib.import_module("cartopy.crs")
    except Exception:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax, False
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    return fig, ax, True


def _draw_cartopy_basemap(ax):
    import importlib
    ccrs = importlib.import_module("cartopy.crs")
    ax.coastlines(resolution="50m", linewidth=0.8, color="#333333")
    # Filtered NE boundary lines instead of cfeature.BORDERS: suppresses lines
    # internal to a merged territory group (e.g. Somalia-Somaliland), matching
    # the UN/GHACOF depiction. Everything else is identical to BORDERS.
    ax.add_geometries(_neutral_border_geoms("50m"), ccrs.PlateCarree(),
                      facecolor="none", edgecolor="#555555", linewidth=0.6)
    # Ask for the bottom/left labels up front rather than enabling all four
    # and switching top/right off afterwards: with cartopy 0.25 on matplotlib
    # >= 3.10, toggling a label side off after creation leaves the axes with a
    # NaN tight bounding box, which crops every bbox_inches="tight" save (and
    # the notebook inline render) to the colorbar/legend. Older cartopy takes
    # only a bool, in which case the toggle is still the way.
    try:
        gl = ax.gridlines(draw_labels=["bottom", "left"], linewidth=0.3,
                          color="#777777", alpha=0.5)
    except (TypeError, ValueError):
        gl = None
    if gl is None or gl.top_labels or gl.right_labels:
        if gl is not None:
            gl.remove()
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#777777", alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False


def _to_0_360(gdf):
    """Shift a -180..180 GeoDataFrame into the 0-360 longitude convention.

    Natural Earth ships in -180..180; forecast grids are often 0-360. Any
    geometry sitting in the western hemisphere (min longitude < 0) is translated
    by +360 so coastlines line up with 0-360 data instead of being clipped to
    the eastern hemisphere. Geometries straddling the prime meridian are rare in
    coastline/border line data and shift wholesale, which is acceptable for a
    context basemap.
    """
    from shapely.affinity import translate

    def _shift(geom):
        return translate(geom, xoff=360.0) if geom.bounds[0] < 0 else geom

    return gdf.assign(geometry=gdf.geometry.apply(_shift))


def _draw_geopandas_basemap(ax, extent):
    """Overplot Natural Earth coastlines/borders via geopandas. Returns True if
    drawn, False if geopandas or the cached shapefiles are unavailable."""
    try:
        import importlib
        gpd = importlib.import_module("geopandas")
    except Exception:
        return False
    if not (_NE_COAST.exists() and _NE_BORDERS.exists()):
        return False
    lon_w, lon_e, lat_s, lat_n = extent
    # Natural Earth is -180..180. When the forecast grid uses the 0-360
    # convention (any longitude past 180), shift the shapefile geometries into
    # 0-360 first; otherwise .cx selects only the eastern hemisphere and clips
    # out coastlines west of the prime meridian.
    use_0_360 = lon_e > 180.0
    try:
        for path, color, lw, z in (
            (_NE_COAST, "#333333", 0.8, 3),
            (_NE_BORDERS, "#555555", 0.6, 4),
        ):
            gdf = gpd.read_file(path)
            if path is _NE_BORDERS:
                # Same territory-group suppression as the cartopy path (Somalia).
                gdf = gdf[~gdf.geometry.apply(_is_internal_border)]
            if use_0_360:
                gdf = _to_0_360(gdf)
            gdf.cx[lon_w:lon_e, lat_s:lat_n].plot(
                ax=ax, color=color, linewidth=lw, zorder=z)
    except Exception:
        return False
    return True


def _discrete_cmap(style):
    import importlib
    mcolors = importlib.import_module("matplotlib.colors")
    palette = list(style.above_colors) + list(style.normal_colors) + list(style.below_colors) + [style.dry_color]
    cmap = mcolors.ListedColormap(palette)
    cmap.set_bad(style.nodata_color)
    return cmap, len(palette)


def _tercile_style_legend(ax, style, below_label, above_label):
    import importlib
    Patch = importlib.import_module("matplotlib.patches").Patch

    def _weak(colors):
        return colors[1] if len(colors) > 1 else colors[-1]

    handles = [
        Patch(facecolor="none", edgecolor="none", label=r"$\bf{Above}$"),
        Patch(facecolor=style.above_colors[-1], label="strong"),
        Patch(facecolor=_weak(style.above_colors), label="weak"),
        Patch(facecolor=_weak(style.normal_colors), edgecolor="0.6", label="Near normal"),
        Patch(facecolor="none", edgecolor="none", label=r"$\bf{Below}$"),
        Patch(facecolor=style.below_colors[-1], label="strong"),
        Patch(facecolor=_weak(style.below_colors), label="weak"),
        Patch(facecolor=style.dry_color, label="Dry season"),
    ]
    if style.lakes:
        handles.append(Patch(facecolor=style.lake_color, label="Lake"))
    no_dominant = _no_dominant_label(style)
    if no_dominant:
        handles.append(Patch(facecolor=style.nodata_color, edgecolor="0.6",
                             label=no_dominant))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=4, frameon=True, framealpha=0.9, fontsize=7,
              title="Probability of category", title_fontsize=8)


_COUNTRY_GEOM_CACHE = {}

# (sorted clip names, lat bytes, lon bytes) -> bool "outside" array. See the
# clip branch of _region_masks; entries are treated as immutable.
_OUTSIDE_MASK_CACHE = {}

# Territories that Natural Earth ships as separate admin_0 records but that are
# rendered as part of another country on UN/WMO-style operational maps (the
# GHACOF depiction). Requesting the parent in ``clip_to`` includes the members,
# and the internal boundary between them is suppressed from the drawn borders.
_TERRITORY_GROUPS = {"Somalia": ("Somaliland",)}


def _expand_territories(names):
    """Add territory-group members whenever their parent country is requested."""
    expanded = set(names)
    for parent, members in _TERRITORY_GROUPS.items():
        if parent in expanded:
            expanded.update(members)
    return expanded


def _country_geometry(names):
    """Prepared union geometry of Natural Earth admin_0 countries matching `names`."""
    import importlib
    shpreader = importlib.import_module("cartopy.io.shapereader")
    unary_union = importlib.import_module("shapely.ops").unary_union
    prep = importlib.import_module("shapely.prepared").prep
    names = _expand_territories(names)
    key = tuple(sorted(names))
    if key not in _COUNTRY_GEOM_CACHE:
        rdr = shpreader.Reader(shpreader.natural_earth(
            resolution="10m", category="cultural", name="admin_0_countries"))
        geoms = [r.geometry for r in rdr.records()
                 if r.attributes.get("NAME", "") in names
                 or r.attributes.get("NAME_LONG", "") in names]
        if not geoms:
            raise ValueError(f"No Natural Earth countries matched names={names!r}")
        # buffer in degrees to include coastal border cells
        _COUNTRY_GEOM_CACHE[key] = prep(unary_union(geoms).buffer(0.3))
    return _COUNTRY_GEOM_CACHE[key]


_INTERNAL_BORDER_CACHE = {}


def _territory_interiors():
    """Shrunken union geometry per territory group, for internal-border tests.

    The negative buffer pulls the union's edge inward so a boundary line that
    runs ALONG the group's exterior (e.g. the Ethiopia-Somalia border) does not
    test as internal, while a line crossing the interior (the Somalia-Somaliland
    de-facto line) does.
    """
    if "interiors" not in _INTERNAL_BORDER_CACHE:
        import importlib
        shpreader = importlib.import_module("cartopy.io.shapereader")
        unary_union = importlib.import_module("shapely.ops").unary_union
        rdr = shpreader.Reader(shpreader.natural_earth(
            resolution="10m", category="cultural", name="admin_0_countries"))
        interiors = []
        for parent, members in _TERRITORY_GROUPS.items():
            group = {parent, *members}
            geoms = [r.geometry for r in rdr.records()
                     if r.attributes.get("NAME", "") in group]
            if geoms:
                interiors.append(unary_union(geoms).buffer(-0.05))
        _INTERNAL_BORDER_CACHE["interiors"] = interiors
    return _INTERNAL_BORDER_CACHE["interiors"]


def _is_internal_border(geom):
    """True if a boundary line lies mostly inside a merged territory group."""
    if geom.length == 0:
        return False
    for interior in _territory_interiors():
        if geom.intersection(interior).length / geom.length > 0.5:
            return True
    return False


def _neutral_border_geoms(resolution="50m"):
    """NE admin_0 boundary lines minus those internal to a territory group."""
    key = ("borders", resolution)
    if key not in _INTERNAL_BORDER_CACHE:
        import importlib
        shpreader = importlib.import_module("cartopy.io.shapereader")
        rdr = shpreader.Reader(shpreader.natural_earth(
            resolution=resolution, category="cultural",
            name="admin_0_boundary_lines_land"))
        _INTERNAL_BORDER_CACHE[key] = [
            r.geometry for r in rdr.records()
            if not _is_internal_border(r.geometry)]
    return _INTERNAL_BORDER_CACHE[key]


def _align_bool_mask(mask, lat, lon, *, name="mask"):
    """Align a boolean mask onto a ``(lat, lon)`` grid; returns a bool ndarray.

    ``mask`` may be a coordinate-bearing DataArray (``lat``/``lon`` coords):
    aligned by nearest-neighbor interpolation on coordinate VALUE, so it lands
    on the correct geographic cells regardless of the mask's resolution,
    registration offset, or latitude ordering; cells outside its coverage (NaN
    after interpolation) come back False. A bare ndarray (no coords) must match
    the grid shape exactly — there is no coordinate information to align it by.
    """
    lat = np.asarray(lat); lon = np.asarray(lon)
    shape = (lat.shape[0], lon.shape[0])
    if hasattr(mask, "interp"):
        from .._spatial import spatial_dims
        mlat, mlon = spatial_dims(mask, context=name)
        if (mlat, mlon) != ("lat", "lon"):     # accept the package's dim aliases
            mask = mask.rename({mlat: "lat", mlon: "lon"})
        dm = mask.astype(float).interp(
            lat=lat, lon=lon, method="nearest").transpose("lat", "lon")
        return np.asarray(dm.values) > 0.5   # NaN (outside coverage) -> False
    arr = np.asarray(mask, dtype=bool)
    if arr.shape != shape:
        raise ValueError(
            f"{name} ndarray shape {arr.shape} does not match the field "
            f"shape {shape}; pass a coordinate-bearing xarray DataArray "
            f"(lat/lon) to auto-align, or match the grid."
        )
    return arr


def _smooth_factor(smooth):
    """Normalize the ``smooth`` argument to an int refinement factor (0 = off)."""
    if smooth is True:
        return 4
    if not smooth:
        return 0
    factor = int(smooth)
    if factor < 2:
        raise ValueError(f"smooth must be True or an int >= 2, got {smooth!r}")
    return factor


def _refine_field(values, lat, lon, factor):
    """Cubic-refine a ``(lat, lon)`` field for smooth contouring.

    Returns ``(fine_values, fine_lat, fine_lon)``, or ``None`` when nothing is
    finite (or a dimension is a single row/column, where contouring is
    meaningless). NaN cells are nearest-filled first (the cubic spline rejects
    non-finite input) and re-masked afterwards via a linearly-interpolated
    coverage mask thresholded at 0.5, so the smoothing changes only the
    boundary shapes — it never invents data outside the original footprint.
    Grids too small for the cubic spline (< 4 points on an axis) fall back to
    linear refinement rather than erroring.
    """
    import importlib
    ndimage = importlib.import_module("scipy.ndimage")

    values = np.asarray(values, dtype=float)
    lat = np.asarray(lat); lon = np.asarray(lon)
    if lat.size < 2 or lon.size < 2:
        return None
    finite = np.isfinite(values)
    if not finite.any():
        return None
    filled = np.where(finite, values,
                      values[tuple(ndimage.distance_transform_edt(
                          ~finite, return_distances=False, return_indices=True))])
    da = xr.DataArray(filled, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    fine_lat = np.linspace(float(lat.min()), float(lat.max()), lat.size * factor)
    fine_lon = np.linspace(float(lon.min()), float(lon.max()), lon.size * factor)
    method = "cubic" if (lat.size >= 4 and lon.size >= 4) else "linear"
    fine = da.interp(lat=fine_lat, lon=fine_lon, method=method)
    keep = xr.DataArray(finite.astype(float), dims=("lat", "lon"),
                        coords={"lat": lat, "lon": lon}).interp(
        lat=fine_lat, lon=fine_lon, method="linear") > 0.5
    return fine.where(keep).values, fine_lat, fine_lon


def _region_masks(lat, lon, style):
    """Boolean (nlat, nlon) region masks derived from a ``TercileStyle``.

    Returns ``(dry, outside)``; each is a bool array, or ``None`` if the
    corresponding style field is unset:
      - ``dry``: cells to grey out, from ``style.dry_mask``.
      - ``outside``: cells outside ``style.clip_to``.

    ``style.dry_mask`` may be:
      - a coordinate-bearing DataArray (has ``lat``/``lon`` coords): aligned to
        the plotted grid by nearest-neighbor interpolation on coordinate VALUE,
        so it lands on the correct geographic cells regardless of the mask's
        resolution, registration offset, or latitude ordering. Cells outside the
        mask's coverage (NaN after interpolation) are treated as not-dry.
      - a bare ndarray (no coords): must match the grid shape exactly, since there
        is no coordinate information to align it by; a mismatch raises a clear
        ``ValueError`` instead of a raw positional-indexing failure.

    Shared by ``_apply_style_masks`` (tercile codes) and ``plot_field``
    (continuous fields) so both mask identically.
    """
    import importlib
    lat = np.asarray(lat); lon = np.asarray(lon)
    shape = (lat.shape[0], lon.shape[0])
    dry = None
    if style is not None and style.dry_mask is not None:
        dry = _align_bool_mask(style.dry_mask, lat, lon, name="dry_mask")
    outside = None
    if style is not None and style.clip_to is not None:
        # The per-cell containment loop below is the slow part of a styled
        # render, and multi-panel figures re-run it once per panel on the same
        # grid. Cache it for the common country-name-list clip (names + grid
        # are hashable); an arbitrary shapely geometry is not safely keyable,
        # so that path stays uncached. Do not mutate the returned array.
        cache_key = None
        if isinstance(style.clip_to, (list, tuple)):
            cache_key = (tuple(sorted(style.clip_to)), lat.tobytes(), lon.tobytes())
            if cache_key in _OUTSIDE_MASK_CACHE:
                return dry, _OUTSIDE_MASK_CACHE[cache_key]
        Point = importlib.import_module("shapely.geometry").Point
        geom = (style.clip_to if not isinstance(style.clip_to, (list, tuple))
                else _country_geometry(list(style.clip_to)))
        # Normalize lon to -180..180 for Natural Earth containment tests.
        lon180 = ((lon + 180) % 360) - 180
        inside = np.zeros(shape, dtype=bool)
        for i, la in enumerate(lat):
            for j, lo in enumerate(lon180):
                if geom.contains(Point(float(lo), float(la))):
                    inside[i, j] = True
        outside = ~inside
        if cache_key is not None:
            if len(_OUTSIDE_MASK_CACHE) > 16:
                _OUTSIDE_MASK_CACHE.clear()
            _OUTSIDE_MASK_CACHE[cache_key] = outside
    return dry, outside


def region_masks(like, style):
    """Public form of the style's spatial masks, as coordinate-bearing DataArrays.

    Returns ``(dry, outside)`` boolean DataArrays on ``like``'s grid — ``dry``
    from ``style.dry_mask`` (aligned as the plotters align it) and ``outside``
    the cells beyond ``style.clip_to``. A mask whose style field is unset
    comes back all-False (nothing masked), so the masking recipe needs no
    None-guards. Use it to mask the *data* the same way the plots mask the
    display, e.g. before ``write_terciles``::

        dry, outside = ds.region_masks(objective, style)
        ds.write_terciles(objective.where(~outside).where(~dry), path)

    ``like`` is any DataArray with lat/lon dims (aliases accepted).
    """
    from .._spatial import spatial_dims

    lat_dim, lon_dim = spatial_dims(like, context="region_masks")
    lat = np.asarray(like[lat_dim].values)
    lon = np.asarray(like[lon_dim].values)
    dry, outside = _region_masks(lat, lon, style)

    def _wrap(arr):
        if arr is None:
            arr = np.zeros((lat.shape[0], lon.shape[0]), dtype=bool)
        return xr.DataArray(arr, dims=(lat_dim, lon_dim),
                            coords={lat_dim: like[lat_dim], lon_dim: like[lon_dim]})

    return _wrap(dry), _wrap(outside)


def _apply_style_masks(code, lat, lon, style):
    """Grey dry cells (code 3n) and mask cells outside the clip geometry (-1)."""
    n = len(style.prob_bins) - 1
    dry, outside = _region_masks(lat, lon, style)
    if dry is not None:
        # Dry cells are greyed unconditionally (matches the GHACOF reference); the
        # clip below then restricts to the domain.
        code[dry] = 3 * n
    if outside is not None:
        code[outside] = -1
    return code


def _draw_smooth_terciles(ax, values, lat, lon, style, factor, is_geo):
    """Category-wise cubic-refined ``contourf`` fills for a styled tercile map.

    Each category is filled where it is the dominant (most-probable) outcome,
    banded by ``style.prob_bins`` in that category's color ramp — the smooth
    GHACOF/ACMAD outlook look. Dry cells get a crisp grey cell overlay; cells
    outside the clip stay unfilled (nodata).
    """
    import importlib
    mcolors = importlib.import_module("matplotlib.colors")
    transform_kw = {}
    if is_geo:
        ccrs = importlib.import_module("cartopy.crs")
        transform_kw = {"transform": ccrs.PlateCarree()}

    dry, outside = _region_masks(lat, lon, style)
    # NaN-safe dominant category: argmax over an all-NaN cell would pick a
    # category arbitrarily, so decide only where the full triple is finite and
    # mark the rest -1, which matches no category and is never filled.
    valid = np.isfinite(values).all(axis=0)
    dom = np.full(valid.shape, -1)
    dom[valid] = np.argmax(values[:, valid], axis=0)
    # A leading category that does not qualify as dominant (under the first
    # bin edge, or contested when the style sets secondary_max) is left
    # unfilled, never clipped up into the weakest band.
    excluded = ~valid | _no_dominant(values, valid, style.prob_bins,
                                     getattr(style, "secondary_max", None))
    if outside is not None:
        excluded = excluded | outside
    if dry is not None:
        excluded = excluded | dry

    for cat, colors in ((0, style.below_colors), (1, style.normal_colors),
                        (2, style.above_colors)):
        band = np.where((dom == cat) & ~excluded, values[cat] * 100.0, np.nan)
        refined = _refine_field(band, lat, lon, factor)
        if refined is None:
            continue
        fine, flat, flon = refined
        # The cubic refinement can ring past the bin range (below the first
        # edge next to weak cells, above the last inside strong blobs), and
        # contourf(extend="neither") would leave those points as white holes
        # inside the category's own footprint. Clip back into the bins: only
        # the drawn band assignment changes, never the probabilities.
        fine = np.clip(fine, style.prob_bins[0], style.prob_bins[-1] - 1e-9)
        ax.contourf(flon, flat, fine, levels=style.prob_bins,
                    colors=list(colors), extend="neither", **transform_kw)

    if dry is not None:
        cells = (dry & ~outside) if outside is not None else dry
        grey = np.ma.masked_invalid(np.where(cells, 1.0, np.nan))
        ax.pcolormesh(lon, lat, grey,
                      cmap=mcolors.ListedColormap([style.dry_color]),
                      vmin=0, vmax=1, shading="auto", zorder=2, **transform_kw)


def plot_tercile_forecast(pr_fcst, *, style=None, ax=None, title=None,
                          variable_kind="precip", legend=True, smooth=False):
    """Dominant-tercile probability map (IRI/PyCPT convention).

    For each grid point, identifies the tercile (below/normal/above) with
    maximum probability and colors it according to the variable convention:

    - `variable_kind="precip"` (default; IRI precipitation convention):
        * below-normal (drier) -> red
        * normal              -> grey
        * above-normal (wetter) -> blue

    - `variable_kind="temp"` (IRI temperature convention; matches the
      everyday "red = hot, blue = cold" intuition):
        * below-normal (cooler) -> blue
        * normal              -> grey
        * above-normal (warmer) -> red

    Color intensity scales with `(max_prob - 1/3)`, saturating at +0.37
    (i.e. 70% probability) so highly confident forecasts don't wash out.

    Coastlines and national borders are drawn over the map when cartopy or
    geopandas (with cached Natural Earth shapefiles) is available, and quietly
    skipped otherwise. A legend below the map shows the three categories.

    Lakes (`style.lakes`) are drawn only on the cartopy/geo path (when cartopy
    is available); the geopandas fallback does not draw lakes.

    ``smooth`` (requires ``style``): render the category fills as cubic-refined
    filled contours instead of grid cells, matching the smooth-boundary look of
    the operational GHACOF/ACMAD graphics. ``True`` refines by a factor of 4;
    pass an int to choose the factor. The probabilities are unchanged — only
    the drawn boundaries are smoothed, and never past the data's footprint.

    Input shape: (tercile=3, lat, lon), values in [0, 1] summing to 1.
    """
    if variable_kind == "precip":
        red_cat = 0
        below_label = "Below normal (drier)"
        above_label = "Above normal (wetter)"
    elif variable_kind == "temp":
        red_cat = 2
        below_label = "Below normal (cooler)"
        above_label = "Above normal (warmer)"
    else:
        raise ValueError(
            f"variable_kind must be 'precip' or 'temp', got {variable_kind!r}"
        )
    blue_cat = 2 if red_cat == 0 else 0

    factor = _smooth_factor(smooth)
    if factor and style is None:
        raise ValueError("smooth rendering requires a TercileStyle (style=...)")

    import importlib
    require_optional("matplotlib", _HINT)
    plt = importlib.import_module("matplotlib.pyplot")
    Patch = importlib.import_module("matplotlib.patches").Patch

    from .._spatial import spatial_dims

    lat_dim, lon_dim = spatial_dims(pr_fcst, context="plot_tercile_forecast")
    values = pr_fcst.transpose("tercile", lat_dim, lon_dim).values
    lon = pr_fcst[lon_dim].values
    lat = pr_fcst[lat_dim].values
    extent = (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))
    if style is not None and style.extent is not None:
        extent = tuple(style.extent)

    if ax is None:
        fig, ax, is_geo = _make_tercile_axes(plt, extent, figsize=(8, 5.5))
    else:
        fig = ax.figure
        is_geo = hasattr(ax, "coastlines")

    if style is None:
        rgb = _tercile_rgb(values, red_cat, blue_cat)
        if is_geo:
            ccrs = importlib.import_module("cartopy.crs")
            ax.imshow(rgb, extent=extent, origin="lower", transform=ccrs.PlateCarree())
            _draw_cartopy_basemap(ax)
        else:
            ax.imshow(rgb, extent=extent, origin="lower", aspect="auto", zorder=1)
            drew = _draw_geopandas_basemap(ax, extent)
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
            if drew:
                ax.grid(color="#777777", linewidth=0.3, alpha=0.5)
    elif factor:
        _draw_smooth_terciles(ax, values, lat, lon, style, factor, is_geo)
        if is_geo:
            _draw_cartopy_basemap(ax)
            if style.lakes:
                cfeature = importlib.import_module("cartopy.feature")
                ax.add_feature(cfeature.NaturalEarthFeature("physical", "lakes", "10m"),
                               facecolor=style.lake_color, edgecolor="none", zorder=3)
        else:
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    else:
        code, _ = _tercile_codes(values, style.prob_bins,
                                 getattr(style, "secondary_max", None))
        cmap, npal = _discrete_cmap(style)
        code = _apply_style_masks(code, lat, lon, style)
        code_masked = np.ma.masked_less(code, 0)
        if is_geo:
            ccrs = importlib.import_module("cartopy.crs")
            ax.pcolormesh(lon, lat, code_masked, cmap=cmap, vmin=0, vmax=npal - 1,
                          shading="auto", transform=ccrs.PlateCarree())
            _draw_cartopy_basemap(ax)
            if style.lakes and is_geo:
                cfeature = importlib.import_module("cartopy.feature")
                ax.add_feature(cfeature.NaturalEarthFeature("physical", "lakes", "10m"),
                               facecolor=style.lake_color, edgecolor="none", zorder=3)
        else:
            ax.pcolormesh(lon, lat, code_masked, cmap=cmap, vmin=0, vmax=npal - 1, shading="auto")
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])

    ax.set_title(title or "Dominant tercile probability")
    if legend and style is None:
        below_color = (1.0, 0.0, 0.0) if red_cat == 0 else (0.0, 0.0, 1.0)
        above_color = (1.0, 0.0, 0.0) if red_cat == 2 else (0.0, 0.0, 1.0)
        legend_handles = [
            Patch(facecolor=below_color, edgecolor="black", linewidth=0.5, label=below_label),
            Patch(facecolor="#999999",   edgecolor="black", linewidth=0.5, label="Normal"),
            Patch(facecolor=above_color, edgecolor="black", linewidth=0.5, label=above_label),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.10),
            ncol=3,
            frameon=True,
            framealpha=0.9,
            fontsize=8,
            title="Dominant tercile  (intensity = confidence)",
            title_fontsize=8,
        )
    elif legend and style is not None:
        _tercile_style_legend(ax, style, below_label, above_label)
    return fig


def plot_field(field, *, style=None, ax=None, cmap="RdBu_r", vmin=None, vmax=None,
               center=None, levels=None, extend="both", smooth=False,
               title=None, grey_dry=True):
    """Continuous (lat, lon) field on the same styled basemap as ``plot_terciles``.

    Draws ``field`` with ``pcolormesh`` using the identical map extent,
    coastlines / borders / gridlines, dry-cell greying, country clip, and lakes
    that ``plot_terciles`` applies for the same ``style`` -- so a difference or
    anomaly panel lines up cell-for-cell with the tercile panels beside it. The
    colors are the caller's (any Matplotlib ``cmap``); the framing and masks come
    entirely from ``style``. Region-agnostic: nothing here encodes a region.

    ``cmap`` / ``vmin`` / ``vmax`` are the usual diverging-map controls; pass
    ``center`` instead to anchor a ``TwoSlopeNorm`` at a value (e.g. 0). Set
    ``grey_dry=False`` to leave dry cells transparent rather than greyed.

    ``levels`` (a sequence of bin edges) switches to a discrete classified
    scale -- a ``BoundaryNorm`` over ``cmap``, with ``extend`` controlling the
    colorbar's out-of-range arrows -- the convention operational anomaly and
    onset maps use. Mutually exclusive with ``vmin``/``vmax``/``center``.
    ``smooth`` (requires ``levels``) renders cubic-refined filled contours
    instead of grid cells; ``True`` refines by a factor of 4, an int picks the
    factor. Smoothing changes only the drawn boundaries, never the data, and
    stays inside the field's finite footprint.

    Returns the Matplotlib mappable, for ``fig.colorbar`` — or ``None`` when
    ``smooth`` finds nothing finite to contour (the panel then says "no data").
    """
    import importlib
    require_optional("matplotlib", _HINT)
    plt = importlib.import_module("matplotlib.pyplot")
    from .._spatial import spatial_dims

    factor = _smooth_factor(smooth)
    if levels is not None and center is not None:
        raise ValueError("pass either levels or center, not both")
    if levels is not None and (vmin is not None or vmax is not None):
        raise ValueError("levels defines the scale; do not also pass vmin/vmax")
    if factor and levels is None:
        raise ValueError("smooth field rendering requires levels=...")

    lat_dim, lon_dim = spatial_dims(field, context="plot_field")
    fld = field.transpose(lat_dim, lon_dim)
    lat = np.asarray(fld[lat_dim].values)
    lon = np.asarray(fld[lon_dim].values)
    values = np.array(fld.values, dtype=float)
    extent = (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))
    if style is not None and style.extent is not None:
        extent = tuple(style.extent)

    if ax is None:
        fig, ax, is_geo = _make_tercile_axes(plt, extent, figsize=(8, 5.5))
    else:
        fig = ax.figure
        is_geo = hasattr(ax, "coastlines")
        if is_geo:
            ccrs = importlib.import_module("cartopy.crs")
            ax.set_extent(extent, crs=ccrs.PlateCarree())

    dry, outside = _region_masks(lat, lon, style) if style is not None else (None, None)
    if outside is not None:
        values[outside] = np.nan            # outside the clip -> nodata (transparent)
    if grey_dry and dry is not None:
        values[dry] = np.nan                # dry -> drawn as grey by the overlay below
    masked = np.ma.masked_invalid(values)

    norm = None
    if center is not None:
        TwoSlopeNorm = importlib.import_module("matplotlib.colors").TwoSlopeNorm
        lo = vmin if vmin is not None else float(np.nanmin(values))
        hi = vmax if vmax is not None else float(np.nanmax(values))
        norm = TwoSlopeNorm(vmin=lo, vcenter=center, vmax=hi)
    Colormap = importlib.import_module("matplotlib.colors").Colormap
    cmap_obj = cmap.copy() if isinstance(cmap, Colormap) else plt.get_cmap(cmap).copy()
    if style is not None:
        cmap_obj.set_bad(style.nodata_color)   # clipped/outside cells render as nodata
    if levels is not None and not factor:
        BoundaryNorm = importlib.import_module("matplotlib.colors").BoundaryNorm
        norm = BoundaryNorm(list(levels), cmap_obj.N, extend=extend)
    kw = dict(cmap=cmap_obj, shading="auto")
    if norm is not None:
        kw["norm"] = norm
    else:
        kw["vmin"] = vmin; kw["vmax"] = vmax

    def _draw_field(transform_kw):
        if factor:
            refined = _refine_field(values, lat, lon, factor)
            if refined is None:      # nothing to contour -- say so, don't crash
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=11, color="#888888")
                return None
            if style is not None:
                # contourf simply skips NaN regions; paint the clipped/invalid
                # cells in nodata_color first so the smooth path renders them
                # the same way the pcolormesh path's set_bad does.
                ListedColormap = importlib.import_module(
                    "matplotlib.colors").ListedColormap
                nodata = np.where(~np.isfinite(values), 1.0, np.nan)
                ax.pcolormesh(lon, lat, np.ma.masked_invalid(nodata),
                              cmap=ListedColormap([style.nodata_color]),
                              vmin=0, vmax=1, shading="auto", **transform_kw)
            fine, flat, flon = refined
            return ax.contourf(flon, flat, fine, levels=list(levels),
                               cmap=cmap_obj, extend=extend, **transform_kw)
        return ax.pcolormesh(lon, lat, masked, **transform_kw, **kw)

    def _grey_overlay():
        if not (grey_dry and dry is not None):
            return
        ListedColormap = importlib.import_module("matplotlib.colors").ListedColormap
        cells = (dry & ~outside) if outside is not None else dry
        grey = np.where(cells, 1.0, np.nan)
        gkw = dict(cmap=ListedColormap([style.dry_color]), vmin=0, vmax=1,
                   shading="auto", zorder=2)
        if is_geo:
            ccrs = importlib.import_module("cartopy.crs")
            ax.pcolormesh(lon, lat, np.ma.masked_invalid(grey),
                          transform=ccrs.PlateCarree(), **gkw)
        else:
            ax.pcolormesh(lon, lat, np.ma.masked_invalid(grey), **gkw)

    if is_geo:
        ccrs = importlib.import_module("cartopy.crs")
        im = _draw_field({"transform": ccrs.PlateCarree()})
        _grey_overlay()
        _draw_cartopy_basemap(ax)
        if style is not None and style.lakes:
            cfeature = importlib.import_module("cartopy.feature")
            ax.add_feature(cfeature.NaturalEarthFeature("physical", "lakes", "10m"),
                           facecolor=style.lake_color, edgecolor="none", zorder=3)
    else:
        im = _draw_field({})
        _grey_overlay()
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])

    ax.set_title(title or "")
    return im


def plot_tercile_comparison(forecast, reference, *, style=None, axes=None,
                            labels=("forecast", "reference", "difference"),
                            diff_cmap="BrBG", diff_limit=40.0, title=None):
    """Three-panel comparison of two tercile forecasts and their difference.

    Draws ``forecast`` and ``reference`` with ``plot_terciles`` and, in the third
    panel, their signed tercile-*tilt* difference via ``plot_field`` -- so all
    three panels share the same extent, masks, and basemap. The tilt is
    ``P(above) - P(below)``; the difference is ``forecast - reference`` in
    percentage points, with ``reference`` regridded onto the forecast grid first.
    With a diverging ``diff_cmap`` (e.g. brown-to-green), positive means the
    forecast leans wetter than the reference.

    Region-agnostic: the roster of forecasts and any file loading stay with the
    caller; this only renders one comparison row.

    Parameters
    ----------
    forecast, reference : xr.DataArray
        Tercile probabilities, dims ``(tercile, lat, lon)`` (tercile ordered
        below, normal, above). ``reference`` is regridded onto the forecast grid.
    style : TercileStyle or None
        Applied to all three panels (palette on the forecasts; extent/masks on all).
    axes : sequence of 3 axes or None
        Three (cartopy) axes to draw into; if None, a 1x3 figure is created.
    labels : (str, str, str)
        Panel titles for forecast, reference, and difference.
    diff_cmap : str or Colormap
        Colormap for the difference panel.
    diff_limit : float
        Symmetric color limit for the difference (``vmin=-diff_limit``, ``vmax=+``).
    title : str or None
        Optional suptitle, used only when this function creates the figure.

    Returns
    -------
    (axes, diff_mappable)
        The three axes and the difference panel's mappable, e.g. for a shared
        ``fig.colorbar``.
    """
    import importlib
    require_optional("matplotlib", _HINT)
    plt = importlib.import_module("matplotlib.pyplot")
    from .._spatial import spatial_dims

    if axes is None:
        try:
            ccrs = importlib.import_module("cartopy.crs")
            fig, axes = plt.subplots(1, 3, figsize=(13, 4.2),
                                     subplot_kw={"projection": ccrs.PlateCarree()})
        except Exception:
            fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    else:
        fig = axes[0].figure

    lab_f, lab_r, lab_d = labels
    plot_tercile_forecast(forecast, style=style, ax=axes[0], legend=False, title=lab_f)
    plot_tercile_forecast(reference, style=style, ax=axes[1], legend=False, title=lab_r)

    # Signed tercile-tilt difference, forecast minus reference, in percentage points.
    lat_dim, lon_dim = spatial_dims(forecast, context="plot_tercile_comparison")
    r_lat, r_lon = spatial_dims(reference, context="plot_tercile_comparison")
    ref_al = (reference.rename({r_lat: lat_dim, r_lon: lon_dim})
              if (r_lat, r_lon) != (lat_dim, lon_dim) else reference)
    ref_on_fc = ref_al.interp({lat_dim: forecast[lat_dim], lon_dim: forecast[lon_dim]})

    def _tilt(p):   # P(above) - P(below); positional so string/int tercile coords both work
        return p.isel(tercile=2) - p.isel(tercile=0)

    signed = (_tilt(forecast) - _tilt(ref_on_fc)) * 100.0
    diff_im = plot_field(signed, style=style, ax=axes[2], cmap=diff_cmap,
                         vmin=-diff_limit, vmax=diff_limit, title=lab_d)

    if title:
        fig.suptitle(title)
    return axes, diff_im


def plot_deterministic_forecast(det_fcst, *, ax=None, title=None,
                                cmap="RdBu_r", center=None):
    """Single-panel pcolormesh of a deterministic field. Input: (lat, lon)."""
    plt, fig, ax = _new_fig(ax)
    if center is not None:
        import importlib
        TwoSlopeNorm = importlib.import_module("matplotlib.colors").TwoSlopeNorm
        v = float(np.abs(det_fcst.values - center).max())
        norm = TwoSlopeNorm(vmin=center - v, vcenter=center, vmax=center + v)
        im = ax.pcolormesh(det_fcst.lon, det_fcst.lat, det_fcst.values,
                           cmap=cmap, norm=norm)
    else:
        im = ax.pcolormesh(det_fcst.lon, det_fcst.lat, det_fcst.values, cmap=cmap)
    ax.set_xlabel("Lon")
    ax.set_ylabel("Lat")
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)
    return fig


def plot_exceedance_probability(exceedance_prob, threshold, *, ax=None):
    """Map of P(forecast > threshold). Input: (lat, lon), values in [0, 1]."""
    plt, fig, ax = _new_fig(ax)
    im = ax.pcolormesh(
        exceedance_prob.lon, exceedance_prob.lat, exceedance_prob.values,
        cmap="viridis", vmin=0, vmax=1,
    )
    ax.set_xlabel("Lon")
    ax.set_ylabel("Lat")
    ax.set_title(f"P(forecast > {threshold})")
    plt.colorbar(im, ax=ax, fraction=0.046, label="probability")
    return fig


def plot_flex_pdf(fcst_mu, fcst_scale, climo_mu, climo_scale, *,
                  location, ax=None):
    """Forecast vs climatology Gaussian PDFs at a single point.

    `*_mu` / `*_scale` may be scalars or DataArrays. If DataArrays, the
    nearest grid point to `location=(lon, lat)` is selected.
    """
    plt, fig, ax = _new_fig(ax, figsize=(7, 4))

    def _scalar(v):
        if hasattr(v, "sel"):
            lon, lat = location
            return float(v.sel(lon=lon, lat=lat, method="nearest"))
        return float(v)

    f_mu, f_sc = _scalar(fcst_mu), _scalar(fcst_scale)
    c_mu, c_sc = _scalar(climo_mu), _scalar(climo_scale)

    lo = min(f_mu - 4 * f_sc, c_mu - 4 * c_sc)
    hi = max(f_mu + 4 * f_sc, c_mu + 4 * c_sc)
    x = np.linspace(lo, hi, 400)

    def _gauss(x, mu, sc):
        return np.exp(-0.5 * ((x - mu) / sc) ** 2) / (sc * np.sqrt(2 * np.pi))

    ax.plot(x, _gauss(x, c_mu, c_sc), color="grey", linewidth=2, label="Climatology")
    ax.plot(x, _gauss(x, f_mu, f_sc), color="tab:blue", linewidth=2, label="Forecast")
    ax.fill_between(x, _gauss(x, f_mu, f_sc), color="tab:blue", alpha=0.15)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title(f"PDF at lon={location[0]}, lat={location[1]}")
    ax.legend()
    return fig


def render_styled_terciles(ax, probs, style, *, title=None, small=False, smooth=False):
    """Draw a binned dominant-tercile map with ``style`` onto an existing ``ax``.

    Thin wrapper over :func:`plot_tercile_forecast` for multi-panel figures: it renders the styled
    (binned dominant-category palette) tercile map onto the supplied ``ax``. ``small=True`` drops
    the category legend and the axis ticks, for compact grids. ``smooth`` is passed through to
    :func:`plot_tercile_forecast`. ``probs`` is a
    ``(tercile, lat, lon)`` fractional-probability DataArray. Returns ``ax``.
    """
    plot_tercile_forecast(probs, style=style, ax=ax, title=title, legend=not small,
                          smooth=smooth)
    if small:
        ax.set_xticks([])
        ax.set_yticks([])
    return ax
