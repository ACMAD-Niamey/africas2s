# Plotting and reporting

Plotting/reporting live in `africas2s.plotting` and `africas2s.reporting`. Importing the subpackage does **not** load matplotlib/cartopy; each function gates on its optional deps and raises a clear `ImportError` with a `pip install africas2s[plotting]` hint. Basemaps use cartopy when available, else geopandas with cached Natural Earth shapefiles (`~/.local/share/cartopy/shapefiles/natural_earth/...`), else plain axes.

## Which plot for what you have

| You have | Call |
|---|---|
| Tercile forecast `(tercile, lat, lon)` | `ds.plot_terciles(fc)` — IRI-style dominant-tercile map |
| Two tercile forecasts to compare | `ds.plot_tercile_comparison(fc, reference)` |
| Deterministic field `(lat, lon)` (ensemble mean, anomaly, ...) | `ds.plot_field(...)` / `plot_deterministic_forecast(...)` |
| Any gridded field `(lat, lon)` — percentile / rank / SPI map, with country outlines or class bins | `ds.plot_field_map(da, classes=..., highlight=..., boundaries=...)` |
| Per-region values `(region,)` (from `rosetta.zonal`) + admin geometries | `ds.plot_choropleth(values, geometries, by=...)` |
| A `CompletionResult` (analog scenario completion) | `ds.plot_accumulation_scenarios(result)` |
| Two index series against each other (SST vs rainfall, coloured by outcome) | `ds.plot_index_scatter(x, y, color_by=...)` |
| Styled tercile map onto your own subplot `ax` | `ds.render_styled_terciles(ax, probs, style)` |
| Several forecasts as one grid (per-model, per-method, tercile and/or continuous) | `ds.plot_matrix(panels, style=...)` |
| Component forecasts plus the combined objective | `ds.plot_components_objective(components, objective, style=...)` |
| A regional centre's colour language from a JSON style file | `ds.TercileStyle.from_json(path, dry_mask=..., ...)` |
| The style's dry/clip masks as data masks (e.g. before `write_terciles`) | `ds.region_masks(field, style)` |
| `SkillReport` with `spatial=True` | `plot_skill_maps(report, ["rpss", ...])`; full PDF: `report.to_pdf(...)` |
| CV tercile hindcasts + obs | `plot_reliability_diagram(cv_terc, obs)` |
| `ComparisonReport` from `skill_compare` | `.to_table()` / `.to_heatmap(path)` / `.to_pdf(path)` |
| `FlexForecastResult` | `plot_exceedance_probability(...)`; point PDF vs climo: `plot_flex_pdf(...)` |
| Fitted CCA method | `plot_eof_modes(...)` / `plot_cca_modes(...)` |
| Predictor/predictand extents (pre-run sanity check) | `plot_domains(...)` |
| Raw fetched GCM/obs data (pre-downscaling quick looks) | not africas2s's job — see the rosetta skill's `references/plotting.md` (plain xarray/cartopy recipes) |

## Top-level re-exports

```python
ds.plot_terciles(...)            # = plotting.forecasts.plot_tercile_forecast
ds.plot_field(...)
ds.plot_tercile_comparison(...)
ds.render_styled_terciles(...)   # = plotting.forecasts.render_styled_terciles
ds.plot_field_map(...)           # = plotting.maps.plot_field_map
ds.plot_choropleth(...)          # = plotting.maps.plot_choropleth
ds.natural_earth_borders(...)    # = plotting.maps.natural_earth_borders
ds.plot_accumulation_scenarios(...)  # = plotting.scenarios.plot_accumulation_scenarios
ds.plot_index_scatter(...)           # = plotting.scenarios.plot_index_scatter
ds.plot_matrix(...)                  # = plotting.panels.plot_matrix
ds.plot_components_objective(...)    # = plotting.panels.plot_components_objective
ds.tercile_legend_handles(...)       # = plotting.panels.tercile_legend_handles
ds.region_masks(...)                 # = plotting.forecasts.region_masks
ds.TercileStyle                      # = plotting.style.TercileStyle (.from_json)
ds.tercile_diverging_cmap(...)       # = plotting.style.tercile_diverging_cmap
```

## Forecast maps

```python
plot_tercile_forecast(pr_fcst, *, style=None, ax=None, title=None,
                      variable_kind="precip", legend=True, smooth=False)
```
IRI-style dominant-tercile map from a `(tercile, lat, lon)` array. `variable_kind` ∈ `{"precip", "temp"}` selects the palette; color intensity scales with `(max_prob − 1/3)`, saturating at 0.37. `smooth` (needs `style`) renders cubic-refined filled contours instead of grid cells, the GHACOF/ACMAD outlook look; `True` = refinement factor 4, or pass an int factor. Smoothing only reshapes the drawn boundaries, never invents data outside the field's footprint.

```python
plot_field(field, *, style=None, ax=None, cmap="RdBu_r", vmin=None, vmax=None,
           center=None, levels=None, extend="both", smooth=False,
           title=None, grey_dry=True) -> mappable | None
    # levels = discrete bin edges (BoundaryNorm) — the classified-scale
    # convention of operational anomaly/onset maps; exclusive with
    # vmin/vmax/center. smooth (needs levels) contours the refined field;
    # returns None (and writes "no data" on the axes) if nothing is finite.
plot_tercile_comparison(forecast, reference, *, style=None, axes=None,
                        labels=("forecast", "reference", "difference"),
                        diff_cmap="BrBG", diff_limit=40.0, title=None) -> (axes, diff_mappable)
    # "tilt" difference = P(above) − P(below), percentage points
plot_deterministic_forecast(det_fcst, *, ax=None, title=None, cmap="RdBu_r", center=None)
plot_exceedance_probability(exceedance_prob, threshold, *, ax=None)
plot_flex_pdf(fcst_mu, fcst_scale, climo_mu, climo_scale, *, location, ax=None)  # location=(lon, lat)
render_styled_terciles(ax, probs, style, *, title=None, small=False, smooth=False)  # -> ax
    # thin wrapper over plot_tercile_forecast(style=) for multi-panel grids;
    # small=True drops the legend and axis ticks. probs is (tercile, lat, lon).
region_masks(like, style) -> (dry, outside)
    # the style's dry_mask and clip_to as bool DataArrays on `like`'s grid
    # (all-False where the style field is unset) — mask the DATA the way the
    # plots mask the display, e.g. objective.where(~outside).where(~dry)
    # before write_terciles.
```

## Panel grids, composites, and style files

One call per figure for the grids every workflow needs; all live in
`plotting.panels` / `plotting.style` and are re-exported at top level.

```python
plot_matrix(panels, *, style=None, ncols=3, skill_mask=None, smooth=False,
            variable_kind="precip", legend=True, legend_detailed=False,
            cmap="RdBu_r", levels=None, vmin=None, vmax=None, center=None,
            extend="both", cbar_label=None, suptitle=None,
            panel_size=(4.2, 3.9), figsize=None) -> Figure
```
Grid of maps. `panels` is `{title: DataArray}` or an iterable of
`(title, data)` / `(title, data, style)`; a `(tercile, lat, lon)` entry renders
as a styled tercile map, a `(lat, lon)` entry as a continuous field, and the
two mix freely (e.g. raw-anomaly and post-CCA panels side by side). Tercile
panels share one figure-level legend (`legend_detailed=True` shows every
probability band); continuous panels share one scale and one colorbar
(auto-computed vmin/vmax over all continuous panels unless given, symmetric
about `center` when set). A per-panel style overrides the figure style — e.g.
`(title, objective, replace(style, dry_mask=None))` for a dry-mask-off panel.
`skill_mask` (bool, True = insufficient skill) blanks cells on every panel at
display time; coordinate-bearing masks are aligned per panel, bare arrays must
match. For mask-before-combine semantics use `ds.mask_by_skill` on the data.
Panel titles draw as in-map chips; per-panel tick/gridline labels are stripped.
Trailing grid cells are hidden. Works without cartopy (plain-axes fallback).

```python
plot_components_objective(components, objective, *, objective_label="Objective",
                          ncols=2, **plot_matrix_kwargs) -> Figure
```
The components-plus-final composite: every component forecast in a grid with
the combined objective as the last panel, bold label and heavier frame.

```python
tercile_legend_handles(style=None, *, variable_kind="precip", detailed=False,
                       include_dry=None, dry_label="Dry-masked / no data") -> [Patch]
```
The palette-derived legend handles, for custom figures. No style: 3 saturated
patches per `variable_kind`. With a style: one patch per category, or per
probability band (`detailed=True`); `include_dry` defaults to whether the
style carries a `dry_mask`.

```python
TercileStyle.from_json(path, **overrides) -> TercileStyle
tercile_diverging_cmap(style, *, name="tercile_diverging", n=256) -> Colormap
    # strongest-below → white → strongest-above ramp for anomaly/tilt/skill panels
```
Institutional colour languages are NOT hardcoded in the package: each workflow
owns its palette as a JSON file of `TercileStyle` fields and loads it with
`from_json`; keyword overrides win and carry the non-JSON fields
(`TercileStyle.from_json("ghacof.json", dry_mask=too_dry, clip_to=IGAD,
extent=(21, 52, -13, 23))`). Keys starting with `_` are comments; unknown keys
raise. Reference copies of the two operational languages ship in
`examples/styles/`: `ghacof.json` (ICPAC/GHACOF: yellow/cyan/green, 5 bands,
lakes on) and `acmad.json` (ACMAD: orange/grey/green, 6 bands with the
33.33–36 extra step).

## General field maps, choropleths, and monitoring plots

Not forecast-specific: render any gridded field, any per-region value, or the analog-completion
outputs. `plot_field_map` and `plot_choropleth` return a matplotlib `Figure`.

```python
plot_field_map(da, *, ax=None, cmap=None, vmin=None, vmax=None, classes=None, clip=None,
               highlight=None, highlight_label="driest on record", boundaries=None,
               title=None, cbar_label=None, figsize=(8, 7))
```
Render a 2-D `(lat, lon)` field (reduce extra dims first). `classes=(bounds, colors[, labels])` is a
generic **discrete-classification** mechanism — `bounds` is N+1 breakpoints, `colors` the N fills,
optional `labels` the class names; draws a stepped colour bar (overrides `cmap`/`vmin`/`vmax`) and
matches how operational rank/percentile maps are shown. `clip` (shapefile / GeoDataFrame / geometry)
masks cells outside a region to NaN (multi-feature inputs dissolved to their union). `highlight`
overpaints cells equal to a value in one saturated colour — the "driest on record" convention, e.g.
`highlight=1` over a `ds.rank_of_record(...)` field. `boundaries` (a GeoDataFrame/GeoSeries, e.g. from
`natural_earth_borders`) overlays admin outlines. Without `classes`, a field that looks like a
fraction (`[0,1]`) defaults to a sequential percentile colormap on `[0,1]`.

```python
plot_choropleth(values, geometries, *, by=None, ax=None, cmap=None, vmin=None, vmax=None,
                classes=None, missing_color="#e8e8e8", edgecolor="#ffffff", linewidth=0.2,
                title=None, cbar_label=None, figsize=(8, 8))
```
Fill admin polygons by a per-`region` value. `values` is indexed by a `region` dim (e.g.
`rosetta.zonal` output); `geometries` is a GeoDataFrame (one row per region); `by` is the geometry
column matching `values`'s `region` labels (defaults to the GeoDataFrame index). `classes` works as
in `plot_field_map`. `missing_color` fills regions with no value (drawn, not dropped, so no holes).

```python
natural_earth_borders(region=None, *, scale="50m") -> GeoDataFrame
```
Load Natural Earth admin-0 country polygons (EPSG:4326) for use as `plot_field_map(boundaries=...)`.
Reads cartopy's cached shapefile **without importing cartopy** (works on a plain-matplotlib install);
`region=[lat_s, lat_n, lon_w, lon_e]` keeps only intersecting countries, `scale` ∈ `"50m"`/`"10m"`.
Raises `FileNotFoundError` if the cached shapefile is absent (install cartopy once to populate it).

```python
plot_accumulation_scenarios(result, *, ax=None, reduce=None, show_consensus=True,
                            color_by_scenario=False, climatology=None, title=None,
                            ylabel="accumulation", figsize=(9, 5.5))
```
Plot the accumulation curves of a `CompletionResult` (observed → forecast → each analog fanning out).
Its `scenarios` must reduce to a single series — select a pixel/region first
(`result.scenarios.sel(region=...)`) or pass `reduce=lambda da: da.mean(["lat", "lon"])`.
`show_consensus` draws the analog-median accumulation; `color_by_scenario` gives one colour per
analog year; pass `climatology` (the result's own `(year, step, …)` archive) to draw the
climatological-median reference. Returns a `Figure`.

```python
plot_index_scatter(x, y, *, ax=None, color_by=None, categories=None, colors=None,
                   highlight=None, highlight_color=None, forecast=None, forecast_marker="*",
                   forecast_color="#f2c14e", forecast_label="Forecast", error_bars=None,
                   trendline=False, trendline_annotate=True, labels=False,
                   xlabel=None, ylabel=None, title=None, figsize=(7, 6))
```
Scatter two index series (`x`, `y` over `year`, aligned on their intersection). `color_by` is a
categorical series over `year` (e.g. observed rainfall tercile) — colours the points, turning a
scatter of ocean states into a statement about rainfall outcomes. `highlight` (a list of years, e.g.
the analogs) draws heavier annotated markers (`highlight_color` fills them, else an open ring).
`forecast=(x, y)` draws the forecast year as a star; `error_bars=(x_bounds, y_bounds)` — each an
`ErrorBounds` or `(lower, upper)` pair — brackets it. `trendline` adds an OLS fit (with R² if
`trendline_annotate`). Returns a `Figure`.

## Diagnostics

```python
plot_domains(predictor_extent, predictand_extent, *, ax=None, title=None)  # extents (lat_s, lat_n, lon_w, lon_e); cartopy
plot_skill_maps(skill_report, metric_names, *, ncols=3)   # grid from SkillReport.spatial; fixed cmap/range per metric; cartopy
plot_reliability_diagram(forecast, obs, *, n_bins=5, ax=None, title=None)  # BN/NN/AN curves
plot_eof_modes(cca_fit, kind="predictor", n_modes=3, *, ncols=3)
plot_cca_modes(cca_fit, n_modes=3)
```

## `TercileStyle`

```python
@dataclass
class TercileStyle:
    below_colors: list[str]; normal_colors: list[str]; above_colors: list[str]
    prob_bins: list[float]        # percent edges; len == n_colors + 1
    dry_mask = None; dry_color = "#bebebe"
    clip_to = None                # list of country NAMEs or a shapely geometry
    lakes = False; lake_color = "#78b8f8"
    nodata_color = "#ffffff"
    extent = None                 # (lon_w, lon_e, lat_s, lat_n)
```

Pass as `style=` to the tercile plotting functions to control palette, probability binning, dry masking, country clipping, lakes, and extent. `TercileStyle.from_json(path, **overrides)` builds one from a JSON style file a workflow owns (see "Panel grids, composites, and style files").

## Reports

```python
report = ds.skill(cv_terc, obs, metrics="svslrf", spatial=True)
report.to_pdf("verification.pdf", style="svslrf")
```

`SkillReport.to_pdf` renders a WMO-SVSLRF verification PDF via `reporting.svslrf.render(report, path)`: cover + mandatory triplet (RPSS, ROC BN/NN/AN, reliability), ROC + reliability diagram page, spatial-map grid, member-contributions page, and a secondary-metrics table.

```python
cmp = ds.skill_compare({"bcsd": fc1, "cca": fc2}, obs, metrics=["rpss"], spatial=True)
cmp.to_table()                    # DataFrame: methods x metrics
cmp.to_heatmap("heatmap.png")     # RdBu, vmin=-1, vmax=1
cmp.to_pdf("comparison.pdf", spatial_maps=True)
```

## Export

```python
ds.write_terciles(tercile_fc, "forecast.nc", title="MAM precip", method="cca")
    # NetCDF with below/normal/above percent (0-100) variables, float32, _FillValue=-9999.0
report.to_geotiff("rpss.tif", "rpss")   # one spatial metric as EPSG:4326 GeoTIFF (rioxarray)
ds.tercile_mae(candidate_probs, "reference.nc")  # MAE in percentage points vs a reference
```

## Figures, saving, and headless use

- No plot function calls `plt.show()` or writes a file — only the report methods (`to_pdf`, `to_heatmap`, `to_geotiff`) take a path. Save maps yourself: `plt.savefig("map.png", dpi=200, bbox_inches="tight")`.
- Single-panel functions accept `ax=` so you can compose them into your own subplot grids; grid-producing functions (`plot_skill_maps`, `plot_eof_modes`, `plot_cca_modes`, `plot_tercile_comparison`, `plot_matrix`, `plot_components_objective`) build their own figure and return it (or its axes) — save via the returned object.
- Return values differ by function: `plot_tercile_forecast`/`plot_deterministic_forecast`/`plot_exceedance_probability`/`plot_matrix`/`plot_components_objective` return the figure; `plot_field` returns the mappable (attach your own colorbar), or `None` when a smooth render finds nothing finite; `plot_tercile_comparison` returns `(axes, diff_mappable)`.
- Headless/CI: set `MPLBACKEND=Agg` (or `matplotlib.use("Agg")` before importing pyplot).
- Cartopy fallback applies to the tercile/forecast maps (`forecasts.py`) and the panel grids (`panels.py`): without cartopy they fall back to geopandas Natural Earth outlines, then plain axes. `plot_skill_maps`, `plot_domains`, `plot_eof_modes`, and `plot_cca_modes` hard-require cartopy and raise `ImportError` without it.
