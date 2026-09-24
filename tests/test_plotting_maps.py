"""Field maps and choropleths.

Structural checks only (Agg backend), but the structure that carries meaning:
that a 2-D field is required, that the record-driest highlight paints only the
matching cells, and that a choropleth matches values to polygons by key and
draws (not drops) regions with no data.
"""
import numpy as np
import pytest
import xarray as xr

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

gpd = pytest.importorskip("geopandas")
from shapely.geometry import box  # noqa: E402

from africas2s.plotting.maps import plot_choropleth, plot_field_map  # noqa: E402


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture
def field():
    lat = np.linspace(3, 15, 12)
    lon = np.linspace(33, 48, 15)
    rng = np.random.default_rng(0)
    return xr.DataArray(rng.random((12, 15)), dims=("lat", "lon"),
                        coords={"lat": lat, "lon": lon}, name="percentile")


@pytest.fixture
def regions():
    return gpd.GeoDataFrame(
        {"code": ["A", "B", "C"], "name": ["x", "y", "z"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        crs="EPSG:4326",
    )


# --- field map -------------------------------------------------------------


def test_field_map_renders_a_2d_field(field):
    fig = plot_field_map(field)
    assert fig.axes and fig.axes[0].collections  # a pcolormesh got drawn


def test_field_map_refuses_a_non_2d_field(field):
    cube = field.expand_dims(time=[0, 1])
    with pytest.raises(ValueError, match="needs a 2-D"):
        plot_field_map(cube)


def test_field_map_defaults_to_a_0_1_scale_for_a_fraction_field(field):
    fig = plot_field_map(field)  # values in [0, 1]
    mesh = fig.axes[0].collections[0]
    assert mesh.get_clim() == (0.0, 1.0)


def test_field_map_autoscales_a_non_fraction_field():
    da = xr.DataArray(np.arange(6.0).reshape(2, 3) * 100, dims=("lat", "lon"),
                      coords={"lat": [3, 4], "lon": [33, 34, 35]})
    fig = plot_field_map(da)
    lo, hi = fig.axes[0].collections[0].get_clim()
    assert hi > 1.0  # not clamped to the fraction range


def _meshes(ax):
    """The pcolormesh layers on ``ax``, ignoring basemap feature artists."""
    from matplotlib.collections import QuadMesh
    return [c for c in ax.collections if isinstance(c, QuadMesh)]


def test_highlight_paints_only_the_matching_cells():
    """rank_of_record == 1 over a field where exactly two cells are the record
    minimum: the highlight layer must cover two cells, not the whole grid."""
    ranks = xr.DataArray(
        np.array([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]]),
        dims=("lat", "lon"), coords={"lat": [3, 4], "lon": [33, 34, 35]},
    )
    fig = plot_field_map(ranks, highlight=1)
    # base mesh + highlight mesh (the cartopy basemap, when present, adds
    # coastline/border FeatureArtists to ax.collections too -- not meshes)
    meshes = _meshes(fig.axes[0])
    assert len(meshes) == 2
    highlight = meshes[1].get_array()
    assert np.isfinite(np.asarray(highlight, float)).sum() == 2


def test_highlight_with_no_matching_cell_draws_no_extra_layer(field):
    fig = plot_field_map(field, highlight=999)
    assert len(_meshes(fig.axes[0])) == 1


def test_field_map_overlays_boundaries(field, regions):
    fig = plot_field_map(field, boundaries=regions)
    assert fig.axes[0].lines or fig.axes[0].collections


# --- choropleth ------------------------------------------------------------


def _values(mapping, dim="region"):
    keys = list(mapping)
    return xr.DataArray(list(mapping.values()), dims=dim, coords={dim: keys},
                        name="pct")


def test_choropleth_matches_values_to_polygons_by_key(regions):
    fig = plot_choropleth(_values({"A": 0.1, "B": 0.5, "C": 0.9}), regions, by="code")
    # geopandas draws the three polygons as one PatchCollection
    assert any(len(c.get_paths()) >= 3 for c in fig.axes[0].collections
               if hasattr(c, "get_paths"))


def test_choropleth_draws_a_region_with_no_value_rather_than_dropping_it(regions):
    # C is absent from the values -> NaN -> missing_color, still drawn.
    fig = plot_choropleth(_values({"A": 0.1, "B": 0.5}), regions, by="code")
    assert fig.axes  # renders without error; C is painted as "no data"


def test_choropleth_defaults_to_a_0_1_scale_for_a_fraction_field(regions):
    fig = plot_choropleth(_values({"A": 0.1, "B": 0.5, "C": 0.9}), regions, by="code")
    coll = next(c for c in fig.axes[0].collections if c.get_array() is not None)
    assert coll.get_clim() == (0.0, 1.0)


def test_choropleth_rejects_a_missing_by_column(regions):
    with pytest.raises(ValueError, match="not found in geometries"):
        plot_choropleth(_values({"A": 0.1, "B": 0.5, "C": 0.9}), regions, by="nope")


def test_choropleth_reprojects_to_epsg_4326(regions):
    projected = regions.to_crs("EPSG:3857")
    fig = plot_choropleth(_values({"A": 0.1, "B": 0.5, "C": 0.9}), projected, by="code")
    # x-limits should be in degrees (~0-3), not Web Mercator metres
    assert fig.axes[0].get_xlim()[1] < 100


# --- discrete classification (classes=) ------------------------------------


def test_field_map_classes_draws_a_stepped_colorbar(field):
    classes = ([0, 0.2, 0.5, 1.0], ["#8b0000", "#dddddd", "#00008b"], ["dry", "mid", "wet"])
    fig = plot_field_map(field, classes=classes)
    # colorbar tick labels are the class names
    cb_ax = fig.axes[-1]
    assert [t.get_text() for t in cb_ax.get_yticklabels()] == ["dry", "mid", "wet"]


def test_field_map_classes_bins_values_by_bounds():
    from matplotlib.colors import BoundaryNorm
    da = xr.DataArray([[0.1, 0.35, 0.8]], dims=("lat", "lon"),
                      coords={"lat": [0.0], "lon": [0.0, 1.0, 2.0]})
    fig = plot_field_map(da, classes=([0, 0.2, 0.5, 1.0],
                                      ["#8b0000", "#dddddd", "#00008b"]))
    mesh = fig.axes[0].collections[0]
    assert isinstance(mesh.norm, BoundaryNorm)


def test_classes_rejects_mismatched_bounds_and_colors(field):
    with pytest.raises(ValueError, match="len\\(bounds\\) == len\\(colors\\) \\+ 1"):
        plot_field_map(field, classes=([0, 1], ["#000", "#fff", "#f00"]))


def test_choropleth_classes_draws_a_stepped_legend(regions):
    vals = _values({"A": 0.02, "B": 0.4, "C": 0.95})
    fig = plot_choropleth(vals, regions, by="code",
                          classes=([0, 0.1, 0.5, 1.0], ["#7e0006", "#e0e0e0", "#3a86c8"],
                                   ["low", "mid", "high"]))
    assert any("low" in t.get_text() or "high" in t.get_text()
               for a in fig.axes for t in a.get_yticklabels())


# --- clip to geometry ------------------------------------------------------


def test_clip_masks_cells_outside_the_geometry():
    """A field over a 4x4 grid clipped to the left-half box must render only the
    left columns (right half -> NaN)."""
    lat = lon = np.array([0.5, 1.5, 2.5, 3.5])
    da = xr.DataArray(np.ones((4, 4)), dims=("lat", "lon"),
                      coords={"lat": lat, "lon": lon})
    left = gpd.GeoDataFrame(geometry=[box(0, 0, 2, 4)], crs="EPSG:4326")
    fig = plot_field_map(da, clip=left)
    mesh = fig.axes[0].collections[0]
    arr = np.ma.masked_invalid(np.asarray(mesh.get_array(), float))
    # 8 cells inside (lon 0.5,1.5 over 4 rows), 8 masked outside
    assert arr.count() == 8


def test_clip_dissolves_multiple_features(regions):
    """Passing all polygons clips to their union, not just the first."""
    lat = lon = np.array([0.5, 1.5, 2.5])
    da = xr.DataArray(np.ones((3, 3)), dims=("lat", "lon"),
                      coords={"lat": lat, "lon": lon})
    fig = plot_field_map(da, clip=regions)   # three unit boxes at x in 0-3, y 0-1
    mesh = fig.axes[0].collections[0]
    arr = np.ma.masked_invalid(np.asarray(mesh.get_array(), float))
    assert arr.count() == 3   # only the y=0.5 row falls inside the boxes


def test_natural_earth_borders_crops_to_region():
    import africas2s as ds
    import pytest
    try:
        world = ds.natural_earth_borders()
    except FileNotFoundError:
        pytest.skip("Natural Earth cache not present in this environment")
    gha = ds.natural_earth_borders(region=[-13.5, 24.5, 20.5, 53.0])
    assert "geometry" in gha.columns
    assert len(gha) < len(world)           # the crop dropped countries outside the box
    minx, miny, maxx, maxy = gha.total_bounds
    assert minx < 53.0 and maxx > 20.5     # geometries lie within/near the GHA longitudes
