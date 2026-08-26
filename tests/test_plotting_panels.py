import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


# ===================================================================
# Plotting panels: plot_matrix / plot_components_objective / styles
# ===================================================================

N_LAT, N_LON = 5, 7
LAT = np.linspace(-5, 5, N_LAT)
LON = np.linspace(30, 45, N_LON)

STYLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "styles"


def _ghacof(**overrides):
    from africas2s import TercileStyle
    return TercileStyle.from_json(STYLES_DIR / "ghacof.json", **overrides)


def _probs(seed=0, lat_name="lat", lon_name="lon"):
    rng = np.random.RandomState(seed)
    p = rng.dirichlet([1.0, 1.0, 1.0], size=(N_LAT, N_LON)).transpose(2, 0, 1)
    return xr.DataArray(
        p, dims=["tercile", lat_name, lon_name],
        coords={"tercile": [0, 1, 2], lat_name: LAT, lon_name: LON},
    )


def _field(seed=0, scale=1.0):
    rng = np.random.RandomState(seed)
    return xr.DataArray(
        rng.randn(N_LAT, N_LON) * scale,
        dims=["lat", "lon"], coords={"lat": LAT, "lon": LON},
    )


def _bool_mask(fn):
    LON2, LAT2 = np.meshgrid(LON, LAT)
    return xr.DataArray(fn(LAT2, LON2), dims=["lat", "lon"],
                        coords={"lat": LAT, "lon": LON})


# ------------------------------------------------------------------ imports

def test_top_level_reexports():
    import africas2s as ds
    for name in ("plot_matrix", "plot_components_objective",
                 "tercile_legend_handles", "TercileStyle",
                 "tercile_diverging_cmap"):
        assert callable(getattr(ds, name))
        assert name in ds.__all__
    for name in ("ghacof_style", "acmad_style"):   # factories removed: styles
        assert not hasattr(ds, name)               # now load from JSON files


# ------------------------------------------------------- panel normalization

def test_normalize_panels_accepts_dict_pairs_and_triples():
    from africas2s.plotting.panels import _normalize_panels

    d = _normalize_panels({"a": 1, "b": 2})
    assert d == [("a", 1, None), ("b", 2, None)]
    p = _normalize_panels([("a", 1), ("b", 2, "style")])
    assert p == [("a", 1, None), ("b", 2, "style")]


def test_normalize_panels_rejects_bad_entries():
    from africas2s.plotting.panels import _normalize_panels

    with pytest.raises(TypeError, match="each panel must be"):
        _normalize_panels([1, 2])
    with pytest.raises(ValueError, match="no panels"):
        _normalize_panels([])


# ------------------------------------------------------- styles from JSON

def test_from_json_loads_shipped_ghacof_file():
    from africas2s import TercileStyle

    style = _ghacof()
    assert isinstance(style, TercileStyle)
    assert len(style.below_colors) == len(style.prob_bins) - 1 == 5
    assert style.lakes is True


def test_from_json_loads_shipped_acmad_file():
    from africas2s import TercileStyle

    style = TercileStyle.from_json(STYLES_DIR / "acmad.json")
    assert len(style.below_colors) == len(style.prob_bins) - 1 == 6
    assert style.prob_bins[0] == 33.33
    assert style.lakes is False


def test_from_json_overrides_win():
    style = _ghacof(dry_color="#123456", lakes=False)
    assert style.dry_color == "#123456"
    assert style.lakes is False


def test_from_json_underscore_keys_are_comments(tmp_path):
    from africas2s import TercileStyle

    p = tmp_path / "style.json"
    p.write_text(json.dumps({
        "_provenance": "hand-written for the test",
        "below_colors": ["#111111"], "normal_colors": ["#222222"],
        "above_colors": ["#333333"], "prob_bins": [33.3, 100.01],
    }))
    style = TercileStyle.from_json(p)
    assert style.below_colors == ["#111111"]


def test_from_json_rejects_unknown_keys(tmp_path):
    from africas2s import TercileStyle

    p = tmp_path / "style.json"
    p.write_text(json.dumps({
        "below_colors": ["#111111"], "normal_colors": ["#222222"],
        "above_colors": ["#333333"], "prob_bins": [33.3, 100.01],
        "below_colours": ["#111111"],       # typo'd field name
    }))
    with pytest.raises(ValueError, match="below_colours"):
        TercileStyle.from_json(p)


def test_from_json_rejects_non_object(tmp_path):
    from africas2s import TercileStyle

    p = tmp_path / "style.json"
    p.write_text(json.dumps(["#111111"]))
    with pytest.raises(ValueError, match="JSON object"):
        TercileStyle.from_json(p)


def test_tercile_diverging_cmap_endpoints():
    pytest.importorskip("matplotlib")
    from matplotlib.colors import to_rgba
    from africas2s.plotting import tercile_diverging_cmap

    style = _ghacof()
    cmap = tercile_diverging_cmap(style)
    assert np.allclose(cmap(0.0), to_rgba(style.below_colors[-1]), atol=0.01)
    assert np.allclose(cmap(1.0), to_rgba(style.above_colors[-1]), atol=0.01)
    assert np.allclose(cmap(0.5), to_rgba("#ffffff"), atol=0.02)


# ------------------------------------------------------------ legend handles

def test_tercile_legend_handles_no_style():
    pytest.importorskip("matplotlib")
    from africas2s.plotting import tercile_legend_handles

    handles = tercile_legend_handles()
    assert len(handles) == 3
    assert "drier" in handles[0].get_label()

    temp = tercile_legend_handles(variable_kind="temp")
    assert "cooler" in temp[0].get_label()


def test_tercile_legend_handles_styled_counts():
    pytest.importorskip("matplotlib")
    from africas2s.plotting import tercile_legend_handles

    style = _ghacof()             # no dry mask -> no dry patch
    assert len(tercile_legend_handles(style)) == 3
    assert len(tercile_legend_handles(style, detailed=True)) == 15
    assert len(tercile_legend_handles(style, detailed=True, include_dry=True)) == 16

    labels = [h.get_label() for h in tercile_legend_handles(style, detailed=True)]
    assert any(">70%" in lbl for lbl in labels)
    assert any(lbl.startswith("Below") for lbl in labels)


def test_tercile_legend_handles_dry_follows_style_mask():
    pytest.importorskip("matplotlib")
    from africas2s.plotting import tercile_legend_handles

    style = _ghacof(dry_mask=_bool_mask(lambda la, lo: la > 3))
    handles = tercile_legend_handles(style)
    assert len(handles) == 4
    assert handles[-1].get_label() == "Dry-masked / no data"


def test_tercile_legend_handles_rejects_bad_kind():
    pytest.importorskip("matplotlib")
    from africas2s.plotting import tercile_legend_handles

    with pytest.raises(ValueError, match="variable_kind"):
        tercile_legend_handles(variable_kind="wind")


# ---------------------------------------------------------------- skill mask

def test_apply_skill_mask_blanks_cells():
    from africas2s.plotting.panels import _apply_skill_mask

    probs = _probs()
    mask = _bool_mask(lambda la, lo: lo > 40)
    out = _apply_skill_mask(probs, mask)
    assert out.where(mask).isnull().all()
    assert out.where(~mask).notnull().sum() == probs.where(~mask).notnull().sum()


def test_apply_skill_mask_aligns_coarser_grid():
    from africas2s.plotting.panels import _apply_skill_mask

    coarse = xr.DataArray(
        [[True, False], [True, False]], dims=["lat", "lon"],
        coords={"lat": [-5.0, 5.0], "lon": [30.0, 45.0]},
    )
    out = _apply_skill_mask(_field(), coarse)
    assert bool(out.sel(lat=-5, lon=30, method="nearest").isnull())
    assert bool(out.sel(lat=-5, lon=45, method="nearest").notnull())


def test_apply_skill_mask_ndarray_shape_mismatch_raises():
    from africas2s.plotting.panels import _apply_skill_mask

    with pytest.raises(ValueError, match="skill_mask"):
        _apply_skill_mask(_field(), np.zeros((2, 2), dtype=bool))


# ------------------------------------------------------------- refine/smooth

def test_smooth_factor_normalization():
    from africas2s.plotting.forecasts import _smooth_factor

    assert _smooth_factor(False) == 0
    assert _smooth_factor(0) == 0
    assert _smooth_factor(True) == 4
    assert _smooth_factor(6) == 6
    with pytest.raises(ValueError, match="smooth"):
        _smooth_factor(1)


def test_refine_field_shape_and_footprint():
    from africas2s.plotting.forecasts import _refine_field

    values = _field(seed=3).values.copy()
    values[:, -2:] = np.nan                     # a NaN margin
    fine, flat, flon = _refine_field(values, LAT, LON, 3)
    assert fine.shape == (N_LAT * 3, N_LON * 3)
    assert flat[0] == LAT[0] and flat[-1] == LAT[-1]
    # the refined field is masked where the original had no data
    assert np.isnan(fine[:, -3:]).all()
    assert np.isfinite(fine[:, : N_LON]).all()


def test_refine_field_all_nan_returns_none():
    from africas2s.plotting.forecasts import _refine_field

    assert _refine_field(np.full((N_LAT, N_LON), np.nan), LAT, LON, 3) is None


# ------------------------------------------------------- plot_field additions

def test_plot_field_levels_returns_discrete_mappable():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    from africas2s.plotting.forecasts import plot_field

    im = plot_field(_field(scale=100.0), levels=[-100, -50, 0, 50, 100])
    assert isinstance(im.norm, BoundaryNorm)
    plt.close("all")


def test_plot_field_levels_conflicts_raise():
    pytest.importorskip("matplotlib")
    from africas2s.plotting.forecasts import plot_field

    with pytest.raises(ValueError, match="levels"):
        plot_field(_field(), levels=[0, 1], vmin=0.0)
    with pytest.raises(ValueError, match="levels"):
        plot_field(_field(), levels=[0, 1], center=0.0)
    with pytest.raises(ValueError, match="smooth"):
        plot_field(_field(), smooth=True)


def test_plot_field_smooth_returns_contour_mappable():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from africas2s.plotting.forecasts import plot_field

    im = plot_field(_field(scale=100.0), levels=[-100, -50, 0, 50, 100], smooth=3)
    assert im is not None and hasattr(im, "levels")
    plt.close("all")


def test_plot_field_smooth_all_nan_says_no_data():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from africas2s.plotting.forecasts import plot_field

    empty = _field() * np.nan
    fig, ax = plt.subplots()
    im = plot_field(empty, ax=ax, levels=[0, 1, 2], smooth=True)
    assert im is None
    assert any("no data" in t.get_text() for t in ax.texts)
    plt.close("all")


def test_plot_terciles_smooth_requires_style():
    pytest.importorskip("matplotlib")
    from africas2s.plotting.forecasts import plot_tercile_forecast

    with pytest.raises(ValueError, match="smooth"):
        plot_tercile_forecast(_probs(), smooth=True)


def test_plot_terciles_smooth_smoke():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from africas2s.plotting.forecasts import plot_tercile_forecast

    style = _ghacof(dry_mask=_bool_mask(lambda la, lo: la > 3), lakes=False)
    fig = plot_tercile_forecast(_probs(), style=style, smooth=True)
    assert fig is not None
    plt.close(fig)


# --------------------------------------------------------------- plot_matrix

def test_plot_matrix_smoke_and_trailing_axes_hidden():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from africas2s.plotting import plot_matrix

    style = _ghacof(lakes=False)
    fig = plot_matrix({"a": _probs(0), "b": _probs(1), "c": _probs(2)},
                      style=style, ncols=2, suptitle="matrix")
    map_axes = fig.axes[:4]
    assert map_axes[3].get_visible() is False
    assert all(ax.get_visible() for ax in map_axes[:3])
    plt.close(fig)


def test_plot_matrix_unstyled_smoke():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from africas2s.plotting import plot_matrix

    fig = plot_matrix([("a", _probs(0)), ("b", _probs(1))])
    assert fig is not None
    plt.close(fig)


def test_plot_matrix_field_panels_share_scale():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from africas2s.plotting import plot_matrix

    fig = plot_matrix([("small", _field(0, scale=1.0)),
                       ("big", _field(1, scale=10.0))], ncols=2, legend=False)
    clims = [ax.collections[0].get_clim() for ax in fig.axes[:2]]
    assert clims[0] == clims[1]
    plt.close(fig)


def test_plot_matrix_mixed_panels_get_colorbar_and_legend():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from africas2s.plotting import plot_matrix

    style = _ghacof(lakes=False)
    fig = plot_matrix([("terc", _probs(0)), ("anom", _field(1))],
                      style=style, ncols=2, cbar_label="mm")
    assert fig.legends, "expected a figure-level tercile legend"
    assert len(fig.axes) > 2, "expected an added colorbar axes"
    plt.close(fig)


def test_plot_matrix_mixed_panels_rows_stay_aligned():
    # The shared colorbar must take space from the whole grid, not just the
    # continuous panels' axes — else a continuous column renders shorter and
    # lower than a tercile column beside it.
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from africas2s.plotting import plot_matrix

    style = _ghacof(lakes=False)
    fig = plot_matrix([("anom", _field(1)), ("terc", _probs(0)),
                       ("anom2", _field(2)), ("terc2", _probs(1))],
                      style=style, ncols=2, cbar_label="mm")
    fig.canvas.draw()
    panels = fig.axes[:4]
    for left, right in ((panels[0], panels[1]), (panels[2], panels[3])):
        lbox, rbox = left.get_position(), right.get_position()
        assert lbox.height == pytest.approx(rbox.height, abs=1e-3), \
            "row panels differ in height"
        assert lbox.y1 == pytest.approx(rbox.y1, abs=1e-3), \
            "row panels differ in top alignment"
    plt.close(fig)


def test_plot_matrix_skill_mask_blanks_every_panel():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from africas2s.plotting import plot_matrix

    # Mask everything: the styled tercile render must then paint no category
    # cells at all (every cell is an incomplete triple -> nodata).
    style = _ghacof(lakes=False)
    all_mask = _bool_mask(lambda la, lo: np.ones_like(la, dtype=bool))
    fig = plot_matrix({"a": _probs(0)}, style=style, skill_mask=all_mask)
    arrays = [c.get_array() for c in fig.axes[0].collections
              if getattr(c, "get_array", None) and c.get_array() is not None]
    assert arrays, "expected the tercile QuadMesh"
    assert all(np.ma.getmaskarray(a).all() or np.isnan(np.ma.getdata(a)).all()
               for a in arrays)
    plt.close(fig)


def test_plot_matrix_accepts_latitude_longitude_dims():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from africas2s.plotting import plot_matrix

    fig = plot_matrix({"a": _probs(0, lat_name="latitude", lon_name="longitude")})
    assert fig is not None
    plt.close(fig)


def test_plot_matrix_per_panel_style_override():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from dataclasses import replace
    from africas2s.plotting import plot_matrix

    style = _ghacof(dry_mask=_bool_mask(lambda la, lo: la > 0), lakes=False)
    no_dry = replace(style, dry_mask=None)
    fig = plot_matrix([("masked", _probs(0)), ("unmasked", _probs(0), no_dry)],
                      style=style, ncols=2)
    assert fig is not None
    plt.close(fig)


def test_plot_matrix_levels_smooth_field_grid():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from africas2s.plotting import plot_matrix, tercile_diverging_cmap

    style = _ghacof(lakes=False)
    fig = plot_matrix([("a", _field(0, 100.0)), ("b", _field(1, 100.0))],
                      style=style, levels=[-150, -50, 50, 150],
                      cmap=tercile_diverging_cmap(style), smooth=3,
                      cbar_label="mm")
    assert fig is not None
    plt.close(fig)


# -------------------------------------------------- plot_components_objective

def test_plot_components_objective_emphasizes_last_panel():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from africas2s.plotting import plot_components_objective

    style = _ghacof(lakes=False)
    comps = {"m1": _probs(0), "m2": _probs(1), "m3": _probs(2)}
    objective = (_probs(0) + _probs(1) + _probs(2)) / 3
    fig = plot_components_objective(comps, objective, style=style,
                                    objective_label="OBJECTIVE")
    map_axes = fig.axes[:4]
    assert all(ax.get_visible() for ax in map_axes)
    obj_ax, comp_ax = map_axes[3], map_axes[0]
    assert max(s.get_linewidth() for s in obj_ax.spines.values()) > \
        max(s.get_linewidth() for s in comp_ax.spines.values())
    obj_chips = [t for t in obj_ax.texts if t.get_text() == "OBJECTIVE"]
    assert obj_chips and obj_chips[0].get_fontweight() == "bold"
    plt.close(fig)


# ------------------------------------------------------------- integration

@pytest.mark.integration
def test_components_objective_full_composite_integration(tmp_path):
    """Full styled composite against real Natural Earth data: country clip +
    lakes + smooth contours + detailed legend, saved to disk. Downloads the
    NE admin-0 shapefiles on first run (network)."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from africas2s.plotting import plot_components_objective

    lat = np.linspace(-12, 5, 30)
    lon = np.linspace(28, 42, 30)
    lon2, lat2 = np.meshgrid(lon, lat)
    rng = np.random.RandomState(0)

    def probs(shift):
        base = np.sin((lon2 + shift) / 4.0)
        above = np.clip(1 / 3 + 0.3 * base, 0.05, 0.9)
        below = np.clip(1 / 3 - 0.3 * base, 0.05, 0.9)
        normal = np.clip(1.0 - above - below, 0.05, None)
        p = np.stack([below, normal, above])
        return xr.DataArray(p / p.sum(0), dims=["tercile", "lat", "lon"],
                            coords={"tercile": [0, 1, 2], "lat": lat, "lon": lon})

    dry = xr.DataArray(lat2 < -10, dims=["lat", "lon"],
                       coords={"lat": lat, "lon": lon})
    style = _ghacof(dry_mask=dry,
                         clip_to=["Kenya", "United Republic of Tanzania", "Uganda"],
                         extent=(28, 42, -12, 5))
    comps = {"model A": probs(0), "model B": probs(2), "model C": probs(4)}
    objective = sum(comps.values()) / 3
    fig = plot_components_objective(comps, objective, style=style, smooth=True,
                                    legend_detailed=True,
                                    suptitle="integration composite")
    out = tmp_path / "composite.png"
    fig.savefig(out, dpi=80)
    plt.close(fig)
    assert out.stat().st_size > 20_000


# -------------------------------------------------------------- region_masks

def test_region_masks_public_wraps_dry_as_dataarray():
    import africas2s as ds

    dry = _bool_mask(lambda la, lo: la > 3)
    style = _ghacof(dry_mask=dry)
    dry_out, outside_out = ds.region_masks(_field(), style)
    assert isinstance(dry_out, xr.DataArray)
    assert dry_out.dims == ("lat", "lon")
    xr.testing.assert_equal(dry_out, dry)
    # no clip_to on the style -> all-False, so ~outside needs no None-guard
    assert isinstance(outside_out, xr.DataArray)
    assert not outside_out.any()


def test_region_masks_all_false_when_unset():
    import africas2s as ds

    dry_out, outside_out = ds.region_masks(_field(), _ghacof())
    assert not dry_out.any() and not outside_out.any()
    # the documented recipe works with mask-free styles
    masked = _field().where(~outside_out).where(~dry_out)
    assert masked.notnull().all()


def test_align_bool_mask_accepts_alias_dims():
    from africas2s.plotting.forecasts import _align_bool_mask

    mask = _bool_mask(lambda la, lo: lo > 40).rename(
        {"lat": "latitude", "lon": "longitude"})
    aligned = _align_bool_mask(mask, LAT, LON)
    assert aligned.shape == (N_LAT, N_LON)
    assert aligned[:, -1].all() and not aligned[:, 0].any()


def test_plot_matrix_partial_vmin_still_shares_scale():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from africas2s.plotting import plot_matrix

    fig = plot_matrix([("small", _field(0, scale=1.0)),
                       ("big", _field(1, scale=10.0))],
                      ncols=2, vmin=0.0, legend=False)
    clims = [ax.collections[0].get_clim() for ax in fig.axes[:2]]
    assert clims[0] == clims[1]
    assert clims[0][0] == 0.0
    plt.close(fig)


def test_smooth_falls_back_on_tiny_grids():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from africas2s.plotting.forecasts import plot_field, _refine_field

    narrow = xr.DataArray(np.random.RandomState(0).randn(3, 8) * 50,
                          dims=["lat", "lon"],
                          coords={"lat": [0.0, 1.0, 2.0], "lon": np.arange(8.0)})
    im = plot_field(narrow, levels=[-100, -50, 0, 50, 100], smooth=True)
    assert im is not None            # linear fallback, no cubic crash
    assert _refine_field(np.ones((1, 8)), [0.0], np.arange(8.0), 3) is None
    plt.close("all")
