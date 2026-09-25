"""FieldScale: packaged classified colour scales for continuous fields, and the
``scale=`` entry points on plot_field / plot_matrix.

A scale is colours + bin edges + which ends are open. Both renderers
(pcolormesh cells and the smooth contour path) must paint exactly the scale's
colours, bin for bin, including the out-of-range ends.
"""
import json

import numpy as np
import pytest
import xarray as xr

from africas2s.plotting.style import FieldScale, TercileStyle

PACKAGED = {"noaa-cpc-anomaly", "ucsb-chirps-anomaly", "ucsb-chirps-total",
            "icpac-onset-date", "icpac-onset-spread"}
_SLOTS = {"neither": 0, "min": 1, "max": 1, "both": 2}


def _hex(rgba):
    from matplotlib.colors import to_hex
    return to_hex(rgba).upper()


def _field(lo, hi, n_lat=6, n_lon=7):
    """A smooth ramp from lo to hi, so every bin between them is hit."""
    v = np.linspace(lo, hi, n_lat * n_lon).reshape(n_lat, n_lon)
    return xr.DataArray(v, dims=("lat", "lon"),
                        coords={"lat": np.linspace(-4, 4, n_lat),
                                "lon": np.linspace(30, 42, n_lon)})


# ------------------------------------------------------------------ loading

def test_top_level_reexport():
    import africas2s
    assert africas2s.FieldScale is FieldScale
    assert "FieldScale" in africas2s.__all__
    from africas2s.plotting import FieldScale as plotting_scale
    assert plotting_scale is FieldScale


def test_every_packaged_scale_is_listed_and_well_formed():
    listed = FieldScale.list_named()
    assert PACKAGED <= set(listed)
    for name in listed:
        scale = FieldScale.named(name)
        assert listed[name]                                   # provenance text
        assert scale.label
        assert list(scale.levels) == sorted(scale.levels)
        assert len(scale.colors) == len(scale.levels) - 1 + _SLOTS[scale.extend]


@pytest.mark.parametrize("name, n_colors, extend", [
    ("noaa-cpc-anomaly", 15, "both"),       # 7 dry + white + 7 wet
    ("ucsb-chirps-anomaly", 14, "both"),    # 6 deficit + white + 7 surplus
    ("ucsb-chirps-total", 16, "both"),      # white under 2 mm ... pale pink over 2500
    ("icpac-onset-date", 12, "both"),       # grey before the window, 10 dekads, later
    ("icpac-onset-spread", 6, "both"),      # under-arrow, 0-5, 5-10, 10-20, 20-30, over 30
])
def test_packaged_scale_shapes(name, n_colors, extend):
    scale = FieldScale.named(name)
    assert len(scale.colors) == n_colors and scale.extend == extend


def test_anomaly_scales_are_white_around_zero():
    for name in ("noaa-cpc-anomaly", "ucsb-chirps-anomaly"):
        scale = FieldScale.named(name)
        assert _hex(scale.cmap(scale.norm(0.0))) == "#FFFFFF"


def test_unknown_name_lists_what_exists():
    with pytest.raises(ValueError, match="noaa-cpc-anomaly"):
        FieldScale.named("nope")


def test_from_json_rejects_unknown_keys(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"colors": ["#000000", "#ffffff"],
                                "levels": [0, 1, 2], "colours": []}))
    with pytest.raises(ValueError, match="colours"):
        FieldScale.from_json(path)


def test_overrides_win_over_the_file():
    scale = FieldScale.named("icpac-onset-spread", label="days")
    assert scale.label == "days"


def test_colour_count_must_match_bins_plus_open_ends():
    with pytest.raises(ValueError, match="3 entries"):
        FieldScale(colors=["#000000", "#ffffff"], levels=[0, 1, 2], extend="min")
    with pytest.raises(ValueError, match="extend"):
        FieldScale(colors=["#000000", "#ffffff"], levels=[0, 1, 2], extend="up")
    with pytest.raises(ValueError, match="increasing"):
        FieldScale(colors=["#000000", "#ffffff"], levels=[2, 1, 0])


def test_bin_colors_drop_the_open_ends():
    scale = FieldScale(colors=["#111111", "#222222", "#333333", "#444444"],
                       levels=[0, 1, 2], extend="both")
    assert scale.bin_colors == ["#222222", "#333333"]
    assert FieldScale.named("icpac-onset-spread").bin_colors == \
        FieldScale.named("icpac-onset-spread").colors[1:-1]


# ------------------------------------------------------------------ rendering

def test_cmap_and_norm_map_values_to_the_listed_colours():
    pytest.importorskip("matplotlib")
    scale = FieldScale.named("noaa-cpc-anomaly")
    got = [_hex(c) for c in scale.cmap(scale.norm(np.array([-500, -350, -30, 0, 30, 450])))]
    c = [x.upper() for x in scale.colors]
    assert got == [c[0], c[1], c[6], c[7], c[8], c[-1]]


def test_plot_field_cells_paint_the_scale_colours():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from africas2s.plotting.forecasts import plot_field

    scale = FieldScale.named("ucsb-chirps-anomaly")
    fig, ax = plt.subplots()
    im = plot_field(_field(-400, 600), ax=ax, scale=scale)
    probe = np.array([-400, -250, 0, 15, 400, 600])
    got = [_hex(c) for c in im.cmap(im.norm(probe))]
    c = [x.upper() for x in scale.colors]
    assert got == [c[0], c[1], c[6], c[7], c[12], c[13]]
    plt.close(fig)


def test_plot_field_smooth_paints_the_same_colours_as_cells():
    """The contour path must not spread N colours evenly over the bins."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from africas2s.plotting.forecasts import plot_field

    scale = FieldScale.named("noaa-cpc-anomaly")
    fig, ax = plt.subplots()
    cs = plot_field(_field(-500, 500), ax=ax, scale=scale, smooth=2)
    assert cs is not None
    # cvalues are the per-band layer values, out-of-range ends included.
    got = [_hex(c) for c in cs.cmap(cs.norm(cs.cvalues))]
    assert got == [x.upper() for x in scale.colors]
    drawn = {_hex(c) for c in cs.get_facecolor()}
    assert drawn <= set(x.upper() for x in scale.colors)
    assert len(drawn) >= 10                       # the ramp crosses most bands
    # And a colorbar off the contour set draws without complaint.
    cb = fig.colorbar(cs, ax=ax)
    assert cb is not None
    plt.close(fig)


def test_plot_field_scale_conflicts_raise():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from africas2s.plotting.forecasts import plot_field

    scale = FieldScale.named("icpac-onset-spread")
    fig, ax = plt.subplots()
    for bad in (dict(levels=[0, 1]), dict(vmin=0), dict(vmax=1), dict(center=0)):
        with pytest.raises(ValueError, match="scale"):
            plot_field(_field(0, 40), ax=ax, scale=scale, **bad)
    plt.close(fig)


def test_plot_matrix_scale_shares_one_colorbar_with_the_scale_label():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from africas2s.plotting.panels import plot_matrix

    scale = FieldScale.named("ucsb-chirps-total")
    fig = plot_matrix({"week 1": _field(0, 800), "week 2": _field(0, 3000)},
                      scale=scale, ncols=2)
    cbars = [a for a in fig.axes if a.get_label() == "<colorbar>"]
    assert len(cbars) == 1
    assert cbars[0].get_xlabel() == scale.label
    plt.close(fig)
    fig = plot_matrix({"a": _field(0, 800)}, scale=scale, smooth=2,
                      cbar_label="mm")
    cbars = [a for a in fig.axes if a.get_label() == "<colorbar>"]
    assert cbars[0].get_xlabel() == "mm"          # an explicit label still wins
    plt.close(fig)
    with pytest.raises(ValueError, match="scale"):
        plot_matrix({"a": _field(0, 1)}, scale=scale, levels=[0, 1])


# ------------------------------------------------------------------ the tercile side

def test_noaa_nmme_was_renamed_noaa_cpc():
    with pytest.raises(ValueError, match="noaa-cpc"):
        TercileStyle.named("noaa-nmme")


def test_icpac_onset_style_and_legend_labels():
    pytest.importorskip("matplotlib")
    from africas2s.plotting.panels import tercile_legend_handles

    style = TercileStyle.named("icpac-onset")
    assert len(style.below_colors) == 7 and style.prob_bins[0] == pytest.approx(33.33)
    assert style.secondary_max is None
    labels = [h.get_label() for h in tercile_legend_handles(style, variable_kind="onset")]
    assert labels[0].startswith("Early") and labels[2].startswith("Late")
    labels = [h.get_label() for h in tercile_legend_handles(variable_kind="onset")]
    assert labels[0].startswith("Early") and labels[2].startswith("Late")


def test_onset_variable_kind_renders():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from africas2s.plotting.forecasts import plot_tercile_forecast

    rng = np.random.default_rng(2)
    p = rng.dirichlet([1, 1, 1], size=(6, 7)).transpose(2, 0, 1)
    probs = xr.DataArray(p, dims=("tercile", "lat", "lon"),
                         coords={"tercile": [0, 1, 2], "lat": np.linspace(-4, 4, 6),
                                 "lon": np.linspace(30, 42, 7)})
    for style in (None, TercileStyle.named("icpac-onset")):
        fig = plot_tercile_forecast(probs, style=style, variable_kind="onset")
        texts = {t.get_text() for t in fig.axes[0].get_legend().get_texts()}
        assert any(t.startswith(("Early", r"$\bf{Early}$")) for t in texts), texts
        plt.close(fig)
