"""render_styled_terciles: ax-level wrapper over plot_tercile_forecast(style=)."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import xarray as xr
from africas2s.plotting import TercileStyle, render_styled_terciles


def _probs():
    rng = np.random.default_rng(0)
    p = rng.random((3, 5, 6)); p = p / p.sum(axis=0, keepdims=True)
    return xr.DataArray(p, dims=("tercile", "lat", "lon"),
                        coords={"tercile": [0, 1, 2], "lat": np.linspace(-4, 4, 5),
                                "lon": np.linspace(34, 44, 6)})


def _style():
    return TercileStyle(below_colors=["#ffe878", "#ff6000"], normal_colors=["#e1e1e1", "#646464"],
                        above_colors=["#c8ffbe", "#37d23c"], prob_bins=[33.33, 50, 100.01])


def test_renders_onto_supplied_axis():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    out = render_styled_terciles(ax, _probs(), _style(), title="panel")
    assert out is ax and ax.get_title() == "panel"
    assert ax.images or ax.collections
    plt.close(fig)


def test_small_drops_ticks():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    render_styled_terciles(ax, _probs(), _style(), small=True)
    assert len(ax.get_xticks()) == 0 and len(ax.get_yticks()) == 0
    plt.close(fig)
