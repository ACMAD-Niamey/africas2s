"""Spatial skill-map plots from a SkillReport."""

from .._optional import require_optional


_HINT = "pip install africas2s[plotting]"

# Per-metric (cmap, vmin, vmax). vmin/vmax of None means auto-scale.
# Fixed ranges enforce cross-panel comparability (issue §3.3 pitfall).
_METRIC_STYLE = {
    "rpss":                    ("RdBu",     -1.0, 1.0),
    "pearson_r":               ("RdBu",     -1.0, 1.0),
    "spearman":                ("RdBu",     -1.0, 1.0),
    "hss":                     ("RdBu",     -1.0, 1.0),
    "heidke_skill_score":      ("RdBu",     -1.0, 1.0),
    "roc":                     ("viridis",   0.0, 1.0),
    "rmse":                    ("viridis_r", None, None),
    "root_mean_squared_error": ("viridis_r", None, None),
}


def plot_skill_maps(skill_report, metric_names, *, ncols=3):
    """Grid of per-metric spatial skill maps from `SkillReport.spatial`.

    Each panel uses a fixed colormap and value range per metric, so panels
    are visually comparable. Unknown metrics fall back to viridis with
    auto-range.

    Parameters
    ----------
    skill_report
        A `africas2s.skill.SkillReport` with populated `.spatial` dict.
    metric_names
        List of metric names to render. Each must be a key of
        `skill_report.spatial`.
    ncols
        Columns in the grid layout. Rows are computed as ceil(n / ncols).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import importlib
    import math

    # require_optional checks the package is present; we then import the
    # submodule directly so we get the right object (not the top-level package).
    require_optional("matplotlib", _HINT)
    require_optional("cartopy", _HINT)
    plt = importlib.import_module("matplotlib.pyplot")
    ccrs = importlib.import_module("cartopy.crs")

    n = len(metric_names)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 4 * nrows),
        subplot_kw={"projection": ccrs.PlateCarree()},
        squeeze=False,
    )

    for i, name in enumerate(metric_names):
        ax = axes[i // ncols][i % ncols]
        if name not in skill_report.spatial:
            ax.set_visible(False)
            continue
        da = skill_report.spatial[name]
        cmap, vmin, vmax = _METRIC_STYLE.get(name, ("viridis", None, None))
        im = ax.pcolormesh(
            da.lon, da.lat, da.values,
            cmap=cmap, vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree(),
        )
        ax.coastlines(linewidth=0.5)
        ax.set_title(name)
        plt.colorbar(im, ax=ax, fraction=0.046, label=name)

    # Hide unused trailing axes
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.tight_layout()
    return fig
