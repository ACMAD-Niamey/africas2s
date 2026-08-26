"""Predictor + predictand domain bbox plot on a Cartopy basemap."""

from .._optional import require_optional


_HINT = "pip install africas2s[plotting]"


def plot_domains(predictor_extent, predictand_extent, *, ax=None, title=None):
    """Draw predictor and predictand bounding boxes on a Cartopy basemap.

    Parameters
    ----------
    predictor_extent : (lat_s, lat_n, lon_w, lon_e)
        Bounding box of the predictor domain. If `lon_w > lon_e`, the box
        is interpreted as crossing the antimeridian (e.g. tropical Pacific).
    predictand_extent : (lat_s, lat_n, lon_w, lon_e)
        Bounding box of the predictand domain.
    ax
        Optional pre-existing matplotlib axes (must be a cartopy GeoAxes).
        If None, a new figure with a PlateCarree axis is created.
    title
        Optional title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import importlib

    require_optional("matplotlib", _HINT)
    require_optional("cartopy", _HINT)
    plt = importlib.import_module("matplotlib.pyplot")
    patches = importlib.import_module("matplotlib.patches")
    ccrs = importlib.import_module("cartopy.crs")

    if ax is None:
        # Centre on the dateline only if either extent crosses it; otherwise
        # the Greenwich-centred view keeps non-Pacific regions in frame.
        crosses_antimeridian = (
            predictor_extent[2] > predictor_extent[3]
            or predictand_extent[2] > predictand_extent[3]
        )
        central_lon = 180 if crosses_antimeridian else 0
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=central_lon))
    else:
        fig = ax.figure

    ax.set_global()
    ax.coastlines(linewidth=0.5)
    ax.gridlines(linestyle=":", alpha=0.5)

    for extent, color, label in (
        (predictor_extent, "tab:red", "predictor"),
        (predictand_extent, "tab:blue", "predictand"),
    ):
        lat_s, lat_n, lon_w, lon_e = extent
        if lon_w > lon_e:
            # Antimeridian crossing: draw two rectangles, one per side
            for w, e in ((lon_w, 180), (-180, lon_e)):
                rect = patches.Rectangle(
                    (w, lat_s), e - w, lat_n - lat_s,
                    linewidth=2, edgecolor=color, facecolor="none",
                    transform=ccrs.PlateCarree(), label=label,
                )
                ax.add_patch(rect)
                label = None  # only label once
        else:
            rect = patches.Rectangle(
                (lon_w, lat_s), lon_e - lon_w, lat_n - lat_s,
                linewidth=2, edgecolor=color, facecolor="none",
                transform=ccrs.PlateCarree(), label=label,
            )
            ax.add_patch(rect)

    ax.legend(loc="lower left")
    if title:
        ax.set_title(title)

    return fig
