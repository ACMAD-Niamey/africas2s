"""EOF and CCA mode plots from a fitted CCAMethod.

These are the standard PyCPT-style diagnostic plots for inspecting what a
CCA fit has learned: the dominant spatial patterns of variability on each
side (EOF modes), and the paired predictor/predictand patterns coupled by
each canonical correlation (CCA modes).
"""

import numpy as np

from .._optional import require_optional


_HINT = "pip install africas2s[plotting]"


def _reconstruct_spatial(loadings, valid_mask, shape):
    """Scatter (n_valid_pts,) loadings back to a (lat, lon) grid, NaN-filling masked points."""
    full = np.full(int(np.prod(shape)), np.nan)
    full[valid_mask] = loadings
    return full.reshape(shape)


def _apply_sign_convention(spatial):
    """Flip so the dominant lobe (largest |value|) is positive (PyCPT convention).

    EOF/CCA mode signs are mathematically arbitrary; without a convention,
    plots randomly flip across runs. Returns the (possibly flipped) array
    and the sign that was applied (+1 or -1) so paired plots can stay locked.
    """
    abs_arr = np.abs(spatial)
    if not np.any(np.isfinite(abs_arr)):
        return spatial, 1.0
    idx = int(np.nanargmax(abs_arr))
    sign = 1.0 if spatial.flat[idx] >= 0 else -1.0
    return spatial * sign, sign


def _coords_to_arrays(coords):
    lat = coords["lat"]
    lon = coords["lon"]
    return (
        lat.values if hasattr(lat, "values") else np.asarray(lat),
        lon.values if hasattr(lon, "values") else np.asarray(lon),
    )


def _projection_for(lon, ccrs):
    """Pick a PlateCarree central longitude that keeps the data in one piece.

    If the data extends past 180°E (i.e. uses a 0-360° convention with
    Pacific-spanning lons), centre the projection on the dateline so the
    region sits in the middle of the frame rather than splitting at the seam.
    """
    central = 180.0 if float(np.max(lon)) > 180.0 else 0.0
    return ccrs.PlateCarree(central_longitude=central)


def _frame_to_data(ax, lat, lon, ccrs, pad_frac=0.05):
    """Zoom the axes to the data bounds with a small fractional pad."""
    lat_pad = (float(np.max(lat)) - float(np.min(lat))) * pad_frac
    lon_pad = (float(np.max(lon)) - float(np.min(lon))) * pad_frac
    ax.set_extent(
        [
            float(np.min(lon)) - lon_pad,
            float(np.max(lon)) + lon_pad,
            float(np.min(lat)) - lat_pad,
            float(np.max(lat)) + lat_pad,
        ],
        crs=ccrs.PlateCarree(),
    )


def plot_eof_modes(cca_fit, kind="predictor", n_modes=3, *, ncols=3):
    """Plot EOF spatial loadings for one side of a fitted CCAMethod.

    Each panel shows the spatial loading of one EOF mode, with explained
    variance in the title and a sign convention applied so plots are
    visually consistent across runs.

    Parameters
    ----------
    cca_fit
        A fitted `CCAMethod` instance.
    kind : {"predictor", "predictand"}
        Which side of the CCA to plot.
    n_modes
        Number of EOF modes to draw. Capped at the available number of modes.
    ncols
        Columns in the grid layout.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import importlib
    import math

    require_optional("matplotlib", _HINT)
    require_optional("cartopy", _HINT)
    plt = importlib.import_module("matplotlib.pyplot")
    ccrs = importlib.import_module("cartopy.crs")

    if kind == "predictor":
        eof, sv, valid, shape, coords = (
            cca_fit.eofx_, cca_fit.svx_, cca_fit.x_valid_,
            cca_fit.predictor_shape_, cca_fit.predictor_coords_,
        )
    elif kind == "predictand":
        eof, sv, valid, shape, coords = (
            cca_fit.eofy_, cca_fit.svy_, cca_fit.y_valid_,
            cca_fit.predictand_shape_, cca_fit.predictand_coords_,
        )
    else:
        raise ValueError(f"kind must be 'predictor' or 'predictand', got {kind!r}")

    n_modes = min(n_modes, eof.shape[1])
    var_frac = sv ** 2 / np.sum(sv ** 2)
    lat, lon = _coords_to_arrays(coords)
    proj = _projection_for(lon, ccrs)

    nrows = math.ceil(n_modes / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 4 * nrows),
        subplot_kw={"projection": proj},
        squeeze=False,
    )

    for i in range(n_modes):
        spatial = _reconstruct_spatial(eof[:, i], valid, shape)
        spatial, _ = _apply_sign_convention(spatial)
        vmax = float(np.nanmax(np.abs(spatial)))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        ax = axes[i // ncols][i % ncols]
        im = ax.pcolormesh(
            lon, lat, spatial,
            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            transform=ccrs.PlateCarree(),
        )
        ax.coastlines(linewidth=0.5)
        _frame_to_data(ax, lat, lon, ccrs)
        ax.set_title(f"{kind} EOF {i + 1} ({var_frac[i] * 100:.1f}% var)")
        plt.colorbar(im, ax=ax, fraction=0.046)

    for j in range(n_modes, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.tight_layout()
    return fig


def plot_cca_modes(cca_fit, n_modes=3):
    """Plot paired predictor + predictand CCA mode patterns with canonical correlation.

    Each row shows one CCA mode: predictor pattern on the left, predictand
    pattern on the right, with the canonical correlation `r` annotated.
    Sign is locked between the two so the pair stays in phase.

    Parameters
    ----------
    cca_fit
        A fitted `CCAMethod` instance.
    n_modes
        Number of CCA modes to draw (one row per mode). Capped at `ncc_`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import importlib

    require_optional("matplotlib", _HINT)
    require_optional("cartopy", _HINT)
    plt = importlib.import_module("matplotlib.pyplot")
    ccrs = importlib.import_module("cartopy.crs")

    n_modes = min(n_modes, cca_fit.ncc_)

    # CCA spatial patterns: linear combinations of EOFs weighted by CCA rotations.
    # cca_fit.s_ is (ncc, nxe), so column i of `eofx_ @ s_.T` gives mode i's
    # predictor pattern. cca_fit.r_ is (nye, ncc), so `eofy_ @ r_` likewise.
    pred_patterns = cca_fit.eofx_ @ cca_fit.s_.T  # (n_pred_pts, ncc)
    pand_patterns = cca_fit.eofy_ @ cca_fit.r_     # (n_pand_pts, ncc)

    p_lat, p_lon = _coords_to_arrays(cca_fit.predictor_coords_)
    o_lat, o_lon = _coords_to_arrays(cca_fit.predictand_coords_)
    p_proj = _projection_for(p_lon, ccrs)
    o_proj = _projection_for(o_lon, ccrs)

    fig = plt.figure(figsize=(10, 4 * n_modes))
    axes = np.empty((n_modes, 2), dtype=object)
    for i in range(n_modes):
        axes[i][0] = fig.add_subplot(n_modes, 2, 2 * i + 1, projection=p_proj)
        axes[i][1] = fig.add_subplot(n_modes, 2, 2 * i + 2, projection=o_proj)

    for i in range(n_modes):
        p_spatial = _reconstruct_spatial(
            pred_patterns[:, i], cca_fit.x_valid_, cca_fit.predictor_shape_
        )
        o_spatial = _reconstruct_spatial(
            pand_patterns[:, i], cca_fit.y_valid_, cca_fit.predictand_shape_
        )
        # Sign convention: pick from predictor's dominant lobe and apply the
        # same sign to the predictand so the paired patterns stay in phase.
        p_spatial, sign = _apply_sign_convention(p_spatial)
        o_spatial = o_spatial * sign

        for col, (spatial, lat, lon, side) in enumerate([
            (p_spatial, p_lat, p_lon, "predictor"),
            (o_spatial, o_lat, o_lon, "predictand"),
        ]):
            vmax = float(np.nanmax(np.abs(spatial)))
            if not np.isfinite(vmax) or vmax == 0:
                vmax = 1.0
            ax = axes[i][col]
            im = ax.pcolormesh(
                lon, lat, spatial,
                cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                transform=ccrs.PlateCarree(),
            )
            ax.coastlines(linewidth=0.5)
            _frame_to_data(ax, lat, lon, ccrs)
            ax.set_title(f"{side} CCA {i + 1} (r={cca_fit.mu_[i]:.2f})")
            plt.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    return fig
