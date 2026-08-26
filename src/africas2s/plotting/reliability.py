"""Reliability diagram (forecast probability vs observed frequency)."""

import numpy as np
from .._optional import require_optional


_HINT = "pip install africas2s[plotting]"


def plot_reliability_diagram(forecast, obs, *, n_bins=5, ax=None, title=None):
    """Reliability diagram: forecast probability vs observed frequency.

    For each tercile category (BN/NN/AN), bins forecast probabilities and
    plots the observed frequency in each bin. The diagonal y=x is the
    perfect-calibration reference; points above the diagonal indicate
    underforecasting, points below indicate overforecasting.
    """
    import importlib
    require_optional("matplotlib", _HINT)
    plt = importlib.import_module("matplotlib.pyplot")

    from ..metrics.rpss import _cpt_boundaries

    t33, t67 = _cpt_boundaries(obs.values)
    obs_vals = obs.values
    obs_cat = np.where(t33 > obs_vals, 0, np.where(t67 > obs_vals, 1, 2))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    cat_labels = ["Below normal", "Normal", "Above normal"]
    cat_colors = ["tab:red", "tab:grey", "tab:blue"]

    for cat in range(3):
        prob = forecast.isel(tercile=cat).values.flatten()
        binary = (obs_cat == cat).astype(float).flatten()
        valid = ~(np.isnan(prob) | np.isnan(binary))
        prob = prob[valid]
        binary = binary[valid]
        if prob.size == 0:
            continue

        bin_x = []
        bin_y = []
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            if i == n_bins - 1:
                bin_mask = (prob >= lo) & (prob <= hi)
            else:
                bin_mask = (prob >= lo) & (prob < hi)
            if bin_mask.sum() == 0:
                continue
            bin_x.append(prob[bin_mask].mean())
            bin_y.append(binary[bin_mask].mean())

        ax.plot(bin_x, bin_y, "o-", color=cat_colors[cat], label=cat_labels[cat], linewidth=1.5)

    # Perfect-calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.6, label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Forecast probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title or "Reliability diagram")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    return fig
