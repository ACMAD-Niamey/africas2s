"""Brier reliability term (calibration error).

For each forecast probability bin, computes
    n_bin × (mean_forecast_in_bin − observed_frequency_in_bin)²
summed over bins and divided by N. 0 = perfectly calibrated; positive =
overconfident or biased.
"""

import numpy as np
from .base import MetricBase
from ..registry import register_metric
from .rpss import _cpt_boundaries


_CAT_LABELS = ("bn", "nn", "an")


def _reliability_bins(forecast, obs, n_bins):
    """Return per-tercile bin payloads:
    [{"tercile": "bn", "bins": [{"mean_prob": ..., "obs_freq": ..., "n": ...}, ...]}, ...]
    """
    t33, t67 = _cpt_boundaries(obs.values)
    obs_vals = obs.values
    obs_cat = np.where(t33 > obs_vals, 0, np.where(t67 > obs_vals, 1, 2))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for cat in range(3):
        prob = forecast.isel(tercile=cat).values.flatten()
        binary = (obs_cat == cat).astype(float).flatten()
        valid = ~(np.isnan(prob) | np.isnan(binary))
        prob = prob[valid]
        binary = binary[valid]
        cat_bins = []
        if prob.size > 0:
            for i in range(n_bins):
                lo, hi = edges[i], edges[i + 1]
                if i == n_bins - 1:
                    bin_mask = (prob >= lo) & (prob <= hi)
                else:
                    bin_mask = (prob >= lo) & (prob < hi)
                n_bin = int(bin_mask.sum())
                if n_bin == 0:
                    continue
                cat_bins.append({
                    "mean_prob": float(prob[bin_mask].mean()),
                    "obs_freq": float(binary[bin_mask].mean()),
                    "n": n_bin,
                })
        out.append({"tercile": _CAT_LABELS[cat], "bins": cat_bins})
    return out


@register_metric("reliability")
class ReliabilityMetric(MetricBase):
    """Brier reliability decomposition term. Lower is better; 0 = perfect."""

    def compute(self, forecast, obs, spatial=False, n_bins=5, **kwargs):
        bins_per_cat = _reliability_bins(forecast, obs, n_bins)
        rel_total = 0.0
        n_total = 0
        for entry in bins_per_cat:
            for b in entry["bins"]:
                rel_total += b["n"] * (b["mean_prob"] - b["obs_freq"]) ** 2
                n_total += b["n"]
        if n_total == 0:
            return float("nan")
        return float(rel_total / n_total)

    def compute_diagram(self, forecast, obs, *, n_bins=5, **kwargs):
        return _reliability_bins(forecast, obs, n_bins)
