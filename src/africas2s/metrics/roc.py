import numpy as np
import xarray as xr
from .base import MetricBase
from ..registry import register_metric
from .rpss import _cpt_boundaries


def _roc_arrays(forecast, obs, cat_idx):
    """Return (fpr, tpr, area) for one tercile category.

    cat_idx: 0=BN, 1=NN, 2=AN. Used by both ROCMetric.compute() (which keeps
    only the area) and ROCMetric.compute_diagram() (which keeps the curve).
    """
    t33, t67 = _cpt_boundaries(obs.values)
    spatial_dims = [d for d in obs.dims if d != "year"]
    spatial_coords = {k: v for k, v in obs.coords.items()
                      if k != "year" and set(obs[k].dims).issubset(set(spatial_dims))}
    t33_da = xr.DataArray(t33, dims=spatial_dims, coords=spatial_coords)
    t67_da = xr.DataArray(t67, dims=spatial_dims, coords=spatial_coords)
    obs_cat = xr.where(t33_da > obs, 0, xr.where(t67_da > obs, 1, 2))

    binary_obs = (obs_cat == cat_idx).values.flatten()
    prob_fcst = forecast.isel(tercile=cat_idx).values.flatten()

    mask = ~(np.isnan(binary_obs) | np.isnan(prob_fcst))
    binary_obs = binary_obs[mask]
    prob_fcst = prob_fcst[mask]

    if len(binary_obs) == 0 or binary_obs.sum() == 0 or binary_obs.sum() == len(binary_obs):
        # Degenerate; mirror the prior scalar return of 0.5 area with a
        # trivial diagonal curve so callers receive a usable shape.
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5

    order = np.argsort(-prob_fcst)
    binary_obs = binary_obs[order]

    n_pos = binary_obs.sum()
    n_neg = len(binary_obs) - n_pos
    tp = np.cumsum(binary_obs)
    fp = np.cumsum(1 - binary_obs)
    tpr = tp / n_pos
    fpr = fp / n_neg

    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def _roc_per_category(forecast, obs, cat_idx):
    """Back-compat shim — area only. Kept so existing imports don't break."""
    _, _, area = _roc_arrays(forecast, obs, cat_idx)
    return area


@register_metric("roc")
class ROCMetric(MetricBase):
    def compute(self, forecast, obs, spatial=False, **kwargs):
        return {
            "roc_bn": _roc_per_category(forecast, obs, 0),
            "roc_nn": _roc_per_category(forecast, obs, 1),
            "roc_an": _roc_per_category(forecast, obs, 2),
        }

    def compute_diagram(self, forecast, obs, **kwargs):
        out = {}
        for label, cat_idx in (("bn", 0), ("nn", 1), ("an", 2)):
            fpr, tpr, area = _roc_arrays(forecast, obs, cat_idx)
            out[label] = {"fpr": fpr, "tpr": tpr, "area": area}
        return out


@register_metric("roc_area_below_normal")
class ROCBelowNormalMetric(MetricBase):
    def compute(self, forecast, obs, spatial=False, **kwargs):
        return _roc_per_category(forecast, obs, 0)


@register_metric("roc_area_above_normal")
class ROCAboveNormalMetric(MetricBase):
    def compute(self, forecast, obs, spatial=False, **kwargs):
        return _roc_per_category(forecast, obs, 2)
