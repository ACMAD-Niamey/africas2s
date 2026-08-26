import numpy as np
from scipy.stats import rankdata
import xarray as xr
from .base import MetricBase
from ..registry import register_metric


def _rank_with_nan(arr):
    """1D rank along an axis with NaN passthrough.

    Returns an array of the same shape; NaN values stay NaN, non-NaN values
    get average ranks among the valid entries (matches scipy default
    `method='average'`).
    """
    out = np.full(arr.shape, np.nan, dtype=float)
    valid = ~np.isnan(arr)
    if valid.any():
        out[valid] = rankdata(arr[valid], method="average")
    return out


@register_metric("spearman")
class SpearmanMetric(MetricBase):
    """Spearman rank correlation.

    Spearman is Pearson correlation computed on rank-transformed values along
    the year axis. Captures monotonic (including nonlinear) relationships
    between forecast and obs. Tied values are handled via average ranks
    (matches `scipy.stats.spearmanr` default).
    """

    def compute(self, forecast, obs, spatial=False, **kwargs):
        if "member" in forecast.dims:
            forecast = forecast.mean("member")
        if "tercile" in forecast.dims:
            raise ValueError(
                "spearman requires a continuous-valued forecast; got a "
                "forecast with a 'tercile' dim (tercile probabilities). Pass "
                "the deterministic ensemble forecast instead."
            )

        f_rank = xr.apply_ufunc(
            _rank_with_nan, forecast,
            input_core_dims=[["year"]],
            output_core_dims=[["year"]],
            vectorize=True,
        )
        o_rank = xr.apply_ufunc(
            _rank_with_nan, obs,
            input_core_dims=[["year"]],
            output_core_dims=[["year"]],
            vectorize=True,
        )

        f_anom = f_rank - f_rank.mean("year")
        o_anom = o_rank - o_rank.mean("year")

        rho = (f_anom * o_anom).sum("year") / (
            np.sqrt((f_anom**2).sum("year")) * np.sqrt((o_anom**2).sum("year"))
        )

        if spatial:
            return rho
        return float(rho.mean())
