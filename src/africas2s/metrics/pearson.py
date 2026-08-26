import numpy as np
import xarray as xr
from .base import MetricBase
from ..registry import register_metric

@register_metric("pearson_r")
class PearsonMetric(MetricBase):
    def compute(self, forecast, obs, spatial=False, **kwargs):
        if "member" in forecast.dims:
            forecast = forecast.mean("member")
        if "tercile" in forecast.dims:
            raise ValueError(
                "pearson_r requires a continuous-valued forecast; got a "
                "forecast with a 'tercile' dim (tercile probabilities). Pass "
                "the deterministic ensemble forecast instead."
            )

        f_anom = forecast - forecast.mean("year")
        o_anom = obs - obs.mean("year")

        r = (f_anom * o_anom).sum("year") / (
            np.sqrt((f_anom**2).sum("year")) * np.sqrt((o_anom**2).sum("year"))
        )

        if spatial:
            return r
        return float(r.mean())
