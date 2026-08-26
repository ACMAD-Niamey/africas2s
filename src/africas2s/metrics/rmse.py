import numpy as np
from .base import MetricBase
from ..registry import register_metric


@register_metric("root_mean_squared_error", aliases=("rmse",))
class RMSEMetric(MetricBase):
    def compute(self, forecast, obs, spatial=False, **kwargs):
        if "member" in forecast.dims:
            forecast = forecast.mean("member")
        if "tercile" in forecast.dims:
            raise ValueError(
                "rmse requires a continuous-valued forecast; got a forecast "
                "with a 'tercile' dim (tercile probabilities). Pass the "
                "deterministic ensemble forecast instead."
            )

        sq_err = (forecast - obs) ** 2
        rmse = np.sqrt(sq_err.mean("year"))

        if spatial:
            return rmse
        return float(rmse.mean())
