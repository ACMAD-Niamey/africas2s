import numpy as np
from .base import MetricBase
from ..registry import register_metric


@register_metric("mean_square_skill_score", aliases=("msss",))
class MSSSMetric(MetricBase):
    """Mean Square Skill Score, 1 - MSE/Var(O). Higher is better; 1 is perfect,
    0 is no better than the climatological mean, negative is worse than climatology.

    Pooled (spatial=False): spatially average the numerator (MSE) and denominator
    (Var(O)) separately, then divide -- the area-aggregated skill-score convention.
    spatial=True: per-gridpoint 1 - MSE/Var(O). NaN-aware.
    """
    def compute(self, forecast, obs, spatial=False, **kwargs):
        if "member" in forecast.dims:
            forecast = forecast.mean("member")
        if "tercile" in forecast.dims:
            raise ValueError(
                "msss requires a continuous-valued forecast; got a forecast with "
                "a 'tercile' dim. Pass the deterministic forecast instead."
            )
        mse = ((forecast - obs) ** 2).mean("year")            # (lat, lon)
        varo = ((obs - obs.mean("year")) ** 2).mean("year")   # (lat, lon)
        if spatial:
            return 1.0 - mse / varo.where(varo > 0)
        den = float(varo.mean())
        if not np.isfinite(den) or den == 0.0:
            return float("nan")
        return 1.0 - float(mse.mean()) / den
