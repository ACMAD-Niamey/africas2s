"""Continuous Ranked Probability Score / Skill Score for a parametric Gaussian forecast."""
import numpy as np
from scipy import stats
from .base import MetricBase
from ..registry import register_metric


def crps_normal(mu, sigma, obs):
    """Closed-form CRPS of N(mu, sigma) for obs (Gneiting et al. 2005). Broadcasts."""
    mu, sigma, obs = np.broadcast_arrays(np.asarray(mu, float),
                                         np.asarray(sigma, float), np.asarray(obs, float))
    sigma = np.maximum(sigma, 1e-12)
    z = (obs - mu) / sigma
    return sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1.0 / np.sqrt(np.pi))


def crps_climatology(obs, clim_std):
    return crps_normal(np.zeros_like(np.asarray(obs, float)), clim_std, obs)


def crpss(crps_forecast_mean, crps_ref_mean):
    if crps_ref_mean == 0:
        return float("nan")
    return 1.0 - float(crps_forecast_mean) / float(crps_ref_mean)


@register_metric("continuous_ranked_probability_skill_score", aliases=("crpss",))
class CRPSSMetric(MetricBase):
    """Score a parametric Gaussian forecast (an xr.Dataset with `mu` and `sigma` over
    (year[,lat,lon]) anomalies) against the climatological N(0, std(obs)) reference."""
    def compute(self, forecast, obs, spatial=False, **kwargs):
        if not (hasattr(forecast, "data_vars") and "mu" in forecast and "sigma" in forecast):
            raise ValueError(
                "crpss expects a Gaussian forecast as an xr.Dataset with 'mu' and 'sigma'."
            )
        mu, sigma = forecast["mu"], forecast["sigma"]
        crps_f = crps_normal(mu.values, sigma.values, obs.values)
        clim_std = float(obs.std())
        crps_r = crps_normal(np.zeros_like(obs.values), clim_std, obs.values)
        import xarray as xr
        cf = xr.DataArray(crps_f, dims=obs.dims, coords=obs.coords)
        cr = xr.DataArray(crps_r, dims=obs.dims, coords=obs.coords)
        if spatial:
            return 1.0 - cf.mean("year") / cr.mean("year")
        return crpss(float(cf.mean()), float(cr.mean()))
