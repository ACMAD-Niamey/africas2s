"""Delta scaling (`delta`) — §19.7, Hay et al. 2000.

The simplest downscaling baseline: take the GCM forecast's anomaly from the GCM
historical climatology, regrid it to the fine obs grid, and add it to the
observed climatology:

    prediction = obs_clim + interp(forecast - gcm_hist_clim)

It assumes the GCM gets *changes* right even where its absolute values are
biased. Trivially simple — useful as a sanity-check reference column, not a
serious competitor.
"""
import xarray as xr

from .base import MethodBase
from ..registry import register_method


@register_method("delta")
class DeltaScalingMethod(MethodBase):
    def fit(self, hindcast, obs, **kwargs):
        reduce_dims = [d for d in ("member", "year") if d in hindcast.dims]
        self.gcm_hist_clim_ = hindcast.mean(reduce_dims)   # (lat, lon) coarse
        self.obs_clim_ = obs.mean("year")                  # (lat, lon) fine
        self.gcm_coords_ = {"lat": hindcast.lat, "lon": hindcast.lon}
        self.obs_coords_ = {"lat": obs.lat, "lon": obs.lon}

    def predict(self, forecast, **kwargs):
        if "year" in forecast.dims and forecast.sizes.get("year") == 1:
            forecast = forecast.isel(year=0, drop=True)

        anomaly = forecast - self.gcm_hist_clim_           # (member, coarse)
        anomaly_fine = anomaly.interp(
            lat=self.obs_coords_["lat"],
            lon=self.obs_coords_["lon"],
            method="linear",
        )
        result = anomaly_fine + self.obs_clim_             # (member, fine)
        return result.transpose("member", "lat", "lon")
