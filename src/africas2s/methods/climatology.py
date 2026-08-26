"""Climatology baseline downscaling method.

Predicts the per-cell long-term mean of the observation climatology,
tiled across the forecast's ensemble members. Ignores the GCM forecast
values entirely — this is the no-skill reference column on the S2S
testbed comparison panel.
"""

import xarray as xr

from .base import MethodBase
from ..registry import register_method


@register_method("climatology")
class ClimatologyMethod(MethodBase):
    """No-skill reference baseline: predicts obs-grid climatology for every member."""

    def __init__(self):
        pass

    def fit(self, hindcast, obs, **kwargs):
        self.climatology_ = obs.mean("year")
        self.obs_coords_ = {"lat": obs.lat, "lon": obs.lon}

    def predict(self, forecast, **kwargs):
        if "year" in forecast.dims and forecast.sizes["year"] == 1:
            forecast = forecast.isel(year=0, drop=True)

        import numpy as np

        n_members = forecast.sizes["member"]
        tiled = np.broadcast_to(
            self.climatology_.values, (n_members, *self.climatology_.values.shape)
        ).copy()

        return xr.DataArray(
            tiled,
            dims=["member", "lat", "lon"],
            coords={
                "member": forecast.member,
                "lat": self.obs_coords_["lat"],
                "lon": self.obs_coords_["lon"],
            },
        )
