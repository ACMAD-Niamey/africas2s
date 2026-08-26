import numpy as np
import xarray as xr
from scipy import interpolate, stats
from .base import MethodBase
from ..registry import register_method

@register_method("bcsd")
class BCSDMethod(MethodBase):
    def fit(self, hindcast, obs, **kwargs):
        self.obs_clim_ = obs.mean("year")

        gcm_mean = hindcast.mean("member")
        obs_coarse = obs.interp(lat=hindcast.lat, lon=hindcast.lon, method="linear")

        self.gcm_sorted_ = np.sort(gcm_mean.values, axis=0)
        self.obs_coarse_sorted_ = np.sort(obs_coarse.values, axis=0)

        self.gcm_coords_ = {"lat": hindcast.lat, "lon": hindcast.lon}
        self.obs_coords_ = {"lat": obs.lat, "lon": obs.lon}
        self.n_years_ = len(hindcast.year)

    def predict(self, forecast, **kwargs):
        # Squeeze out a singleton year dim so the per-member slice is always
        # (lat, lon).  The CV loop passes a year-selected slice (year dim
        # present) while some callers may already have squeezed it.
        if "year" in forecast.dims and forecast.sizes["year"] == 1:
            forecast = forecast.isel(year=0, drop=True)
        result_members = []
        for m in range(len(forecast.member)):
            fcst_m = forecast.isel(member=m).values

            corrected = np.zeros_like(fcst_m)
            for i in range(fcst_m.shape[0]):
                for j in range(fcst_m.shape[1]):
                    gcm_vals = self.gcm_sorted_[:, i, j]
                    obs_vals = self.obs_coarse_sorted_[:, i, j]
                    quantile = np.searchsorted(gcm_vals, fcst_m[i, j]) / self.n_years_
                    quantile = np.clip(quantile, 0.01, 0.99)
                    idx = quantile * (self.n_years_ - 1)
                    lo = int(np.floor(idx))
                    hi = min(lo + 1, self.n_years_ - 1)
                    frac = idx - lo
                    corrected[i, j] = obs_vals[lo] * (1 - frac) + obs_vals[hi] * frac

            corrected_da = xr.DataArray(
                corrected, dims=["lat", "lon"], coords=self.gcm_coords_
            )
            fine = corrected_da.interp(
                lat=self.obs_coords_["lat"], lon=self.obs_coords_["lon"], method="linear"
            )

            obs_clim_coarse = self.obs_clim_.interp(
                lat=self.gcm_coords_["lat"], lon=self.gcm_coords_["lon"], method="linear"
            )
            obs_clim_back = obs_clim_coarse.interp(
                lat=self.obs_coords_["lat"], lon=self.obs_coords_["lon"], method="linear"
            )
            spatial_detail = self.obs_clim_ - obs_clim_back
            fine_result = fine + spatial_detail
            result_members.append(fine_result.values)

        return xr.DataArray(
            np.stack(result_members),
            dims=["member", "lat", "lon"],
            coords={
                "member": forecast.member,
                "lat": self.obs_coords_["lat"],
                "lon": self.obs_coords_["lon"],
            },
        )
