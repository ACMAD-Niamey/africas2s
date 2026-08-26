"""Detrended Quantile Mapping (DQM) — §19.2, Cannon et al. 2015.

Shares the QM scaffold (#42) but removes a per-cell linear trend *before* fitting
the CDFs, quantile-maps the detrended anomalies, then re-adds the GCM trend so
the model's climate-change signal is preserved (the defining feature of DQM vs
plain QM).

Forecast-year caveat (V1). DQM re-adds the GCM trend evaluated at the forecast
time. ``predict()`` receives no year, so V1 re-adds the trend at the **most
recent hindcast year** — correct for the common ``downscale()`` usage where the
forecast is the latest/"current" season. An explicit forecast-year hook can be
added later if future-projection use cases need it.

Pitfall (§19.2). A trend estimated over a ~24-year hindcast may be barely
significant; noisy detrending can degrade rather than improve skill. Use with
that caveat in mind.
"""
import numpy as np
import xarray as xr

from ..registry import register_method
from .qm import QuantileMappingMethod


@register_method("dqm")
class DetrendedQuantileMappingMethod(QuantileMappingMethod):
    def fit(self, hindcast, obs, **kwargs):
        gcm_mean = hindcast.mean("member")
        obs_coarse = obs.interp(lat=hindcast.lat, lon=hindcast.lon, method="linear")

        years = hindcast.year.values.astype(float)
        t = years - years.mean()              # centred time axis (so Σt = 0)
        denom = float((t ** 2).sum()) or 1.0
        # Re-add the GCM trend at the latest hindcast year (see module docstring).
        self.t_forecast_ = float(t[-1])

        # Per-cell OLS slope on the centred time axis: slope = Σ t·y / Σ t².
        self.gcm_slope_ = np.tensordot(t, gcm_mean.values, axes=(0, 0)) / denom
        obs_slope = np.tensordot(t, obs_coarse.values, axes=(0, 0)) / denom

        # Detrend (remove the trend, keep the series mean — t is centred).
        gcm_dt = gcm_mean.values - self.gcm_slope_[None] * t[:, None, None]
        obs_dt = obs_coarse.values - obs_slope[None] * t[:, None, None]

        self.n_years_ = len(years)
        self.gcm_coords_ = {"lat": hindcast.lat, "lon": hindcast.lon}
        self.obs_coords_ = {"lat": obs.lat, "lon": obs.lon}

        if self.variant == "empirical":
            self.gcm_sorted_ = np.sort(gcm_dt, axis=0)
            self.obs_sorted_ = np.sort(obs_dt, axis=0)
        else:  # parametric (Gaussian) on the detrended series
            self.gcm_mean_ = gcm_dt.mean(axis=0)
            self.gcm_std_ = gcm_dt.std(axis=0, ddof=1)
            self.obs_mean_ = obs_dt.mean(axis=0)
            self.obs_std_ = obs_dt.std(axis=0, ddof=1)

    def predict(self, forecast, **kwargs):
        if "year" in forecast.dims and forecast.sizes.get("year") == 1:
            forecast = forecast.isel(year=0, drop=True)

        trend = self.gcm_slope_ * self.t_forecast_   # (lat, lon) GCM trend offset
        results = []
        for mi in range(len(forecast.member)):
            x = forecast.isel(member=mi).values - trend            # detrend forecast
            corrected = (
                self._empirical_map(x)
                if self.variant == "empirical"
                else self._parametric_map(x)
            )
            corrected = corrected + trend                          # re-add GCM trend
            corrected_da = xr.DataArray(
                corrected, dims=["lat", "lon"], coords=self.gcm_coords_
            )
            fine = corrected_da.interp(
                lat=self.obs_coords_["lat"],
                lon=self.obs_coords_["lon"],
                method="linear",
            )
            results.append(fine.values)

        return xr.DataArray(
            np.stack(results),
            dims=["member", "lat", "lon"],
            coords={"member": forecast.member, **self.obs_coords_},
        )
