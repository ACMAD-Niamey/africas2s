"""Quantile Mapping (QM) bias correction + downscaling (§19.1).

Maps the forecast's distribution onto the observed distribution per grid cell,
then regrids the corrected coarse field up to the fine obs grid. Two variants:

- ``"empirical"`` (default): nonparametric CDF matching, ``F_obs^-1(F_gcm(x))``.
- ``"parametric"``: Gaussian CDF matching — z-score the forecast against the
  GCM climatology, then rescale by the obs climatology
  (``obs_mean + (x - gcm_mean) / gcm_std * obs_std``).

Like BCSD, QM corrects at the GCM grid (obs interpolated down to it) and then
interpolates the corrected field up to the obs grid. Unlike BCSD, QM applies no
spatial-disaggregation detail term — it is the pure bias-correction baseline.

Limitations (§19.1):
- Empirical QM cannot extrapolate beyond the training extrema; values outside
  the GCM/obs training support are clamped to the nearest observed quantile
  (``numpy.interp`` endpoint behaviour), not linearly extrapolated.
- Exact ties in a cell's GCM climatology make the empirical CDF non-strict;
  with continuous fields this is not hit in practice.
"""
import numpy as np
import xarray as xr

from ._qm_kernel import empirical_map, parametric_map
from .base import MethodBase
from ..registry import register_method


@register_method("qm")
class QuantileMappingMethod(MethodBase):
    def __init__(self, variant="empirical"):
        if variant not in ("empirical", "parametric"):
            raise ValueError(
                f"variant must be 'empirical' or 'parametric'; got {variant!r}."
            )
        self.variant = variant

    def fit(self, hindcast, obs, **kwargs):
        gcm_mean = hindcast.mean("member")
        # Bring obs down to the GCM grid so the per-cell CDFs are co-located.
        obs_coarse = obs.interp(lat=hindcast.lat, lon=hindcast.lon, method="linear")

        self.n_years_ = len(hindcast.year)
        self.gcm_coords_ = {"lat": hindcast.lat, "lon": hindcast.lon}
        self.obs_coords_ = {"lat": obs.lat, "lon": obs.lon}

        if self.variant == "empirical":
            self.gcm_sorted_ = np.sort(gcm_mean.values, axis=0)
            self.obs_sorted_ = np.sort(obs_coarse.values, axis=0)
        else:  # parametric (Gaussian)
            self.gcm_mean_ = gcm_mean.mean("year").values
            self.gcm_std_ = gcm_mean.std("year", ddof=1).values
            self.obs_mean_ = obs_coarse.mean("year").values
            self.obs_std_ = obs_coarse.std("year", ddof=1).values

    def predict(self, forecast, **kwargs):
        if "year" in forecast.dims and forecast.sizes.get("year") == 1:
            forecast = forecast.isel(year=0, drop=True)

        results = []
        for mi in range(len(forecast.member)):
            x = forecast.isel(member=mi).values  # (c_lat, c_lon)
            corrected = (
                self._empirical_map(x)
                if self.variant == "empirical"
                else self._parametric_map(x)
            )
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

    def _empirical_map(self, x):
        """F_obs^-1(F_gcm(x)) per cell, via the shared transfer function."""
        out = np.empty(x.shape, dtype=float)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                # Clamped tails: a bias corrector must not invent values the
                # observations have never shown (documented limitation).
                out[i, j] = empirical_map(
                    x[i, j], self.gcm_sorted_[:, i, j], self.obs_sorted_[:, i, j]
                )
        return out

    def _parametric_map(self, x):
        """Gaussian quantile matching: rescale the GCM anomaly into obs units."""
        return parametric_map(
            x, self.gcm_mean_, self.gcm_std_, self.obs_mean_, self.obs_std_
        )
