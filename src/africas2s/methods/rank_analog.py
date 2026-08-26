"""Rank-analog (empirical-quantile) downscaling method.

Adapted from `alecjong-lab/ECMWF-S2S4AFRICA::rank_upscale_and_align`. The
algorithm ranks a forecast within the historical hindcast climatology per
cell, replicates each coarse cell into an ``upscale_factor × upscale_factor``
block at fine resolution, crops to the obs grid, smooths with grey-closing
plus a Gaussian filter, then uses the smoothed integer rank field as an
index into the per-cell sorted observation climatology.

The method operates on a single timestep at a time, matching africas2s's
``MethodBase`` interface (year/member/lat/lon hindcast, year/lat/lon obs,
member/lat/lon forecast). Multi-lead-time orchestration belongs to the
caller — see the S2S downscaling testbed for an example.
"""

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, grey_closing

from .base import MethodBase
from ..registry import register_method


@register_method("rank-analog")
class RankAnalogMethod(MethodBase):
    """Rank-analog quantile-mapping downscaler.

    Parameters
    ----------
    closing_size : int, default 50
        Window size for ``scipy.ndimage.grey_closing`` applied to the
        upscaled rank field. Fills low-precip pockets from surrounding cells
        before the Gaussian smooth.
    gaussian_sigma : float, default 1.5
        Standard deviation for the post-closing Gaussian filter.
    upscale_factor : int or None, default None
        Integer factor for the nearest-neighbour rank upscale. If ``None``,
        derived at fit time from the ratio of obs grid size to hindcast grid
        size (rounded, minimum 1).
    """

    def __init__(self, closing_size=50, gaussian_sigma=1.5, upscale_factor=None):
        self.closing_size = closing_size
        self.gaussian_sigma = gaussian_sigma
        self.upscale_factor = upscale_factor

    def fit(self, hindcast, obs, **kwargs):
        if "member" in hindcast.dims:
            hindcast_mean = hindcast.mean("member")
        else:
            hindcast_mean = hindcast

        sorted_arr = np.sort(obs.values, axis=obs.dims.index("year"))
        obs_sorted = xr.DataArray(
            sorted_arr,
            dims=obs.dims,
            coords={d: obs.coords[d] for d in obs.dims if d in obs.coords},
        )

        self.hindcast_mean_ = hindcast_mean
        self.obs_sorted_ = obs_sorted
        self.gcm_coords_ = {"lat": hindcast.lat, "lon": hindcast.lon}
        self.obs_coords_ = {"lat": obs.lat, "lon": obs.lon}
        self.n_years_ = int(hindcast_mean.sizes["year"])

        if self.upscale_factor is None:
            # Ceiling, not round: the upscaled rank field gets cropped to
            # the obs grid, so we need the upscale to AT LEAST cover the obs
            # extent. round() can produce a factor that's one short for
            # non-integer ratios (e.g. obs=43 / hindcast=7 = 6.14 → round=6,
            # but 7×6=42 < 43; ceil=7 → 7×7=49 → crop to 43).
            import math
            self.upscale_factor_ = max(
                math.ceil(len(obs.lat) / len(hindcast.lat)),
                math.ceil(len(obs.lon) / len(hindcast.lon)),
                1,
            )
        else:
            self.upscale_factor_ = int(self.upscale_factor)

    def predict(self, forecast, **kwargs):
        if "year" in forecast.dims and forecast.sizes["year"] == 1:
            forecast = forecast.isel(year=0, drop=True)

        hist_sorted = np.sort(
            self.hindcast_mean_.values,
            axis=self.hindcast_mean_.dims.index("year"),
        )

        target_h = len(self.obs_coords_["lat"])
        target_w = len(self.obs_coords_["lon"])
        upscale = self.upscale_factor_

        result_members = []
        for m_idx in range(forecast.sizes["member"]):
            fcst = forecast.isel(member=m_idx).values  # (coarse_lat, coarse_lon)

            ranks = np.empty(fcst.shape, dtype=np.intp)
            for i in range(fcst.shape[0]):
                for j in range(fcst.shape[1]):
                    ranks[i, j] = np.searchsorted(hist_sorted[:, i, j], fcst[i, j])
            ranks = np.clip(ranks, 0, self.n_years_ - 1)

            upscaled = np.repeat(np.repeat(ranks, upscale, axis=0), upscale, axis=1)
            aligned = upscaled[:target_h, :target_w]

            if aligned.shape != (target_h, target_w):
                raise ValueError(
                    f"Upscaled rank field {aligned.shape} cannot align to obs "
                    f"grid ({target_h}, {target_w}). Increase upscale_factor "
                    f"or check that hindcast and obs grids overlap."
                )

            smoothed = grey_closing(
                aligned, size=(self.closing_size, self.closing_size)
            )
            smoothed = gaussian_filter(
                smoothed.astype(np.float32), sigma=self.gaussian_sigma
            )
            idx_field = np.clip(
                smoothed.astype(np.intp), 0, self.n_years_ - 1
            )

            ii, jj = np.indices(idx_field.shape)
            member_result = self.obs_sorted_.values[idx_field, ii, jj]
            result_members.append(member_result)

        return xr.DataArray(
            np.stack(result_members),
            dims=["member", "lat", "lon"],
            coords={
                "member": forecast.member,
                "lat": self.obs_coords_["lat"],
                "lon": self.obs_coords_["lon"],
            },
        )
