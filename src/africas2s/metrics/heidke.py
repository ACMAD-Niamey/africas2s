import numpy as np
import xarray as xr
from .base import MetricBase
from .rpss import _cpt_boundaries
from ..registry import register_metric


@register_metric("heidke_skill_score", aliases=("hss",))
class HSSMetric(MetricBase):
    """Heidke Skill Score for tercile categorical forecasts.

    Forecast input is tercile-probability shaped (year, tercile, lat, lon);
    HSS is a categorical metric, so the tercile dim is collapsed by argmax to
    a deterministic prediction. Obs is categorized using the same CPT-style
    tercile boundaries as RPSS (`_cpt_boundaries`).

    Two HSS conventions exist (per-cell averaged vs pooled contingency table).
    This implementation pools all (cell, year) pairs into a single contingency
    table when `spatial=False`, and computes per-cell HSS when `spatial=True`.
    """

    def compute(self, forecast, obs, spatial=False, **kwargs):
        obs_vals = obs.values  # (year, lat, lon)
        t33, t67 = _cpt_boundaries(obs_vals)
        obs_cat = np.where(t33 > obs_vals, 0, np.where(t67 > obs_vals, 1, 2))

        # Ensure forecast axes match obs (year, *spatial_dims) before argmax,
        # otherwise fixtures with shape (tercile, lat, lon, year) produce a
        # transposed fcst_cat that won't broadcast against obs_cat.
        spatial_dims = [d for d in obs.dims if d != "year"]
        fcst = forecast.transpose("year", "tercile", *spatial_dims)
        fcst_cat = fcst.argmax("tercile").values  # (year, *spatial_dims)

        nan_mask = np.isnan(obs_vals) | np.isnan(t33)[None, ...]

        if spatial:
            spatial_dims = [d for d in obs.dims if d != "year"]
            spatial_coords = {
                k: v for k, v in obs.coords.items()
                if k != "year" and set(obs[k].dims).issubset(set(spatial_dims))
            }
            valid = ~nan_mask
            n_eff = valid.sum(axis=0)  # (lat, lon)
            correct = ((fcst_cat == obs_cat) & valid).sum(axis=0).astype(float)
            expected = np.zeros_like(correct, dtype=float)
            for c in range(3):
                n_fcst = ((fcst_cat == c) & valid).sum(axis=0).astype(float)
                n_obs = ((obs_cat == c) & valid).sum(axis=0).astype(float)
                with np.errstate(invalid="ignore", divide="ignore"):
                    expected = expected + n_fcst * n_obs / np.where(n_eff > 0, n_eff, np.nan)
            denom = n_eff - expected
            with np.errstate(invalid="ignore", divide="ignore"):
                hss = np.where(denom > 0, (correct - expected) / denom, np.nan)
            return xr.DataArray(hss, dims=spatial_dims, coords=spatial_coords)

        # Pooled: flatten (cell, year) and compute one global HSS
        valid = ~nan_mask
        n_eff = int(valid.sum())
        if n_eff == 0:
            return float("nan")
        correct = float(((fcst_cat == obs_cat) & valid).sum())
        expected = 0.0
        for c in range(3):
            n_fcst = float(((fcst_cat == c) & valid).sum())
            n_obs = float(((obs_cat == c) & valid).sum())
            expected += n_fcst * n_obs / n_eff
        denom = n_eff - expected
        if denom <= 0:
            return float("nan")
        return float((correct - expected) / denom)
