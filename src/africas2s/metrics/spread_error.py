"""Spread-error diagnostic (Fortin et al. 2014).

Compares ensemble spread (std across members) to ensemble error
(|ensemble mean - obs|). A well-calibrated ensemble has spread ≈ error.

Two registered metrics summarise the relationship:
- ``spread_error_ratio``: mean(spread) / mean(error); ≈ 1 = calibrated,
  < 1 = underdispersed, > 1 = overdispersed.
- ``spread_error_correlation``: Pearson r between per-year spread and
  per-year error; > 0 = spread tracks error.

A non-registry helper ``spread_error_diagnostics`` exposes the raw
(spread, error) pairs for plotting.

Input contract: ``forecast`` must carry a ``member`` dim. Probabilistic
methods that sample (CorrDiff, EMOS, U-Net + MC dropout) should pack
their samples onto a ``member`` dim before calling.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import xarray as xr

from .base import MetricBase
from ..registry import register_metric


def _spread_and_error(forecast, obs, *, spatial):
    """Compute spread and error arrays.

    Returns (spread, error). With ``spatial=False`` both have dims
    ``(year,)``; with ``spatial=True`` both have dims ``(year, lat, lon)``.
    """
    if "member" not in forecast.dims:
        raise ValueError(
            f"spread_error requires a 'member' dim; got dims={tuple(forecast.dims)}"
        )
    spread = forecast.std("member")
    error = np.abs(forecast.mean("member") - obs)
    if not spatial:
        # Collapse any non-year dim (typically lat/lon). The spec names
        # ("lat", "lon") explicitly; doing it dynamically keeps the kernel
        # working on forecasts whose spatial axes are named differently.
        # `member` is already gone after the std/mean calls above.
        non_year = [d for d in spread.dims if d != "year"]
        if non_year:
            spread = spread.mean(non_year)
            error = error.mean(non_year)
    return spread, error


@register_metric("spread_error_ratio")
class SpreadErrorRatioMetric(MetricBase):
    """mean(spread) / mean(error). ≈ 1 = calibrated."""

    def compute(self, forecast, obs, spatial=False, **kwargs):
        spread, error = _spread_and_error(forecast, obs, spatial=spatial)
        if spatial:
            return spread.mean("year") / error.mean("year")
        return float(spread.mean() / error.mean())


@register_metric("spread_error_correlation")
class SpreadErrorCorrelationMetric(MetricBase):
    """Pearson r between per-year spread and per-year error.

    >0 means spread tracks error (a desirable property of a calibrated
    ensemble). Returns NaN with a warning if fewer than 3 years are
    available.
    """

    def compute(self, forecast, obs, spatial=False, **kwargs):
        spread, error = _spread_and_error(forecast, obs, spatial=spatial)
        if spread.sizes.get("year", 0) < 3:
            warnings.warn(
                "spread_error_correlation needs at least 3 years; returning NaN",
                RuntimeWarning,
                stacklevel=2,
            )
            if spatial:
                # .mean("year") handles n_year=0 gracefully (returns NaN-filled).
                return xr.full_like(spread.mean("year"), np.nan, dtype=float)
            return float("nan")
        r = xr.corr(spread, error, dim="year")
        if spatial:
            return r
        return float(r)


@dataclass
class SpreadErrorDiagnostics:
    """Raw per-year (or per-grid) spread and error arrays.

    Intended for plotting / exploratory analysis. The two registered
    metrics summarise this into scalars.
    """
    spread: xr.DataArray
    error: xr.DataArray


def spread_error_diagnostics(forecast, obs, *, spatial=False):
    """Compute the raw (spread, error) pairs underlying the registered metrics.

    Parameters
    ----------
    forecast : xr.DataArray
        Must carry a ``member`` dim. Other dims must include ``year``.
    obs : xr.DataArray
        Observation aligned with ``forecast`` on every non-``member`` dim.
    spatial : bool, default False
        If False, returned arrays have dims ``(year,)`` (spatial mean).
        If True, returned arrays retain ``(year, lat, lon)``.
    """
    spread, error = _spread_and_error(forecast, obs, spatial=spatial)
    return SpreadErrorDiagnostics(spread=spread, error=error)
