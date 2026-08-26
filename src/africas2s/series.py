"""Calibration and uncertainty for scalar forecast series.

Everything else in africas2s calibrates a *field*. But a forecast is often a
single number per year — a Niño3.4 value, a dipole-mode index, a region-averaged
rainfall total — and such a series needs the same two things a field does:
correction against what was observed, and an honest interval around the answer.

* :func:`quantile_map` maps a forecast series onto an observed distribution,
  using the same transfer function as the gridded ``qm`` downscaler
  (:mod:`africas2s.methods._qm_kernel`), so the two cannot disagree.
* :func:`error_bounds` turns a hindcast's realised errors into a confidence
  interval around a new forecast. No distributional assumption is required —
  the historical errors *are* the distribution — though a Gaussian fit is
  available when the sample is small.

Neither function knows what the series measures. A forecast index, a
station-mean rainfall total, and a basin-mean SST are all just series.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.stats import norm

from .methods._qm_kernel import empirical_map, parametric_map

__all__ = ["quantile_map", "error_bounds", "ErrorBounds"]


def _as_sample(values, name: str) -> np.ndarray:
    """Flatten a reference sample to a 1-D finite float array."""
    arr = np.asarray(
        values.values if isinstance(values, xr.DataArray) else values, dtype=float
    ).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError(f"{name} contains no finite values")
    return finite


def quantile_map(
    x,
    source,
    target,
    *,
    variant: str = "empirical",
    extrapolate: str = "clamp",
):
    """Map ``x`` from the distribution of ``source`` onto that of ``target``.

    The canonical use is bias-correcting a forecast index against observations:
    ``source`` is the model's hindcast series, ``target`` the observed series
    over the same (or a chosen baseline) period, and ``x`` the new forecast.

    Parameters
    ----------
    x : scalar, array or xr.DataArray
        The value(s) to map. A DataArray's dims and coords are preserved.
    source, target : array-like
        Reference samples. They need not be the same length — each is reduced to
        its own empirical CDF first. Non-finite entries are dropped.
    variant : {"empirical", "parametric"}
        ``"empirical"`` matches the sample CDFs directly. ``"parametric"``
        matches Gaussian fits, which is more stable on short records but
        assumes symmetry.
    extrapolate : {"clamp", "linear"}
        What to do with an ``x`` outside ``source``'s range. ``"clamp"`` returns
        the nearest value in ``target`` — safe, but it will silently truncate a
        forecast of a record-breaking event down to the strongest event in the
        training record. ``"linear"`` continues the transfer function's end
        slope. Ignored by the parametric variant, which extrapolates by
        construction.

    Returns
    -------
    Same type as ``x``.
    """
    source_sample = _as_sample(source, "source")
    target_sample = _as_sample(target, "target")

    if variant == "parametric":
        mapped_fn = lambda values: parametric_map(  # noqa: E731
            values,
            source_sample.mean(),
            source_sample.std(ddof=1),
            target_sample.mean(),
            target_sample.std(ddof=1),
        )
    elif variant == "empirical":
        source_sorted = np.sort(source_sample)
        target_sorted = np.sort(target_sample)
        mapped_fn = lambda values: empirical_map(  # noqa: E731
            values, source_sorted, target_sorted, extrapolate=extrapolate
        )
    else:
        raise ValueError(
            f"variant must be 'empirical' or 'parametric', got {variant!r}"
        )

    if isinstance(x, xr.DataArray):
        return xr.DataArray(
            np.reshape(mapped_fn(x.values.ravel()), x.shape),
            dims=x.dims, coords=x.coords, name=x.name, attrs=x.attrs,
        )
    return mapped_fn(x)


@dataclass(frozen=True)
class ErrorBounds:
    """A confidence interval built from a hindcast's realised errors."""

    lower: object
    upper: object
    level: float
    bias: object
    errors: xr.DataArray

    def __iter__(self):
        """So ``lower, upper = error_bounds(...)`` reads naturally."""
        return iter((self.lower, self.upper))


def error_bounds(
    hindcast_prediction: xr.DataArray,
    hindcast_obs: xr.DataArray,
    forecast,
    *,
    level: float = 0.8,
    dim: str = "year",
    method: str = "empirical",
) -> ErrorBounds:
    """A ``level``-confidence interval around ``forecast`` from historical errors.

    The error sample is ``hindcast_prediction - hindcast_obs``. Because a value
    that was observed is ``prediction - error``, the interval around a new
    forecast is the forecast minus the error distribution's tail quantiles —
    which also removes any mean bias the model carries, for free.

    Parameters
    ----------
    hindcast_prediction, hindcast_obs : xr.DataArray
        Paired over ``dim``. Any other dims broadcast, so a gridded or
        multi-model interval falls out of the same call.
    forecast : scalar or xr.DataArray
        The value to bracket. Must not carry ``dim``.
    level : float
        Central coverage, e.g. ``0.8`` for an 80% interval.
    method : {"empirical", "gaussian"}
        ``"empirical"`` reads the tail quantiles straight off the error sample
        and assumes nothing about its shape, but cannot resolve a tail finer
        than ``1 / n``. ``"gaussian"`` fits a normal to the errors, which
        extrapolates into the tails at the cost of assuming symmetry.

    Returns
    -------
    ErrorBounds
        With ``.lower``, ``.upper``, ``.bias`` and the ``.errors`` sample.
        Unpacks as ``(lower, upper)``.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie strictly between 0 and 1, got {level}")
    if dim not in hindcast_prediction.dims or dim not in hindcast_obs.dims:
        raise ValueError(
            f"both hindcast arrays must carry the sample dim {dim!r}; got "
            f"{tuple(hindcast_prediction.dims)} and {tuple(hindcast_obs.dims)}"
        )
    if isinstance(forecast, xr.DataArray) and dim in forecast.dims:
        raise ValueError(
            f"forecast must not carry the sample dim {dim!r} — it is a value to "
            "bracket, not part of the error sample."
        )

    errors = (hindcast_prediction - hindcast_obs).rename("forecast_error")
    tail = (1.0 - level) / 2.0
    bias = errors.mean(dim, skipna=True)

    if method == "empirical":
        n = int(errors.notnull().sum(dim).min())
        if n < 2:
            raise ValueError("need at least two paired hindcast years")
        high = errors.quantile(1.0 - tail, dim=dim, skipna=True).drop_vars("quantile")
        low = errors.quantile(tail, dim=dim, skipna=True).drop_vars("quantile")
    elif method == "gaussian":
        spread = errors.std(dim, ddof=1, skipna=True)
        z = norm.ppf(1.0 - tail)
        high, low = bias + z * spread, bias - z * spread
    else:
        raise ValueError(
            f"method must be 'empirical' or 'gaussian', got {method!r}"
        )

    # obs = prediction - error, so the upper bound on obs comes from the *lower*
    # error quantile. Getting this backwards silently inverts the interval.
    return ErrorBounds(
        lower=forecast - high,
        upper=forecast - low,
        level=level,
        bias=bias,
        errors=errors,
    )
