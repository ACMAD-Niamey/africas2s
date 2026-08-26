"""The quantile-mapping transfer function, independent of what it is applied to.

Quantile mapping is one piece of arithmetic — read a value's quantile off a
source sample, return the value at that quantile of a target sample — and
africas2s applies it in two places: per grid cell in
:class:`~africas2s.methods.qm.QuantileMappingMethod`, and to a 1-D index series
in :mod:`africas2s.series`. Both call this module, so the gridded and series
paths cannot drift apart.

Two departures from a naive implementation, both deliberate:

* **Unequal sample sizes are allowed.** Source and target are reduced to their
  own plotting positions before the quantiles are matched, so a 25-year
  forecast record can be mapped onto a 45-year observed record. When the two
  are the same length this is numerically identical to the paired-sort form.

* **The tails have a policy.** ``numpy.interp`` clamps out-of-support inputs to
  the nearest reference value. That is the right default for a bias corrector
  (it never invents a value the observations have never shown), and the wrong
  one for forecasting an extreme: a record-strength El Niño mapped through a
  clamped transfer function silently comes back as merely the strongest thing
  in the training record. ``extrapolate="linear"`` continues the transfer
  function's end slopes instead.
"""
from __future__ import annotations

import numpy as np

__all__ = ["plotting_positions", "empirical_map", "parametric_map"]

_EXTRAPOLATE = ("clamp", "linear")


def plotting_positions(n: int) -> np.ndarray:
    """Midpoint plotting positions ``(i + 0.5) / n`` for a sample of size ``n``.

    Midpoints (rather than ``i / (n - 1)``) keep the empirical CDF strictly
    inside ``(0, 1)``, so neither endpoint is asserted to be the distribution's
    true minimum or maximum.
    """
    return (np.arange(n) + 0.5) / n


def _linear_tails(x, y, source_sorted, target_sorted):
    """Replace clamped tail values with an end-slope linear extrapolation.

    Falls back to the clamped value where the two extreme reference points are
    tied (a zero-width source interval gives no slope to extend).
    """
    x = np.asarray(x, dtype=float)
    y = np.array(y, dtype=float, copy=True)

    for below, i0, i1 in ((True, 0, 1), (False, -1, -2)):
        if source_sorted.size < 2:
            continue
        mask = (x < source_sorted[0]) if below else (x > source_sorted[-1])
        if not np.any(mask):
            continue
        dx = source_sorted[i0] - source_sorted[i1]
        if dx == 0:
            continue  # tied endpoints: no slope, keep the clamp
        slope = (target_sorted[i0] - target_sorted[i1]) / dx
        y[mask] = target_sorted[i0] + slope * (x[mask] - source_sorted[i0])
    return y


def empirical_map(
    x,
    source_sorted: np.ndarray,
    target_sorted: np.ndarray,
    *,
    extrapolate: str = "clamp",
) -> np.ndarray:
    """Map ``x`` from the ``source`` distribution onto the ``target`` one.

    ``F_target^-1(F_source(x))``, with both CDFs estimated by midpoint plotting
    positions on the (already sorted) reference samples.

    Non-finite values anywhere in ``x``, ``source_sorted`` or ``target_sorted``
    produce NaN rather than a silently wrong mapping: a reference sample with a
    hole has no well-defined CDF.
    """
    if extrapolate not in _EXTRAPOLATE:
        raise ValueError(
            f"extrapolate must be one of {_EXTRAPOLATE}, got {extrapolate!r}"
        )
    scalar = np.ndim(x) == 0
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))

    if source_sorted.size == 0 or target_sorted.size == 0:
        raise ValueError("empirical_map needs non-empty source and target samples")

    if not (np.all(np.isfinite(source_sorted)) and np.all(np.isfinite(target_sorted))):
        out = np.full(x_arr.shape, np.nan)
        return out.item() if scalar else out

    pp_source = plotting_positions(source_sorted.size)
    pp_target = plotting_positions(target_sorted.size)

    quantile = np.interp(x_arr, source_sorted, pp_source)
    mapped = np.interp(quantile, pp_target, target_sorted)

    if extrapolate == "linear":
        mapped = _linear_tails(x_arr, mapped, source_sorted, target_sorted)

    mapped = np.where(np.isfinite(x_arr), mapped, np.nan)
    return mapped.item() if scalar else mapped


def parametric_map(x, source_mean, source_std, target_mean, target_std):
    """Gaussian quantile matching: rescale the source anomaly into target units.

    ``target_mean + (x - source_mean) / source_std * target_std``. A degenerate
    source spread (std below 1e-12) is treated as unit spread, which passes the
    anomaly through unscaled rather than dividing by zero.
    """
    std = np.where(np.asarray(source_std) < 1e-12, 1.0, source_std)
    return target_mean + ((x - source_mean) / std) * target_std
