"""Scoped suppression of numpy's benign "All-NaN slice encountered" warning.

AfricaS2S computes per-cell statistics (tercile boundaries via ``xarray.quantile``,
climatologies via ``nanmean``, logit labels via ``nanpercentile``) over gridded
fields whose masked cells — ocean, out-of-region, dry-masked — are legitimately
all-NaN along the reduction axis. numpy warns "All-NaN slice encountered" on those
slices even though NaN-in / NaN-out is exactly the intended result (the cell has
no forecast and is masked downstream).

Decorate a public entry point with :func:`quiet_all_nan_slices` to silence just
that one message for the duration of the call. It uses ``catch_warnings`` so the
process-wide warning filters are restored on exit — no global state is mutated,
and only this specific benign message is affected.
"""
from __future__ import annotations

import functools
import warnings


def quiet_all_nan_slices(fn):
    """Wrap ``fn`` so numpy's "All-NaN slice encountered" RuntimeWarning is
    suppressed while it runs (masked grid cells are expected to be all-NaN)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "All-NaN slice encountered", RuntimeWarning)
            return fn(*args, **kwargs)
    return wrapper
