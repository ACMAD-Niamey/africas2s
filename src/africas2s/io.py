"""Public IO helpers for tercile-probability forecasts.

``write_terciles`` and ``tercile_mae`` are verbatim lifts of consumer code that
already existed in the ICPAC MAM-replication pipeline
(``rosetta_africas2s/run_pipeline.py::write_tercile_netcdf`` and
``rosetta_africas2s/metrics.py::load_probs`` / ``metrics``). They are promoted
here so downstream notebooks/scripts can write and score tercile forecasts
using only the public ``africas2s`` API, without reaching into a specific
consumer repo's internals.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def write_terciles(probs: xr.DataArray, path, *, title: str, method: str = "") -> None:
    """Write tercile probabilities as a below/normal/above percent NetCDF.

    ``probs`` is a ``(tercile, lat, lon)``-ish DataArray (any order; it is
    transposed) with fractional probabilities (0-1) for tercile categories
    ``0=below, 1=normal, 2=above``. Values are renormalized across ``tercile``
    (divided by their sum where the sum is finite and > 0; left as NaN
    otherwise) and written out as percentages (0-100).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    probs = probs.transpose("tercile", "lat", "lon")
    total = probs.sum("tercile", skipna=False)
    valid = np.isfinite(total) & (total > 0)
    probs = xr.where(valid, probs / total, np.nan)

    ds = xr.Dataset(
        {
            "below": (("lat", "lon"), (probs.sel(tercile=0) * 100.0).values),
            "normal": (("lat", "lon"), (probs.sel(tercile=1) * 100.0).values),
            "above": (("lat", "lon"), (probs.sel(tercile=2) * 100.0).values),
        },
        coords={"lat": probs.lat.values, "lon": probs.lon.values},
        attrs={"title": title, "method": method},
    )
    enc = {name: {"_FillValue": -9999.0, "dtype": "float32"} for name in ds.data_vars}
    ds.to_netcdf(path, encoding=enc)


def _load_probs(path) -> xr.DataArray:
    """Load a below/normal/above-percent NetCDF as a (tercile, lat, lon) DataArray."""
    ds = xr.open_dataset(path)
    return xr.concat(
        [ds["below"], ds["normal"], ds["above"]],
        dim=xr.IndexVariable("tercile", [0, 1, 2]),
    )


def tercile_mae(probs: xr.DataArray, reference) -> float:
    """Mean absolute error (in percentage points) between candidate and reference
    tercile-probability forecasts.

    ``probs`` is a ``(tercile, lat, lon)`` DataArray of fractional probabilities
    (0-1). ``reference`` is either a path to a below/normal/above-percent
    NetCDF (as written by :func:`write_terciles`) or an already-loaded
    ``(tercile, lat, lon)`` DataArray in percent (0-100).

    If the candidate and reference grids differ, the reference is regridded
    onto the candidate's lat/lon via linear interpolation before an
    inner-join alignment. Only grid cells where all three tercile categories
    are finite on both sides contribute to the mean.
    """
    if isinstance(reference, (str, Path)):
        reference_percent = _load_probs(reference)
    else:
        reference_percent = reference

    cand_percent = probs * 100.0
    if (
        "lat" in cand_percent.coords
        and "lon" in cand_percent.coords
        and (
            cand_percent.sizes.get("lat") != reference_percent.sizes.get("lat")
            or cand_percent.sizes.get("lon") != reference_percent.sizes.get("lon")
            or not np.array_equal(cand_percent.lat.values, reference_percent.lat.values)
            or not np.array_equal(cand_percent.lon.values, reference_percent.lon.values)
        )
    ):
        reference_percent = reference_percent.interp(
            lat=cand_percent.lat.values, lon=cand_percent.lon.values
        )
    cand_percent, reference_percent = xr.align(cand_percent, reference_percent, join="inner")
    valid = np.isfinite(cand_percent).all("tercile") & np.isfinite(reference_percent).all("tercile")
    c = cand_percent.where(valid)
    r = reference_percent.where(valid)
    return float(abs(c - r).mean())
