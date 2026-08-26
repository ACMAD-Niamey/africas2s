"""Shared lat/lon dimension resolution for gridded DataArrays.

A single place to recognize the spatial-axis naming conventions africas2s
accepts (lat/latitude/Y/y and lon/longitude/X/x), so every entry point —
indices, ensemble regression, logistic calibration, plotting — agrees on which
dims are spatial. Adding a new alias here reaches all of them at once.
"""
from __future__ import annotations

import xarray as xr

_LAT_ALIASES = ("lat", "latitude", "Y", "y")
_LON_ALIASES = ("lon", "longitude", "X", "x")


def spatial_dims(da: xr.DataArray, *, context: str = "data") -> tuple[str, str]:
    """Return the ``(lat_dim, lon_dim)`` names on ``da``.

    Resolves the common aliases (lat/latitude/Y/y and lon/longitude/X/x).
    Raises ``ValueError`` naming ``context`` when either axis can't be found.
    """
    lat = next((d for d in _LAT_ALIASES if d in da.dims), None)
    lon = next((d for d in _LON_ALIASES if d in da.dims), None)
    if lat is None or lon is None:
        raise ValueError(
            f"{context} could not find lat/lon dims on data with dims "
            f"{tuple(da.dims)}; expected one of {_LAT_ALIASES} and "
            f"{_LON_ALIASES}."
        )
    return lat, lon
