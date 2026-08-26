"""Unit tests for the shared lat/lon dimension resolver (africas2s._spatial)."""
import numpy as np
import pytest
import xarray as xr

from africas2s._spatial import spatial_dims


def _da(lat_name, lon_name):
    return xr.DataArray(
        np.zeros((2, 3)),
        dims=[lat_name, lon_name],
        coords={lat_name: [0, 1], lon_name: [0, 1, 2]},
    )


@pytest.mark.parametrize("lat_name", ["lat", "latitude", "Y", "y"])
@pytest.mark.parametrize("lon_name", ["lon", "longitude", "X", "x"])
def test_resolves_all_supported_aliases(lat_name, lon_name):
    assert spatial_dims(_da(lat_name, lon_name)) == (lat_name, lon_name)


def test_raises_when_dims_missing():
    da = xr.DataArray(np.zeros((2, 3)), dims=["time", "ensemble"])
    with pytest.raises(ValueError, match="could not find lat/lon dims"):
        spatial_dims(da)


def test_context_appears_in_error():
    da = xr.DataArray(np.zeros(2), dims=["time"])
    with pytest.raises(ValueError, match="my_caller"):
        spatial_dims(da, context="my_caller")
