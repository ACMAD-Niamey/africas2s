"""africas2s.climate.seasonal_reduce: per-year aggregation over selected calendar months."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from africas2s.climate import seasonal_reduce


def _monthly(years=range(2000, 2010), lat=(-2.0, 0.0, 2.0), lon=(34.0, 38.0)):
    time = pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-31", freq="MS")
    rng = np.random.default_rng(0)
    data = rng.random((len(time), len(lat), len(lon))) + 1
    return xr.DataArray(data, dims=("time", "lat", "lon"),
                        coords={"time": time, "lat": list(lat), "lon": list(lon)})


def test_matches_the_handrolled_core():
    da = _monthly()
    months = [10, 11, 12]
    got = seasonal_reduce(da, months, how="sum")
    # the exact expression the AGU consumers hand-rolled 8 times
    ref = da.sel(time=da["time.month"].isin(months)).groupby("time.year").sum("time")
    assert got.equals(ref)


def test_replaces_time_with_year_and_keeps_space():
    da = _monthly()
    out = seasonal_reduce(da, [7, 8, 9])
    assert "year" in out.dims and "time" not in out.dims
    assert out.sizes["lat"] == 3 and out.sizes["lon"] == 2
    assert out.sizes["year"] == 10


def test_how_mean_vs_sum():
    da = _monthly()
    s = seasonal_reduce(da, [1, 2, 3], how="sum")
    m = seasonal_reduce(da, [1, 2, 3], how="mean")
    assert np.allclose(s.values, m.values * 3)          # 3 months


def test_validates_months_and_how():
    da = _monthly()
    with pytest.raises(ValueError, match="1..12"):
        seasonal_reduce(da, [0, 13])
    with pytest.raises(ValueError, match="1..12"):
        seasonal_reduce(da, [])
    with pytest.raises(ValueError, match="how must be"):
        seasonal_reduce(da, [6], how="median")


def test_requires_time_dim():
    da = _monthly().mean("time")
    with pytest.raises(ValueError, match="not found"):
        seasonal_reduce(da, [6])
