"""Tests for the public IO/plotting helpers added in tasks B2+B3+B4.

- B2: africas2s.write_terciles  (verbatim lift of run_pipeline.write_tercile_netcdf)
- B3: africas2s.tercile_mae     (verbatim lift of metrics.load_probs + metrics.metrics
                                  avg_probability_mae branch)
- B4: africas2s.plot_terciles   (re-export of plotting.forecasts.plot_tercile_forecast)
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import africas2s


# ---------------------------------------------------------------------------
# B2: write_terciles
# ---------------------------------------------------------------------------


def test_write_terciles_roundtrip(tmp_path):
    probs = xr.DataArray(
        np.full((3, 2, 2), 1 / 3),
        dims=("tercile", "lat", "lon"),
        coords={"tercile": [0, 1, 2], "lat": [0, 1], "lon": [0, 1]},
    )
    p = tmp_path / "t.nc"
    africas2s.write_terciles(probs, p, title="x", method="m")
    ds = xr.open_dataset(p)
    assert set(ds.data_vars) == {"below", "normal", "above"}
    np.testing.assert_allclose(ds["below"].values, 100 / 3, atol=1e-3)
    assert ds.attrs["title"] == "x"
    assert ds.attrs["method"] == "m"


def test_write_terciles_renormalizes_and_masks_invalid(tmp_path):
    # tercile probs that do not sum to 1 should be renormalized; a fully-NaN
    # column should stay NaN (not divide-by-zero garbage).
    raw = np.array(
        [
            [[0.2, np.nan], [0.5, 0.0]],
            [[0.2, np.nan], [0.5, 0.0]],
            [[0.2, np.nan], [0.0, 0.0]],
        ]
    )
    probs = xr.DataArray(
        raw,
        dims=("tercile", "lat", "lon"),
        coords={"tercile": [0, 1, 2], "lat": [0, 1], "lon": [0, 1]},
    )
    p = tmp_path / "t.nc"
    africas2s.write_terciles(probs, p, title="x")
    ds = xr.open_dataset(p)
    # cell (0,0): sums to 0.6 -> renormalized so below/normal/above ~ 33.3/33.3/33.3
    np.testing.assert_allclose(
        [ds["below"].values[0, 0], ds["normal"].values[0, 0], ds["above"].values[0, 0]],
        [100 / 3, 100 / 3, 100 / 3],
        atol=1e-3,
    )
    # cell (0,1): all-NaN input -> stays NaN
    assert np.isnan(ds["below"].values[0, 1])
    # cell (1,1): sums to 0 (not > 0) -> stays NaN, not divide-by-zero
    assert np.isnan(ds["below"].values[1, 1])
    # encoding: float32 with the expected fill value
    assert ds["below"].encoding.get("_FillValue") == -9999.0
    assert ds["below"].dtype == np.float32 or ds["below"].encoding.get("dtype") == "float32"


# ---------------------------------------------------------------------------
# B3: tercile_mae
# ---------------------------------------------------------------------------


def test_tercile_mae_zero_when_equal(tmp_path):
    probs = xr.DataArray(
        np.full((3, 2, 2), 1 / 3),
        dims=("tercile", "lat", "lon"),
        coords={"tercile": [0, 1, 2], "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    p = tmp_path / "ref.nc"
    africas2s.write_terciles(probs, p, title="x")
    # Not bit-exact zero: write_terciles encodes to float32 (per the B2 spec),
    # so round-tripping through NetCDF introduces ~1e-6 float32 quantization
    # error. That precision loss is inherent to the on-disk encoding (matches
    # the consumer's own write_tercile_netcdf) and not specific to this lift.
    assert africas2s.tercile_mae(probs, p) == pytest.approx(0.0, abs=1e-5)


def test_tercile_mae_accepts_dataarray_reference():
    rng = np.random.default_rng(1)
    cand = xr.DataArray(
        rng.random((3, 3, 3)),
        dims=("tercile", "lat", "lon"),
        coords={"tercile": [0, 1, 2], "lat": [0.0, 1.0, 2.0], "lon": [0.0, 1.0, 2.0]},
    )
    ref_percent = cand * 100.0  # identical up to the *100 scaling -> MAE 0
    result = africas2s.tercile_mae(cand, ref_percent)
    assert result == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# B4: plot_terciles export
# ---------------------------------------------------------------------------


def test_plot_terciles_is_exported():
    assert hasattr(africas2s, "plot_terciles")
    from africas2s.plotting.forecasts import plot_tercile_forecast

    assert africas2s.plot_terciles is plot_tercile_forecast
    assert "plot_terciles" in africas2s.__all__
