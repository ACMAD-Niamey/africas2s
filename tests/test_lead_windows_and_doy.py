"""Sub-seasonal helpers: lead-window reduction and day-of-year climatology.

A sub-seasonal forecast lives on a lead axis, not a calendar one, so the
horizons a user asks for ("next week", "next 30 days") are windows of lead.
`seasonal_reduce` cannot express them -- it selects calendar months and
collapses to a `year` dim.
"""
import numpy as np
import pytest
import xarray as xr

import africas2s as a2s
from africas2s.climate import LEAD_WINDOWS_S2S, doy_anomaly, doy_climatology, lead_window_reduce


def _forecast(n_leads=46, units="hours"):
    """A lead-axis forecast shaped like ECMWF S2S: value == lead day number."""
    step = 24 if units == "hours" else 1
    lead = np.arange(1, n_leads + 1, dtype="float64") * step
    data = np.tile(np.arange(1, n_leads + 1, dtype="float32")[:, None, None], (1, 2, 3))
    da = xr.DataArray(
        data, dims=("lead_time", "lat", "lon"),
        coords={"lead_time": lead, "lat": [0.0, 1.0], "lon": [50.0, 51.0, 52.0]},
    )
    da["lead_time"].attrs["units"] = units
    return da


# --------------------------------------------------------------- lead windows

def test_lead_window_reduce_collapses_lead_to_named_windows():
    out = lead_window_reduce(_forecast())
    assert "lead_time" not in out.dims
    assert out.sizes["window"] == 5
    assert list(out.window.values) == list(LEAD_WINDOWS_S2S)
    assert out.sizes["lat"] == 2 and out.sizes["lon"] == 3      # other dims survive


def test_lead_window_reduce_selects_the_right_days():
    """Value equals lead day, so week1's mean is the mean of days 1-7."""
    out = lead_window_reduce(_forecast())
    assert float(out.sel(window="week1").isel(lat=0, lon=0)) == pytest.approx(4.0)   # mean(1..7)
    assert float(out.sel(window="week4").isel(lat=0, lon=0)) == pytest.approx(25.0)  # mean(22..28)
    assert float(out.sel(window="day1_30").isel(lat=0, lon=0)) == pytest.approx(15.5)  # mean(1..30)


def test_lead_window_reduce_reads_days_units_too():
    """A source filing leads in days must not be silently divided by 24."""
    out = lead_window_reduce(_forecast(units="days"))
    assert float(out.sel(window="week1").isel(lat=0, lon=0)) == pytest.approx(4.0)


def test_lead_window_reduce_lead_units_override_beats_the_attribute():
    """A mislabelled axis is recoverable without rewriting the data."""
    da = _forecast(units="days")
    da["lead_time"].attrs["units"] = "hours"         # wrong attribute
    # Trusting the attribute divides the axis by 24, so "week1" lands on a
    # different set of leads entirely -- and silently, since over-selection is
    # not something require_complete can catch.
    wrong = lead_window_reduce(da, {"week1": (1, 7)})
    right = lead_window_reduce(da, {"week1": (1, 7)}, lead_units="days")
    assert float(right.sel(window="week1").isel(lat=0, lon=0)) == pytest.approx(4.0)
    assert float(wrong.sel(window="week1").isel(lat=0, lon=0)) != pytest.approx(4.0)


def test_lead_window_reduce_rejects_an_incomplete_window():
    """A 4-day 'week' is a biased estimate that looks exactly like a clean one."""
    short = _forecast(n_leads=10)
    with pytest.raises(ValueError, match="only 3 of 7 days"):
        lead_window_reduce(short, {"week2": (8, 14)})


def test_lead_window_reduce_allows_incomplete_when_asked():
    short = _forecast(n_leads=10)
    out = lead_window_reduce(short, {"week2": (8, 14)}, require_complete=False)
    assert float(out.sel(window="week2").isel(lat=0, lon=0)) == pytest.approx(9.0)  # mean(8,9,10)


def test_lead_window_reduce_rejects_a_window_off_the_axis():
    with pytest.raises(ValueError, match="selects no leads"):
        lead_window_reduce(_forecast(n_leads=10), {"far": (40, 46)})


def test_lead_window_reduce_rejects_reversed_bounds():
    with pytest.raises(ValueError, match="is after last day"):
        lead_window_reduce(_forecast(), {"bad": (14, 7)})


def test_lead_window_reduce_requires_a_lead_dim():
    da = xr.DataArray(np.zeros((2, 3)), dims=("lat", "lon"))
    with pytest.raises(ValueError, match="needs a 'lead_time' dimension"):
        lead_window_reduce(da)


def test_lead_window_reduce_sum_for_accumulating_variables():
    out = lead_window_reduce(_forecast(), {"week1": (1, 7)}, how="sum")
    assert float(out.sel(window="week1").isel(lat=0, lon=0)) == pytest.approx(28.0)  # 1+..+7


# ------------------------------------------------------------ doy climatology

def _daily(years=(2020, 2021, 2022), amp=5.0):
    """A pure annual cycle plus a per-year offset, so the climatology is knowable."""
    times, vals = [], []
    for i, y in enumerate(years):
        dates = np.arange(np.datetime64(f"{y}-01-01"), np.datetime64(f"{y + 1}-01-01"))
        doy = np.arange(1, len(dates) + 1)
        times.append(dates)
        vals.append(amp * np.sin(2 * np.pi * doy / 365.0) + i)
    time = np.concatenate(times)
    data = np.concatenate(vals).astype("float32")
    return xr.DataArray(data, dims="time", coords={"time": time})


def test_doy_climatology_is_indexed_by_dayofyear():
    clim = doy_climatology(_daily())
    assert "dayofyear" in clim.dims
    assert "time" not in clim.dims
    assert clim.sizes["dayofyear"] in (365, 366)


def test_doy_climatology_averages_the_per_year_offsets():
    """Offsets 0,1,2 -> climatology sits on the +1 mean, cycle shape preserved."""
    clim = doy_climatology(_daily())
    expected = 5.0 * np.sin(2 * np.pi * 100 / 365.0) + 1.0
    assert float(clim.sel(dayofyear=100)) == pytest.approx(expected, abs=0.05)


def test_doy_climatology_honours_the_baseline():
    """Restricting to the first year alone drops the +1/+2 offsets."""
    clim = doy_climatology(_daily(), baseline=(2020, 2020))
    expected = 5.0 * np.sin(2 * np.pi * 100 / 365.0)
    assert float(clim.sel(dayofyear=100)) == pytest.approx(expected, abs=0.05)


def test_doy_climatology_rejects_an_empty_baseline():
    with pytest.raises(ValueError, match="no data in baseline"):
        doy_climatology(_daily(), baseline=(1990, 1991))


def test_doy_climatology_smoothing_wraps_the_year_boundary():
    """31 Dec and 1 Jan must stay continuous, not step at the seam."""
    clim = doy_climatology(_daily(), smooth=31)
    jump = abs(float(clim.isel(dayofyear=0)) - float(clim.isel(dayofyear=-1)))
    raw = doy_climatology(_daily())
    raw_jump = abs(float(raw.isel(dayofyear=0)) - float(raw.isel(dayofyear=-1)))
    assert jump <= raw_jump + 0.05          # smoothing must not introduce a seam
    assert clim.sizes["dayofyear"] == raw.sizes["dayofyear"]


def test_doy_anomaly_against_an_external_climatology():
    """The forecast/observation comparison: difference against a reference built elsewhere."""
    obs = _daily()
    clim = doy_climatology(obs, baseline=(2020, 2020))
    anom = doy_anomaly(obs, clim)
    assert "time" in anom.dims
    # year 2020 differenced against its own climatology is ~zero
    y2020 = anom.sel(time=slice("2020-01-01", "2020-12-31"))
    assert float(abs(y2020).max()) < 1e-4


def test_doy_anomaly_builds_its_own_climatology_when_not_given():
    anom = doy_anomaly(_daily())
    assert float(anom.mean()) == pytest.approx(0.0, abs=1e-5)


# ------------------------------------------------------------------ eio index

def test_eio_is_registered_as_the_absolute_eastern_pole():
    eio = a2s.Index.named("eio")
    assert eio.name == "eio"
    wio = a2s.Index.named("wio")
    # EIO mirrors WIO: absolute units, cos-lat weighted, one pole box each.
    assert eio.transform == wio.transform
