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
    # Trusting the attribute divides the axis by 24, so 46 "hours" span only two
    # days: every lead lands in day 1 or day 2 and the 7-day window is incomplete.
    # (Before steps were assigned to the day they end in, this over-selected
    # leads 24-46 silently.) The override recovers the axis.
    with pytest.raises(ValueError, match="2 of 7 days"):
        lead_window_reduce(da, {"week1": (1, 7)})
    right = lead_window_reduce(da, {"week1": (1, 7)}, lead_units="days")
    assert float(right.sel(window="week1").isel(lat=0, lon=0)) == pytest.approx(4.0)


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


# ------------------------------------------- timedelta leads and sub-daily steps
#
# acmadDL's issuance-keyed products (CHIRPS-GEFS, the rhiza/* forecasts) carry
# lead_time as a timedelta64, and the rhiza dynamical.org products step every
# 3-6 hours rather than daily. The reducer must read both: a timedelta axis
# needs no units attribute, a step belongs to the day it ends in, and summing
# a rate over sub-daily steps must integrate rather than add.

import pandas as pd
from africas2s.climate import forecast_increments, forecast_window


def _td_forecast(step_hours=24, n_days=15, units="mm/day", with_init=True, members=3):
    """acmadDL-shaped single-issuance forecast: (init_time, lead_time, member, lat, lon).

    Values are the lead day number (1-based, the day a step ends in) so sums and
    means over a window are predictable: a day-1 total of a 2 mm/day rate is 2 mm.
    """
    lead = pd.to_timedelta(np.arange(step_hours, n_days * 24 + step_hours, step_hours), unit="h")
    day = np.ceil(np.asarray(lead / pd.Timedelta(days=1))).astype("float32")
    data = np.broadcast_to(day[None, :, None, None, None], (1, lead.size, members, 2, 3)).copy()
    init = np.array(["2026-09-28T00:00:00"], dtype="datetime64[ns]")
    da = xr.DataArray(
        data, dims=("init_time", "lead_time", "member", "lat", "lon"),
        coords={"init_time": init, "lead_time": lead.values, "member": np.arange(members),
                "lat": [0.0, 1.0], "lon": [50.0, 51.0, 52.0]},
        attrs={"units": units},
    )
    da = da.assign_coords(time=da["init_time"] + da["lead_time"])     # acmadDL's valid time
    return da if with_init else da.squeeze("init_time", drop=True)


def test_lead_window_reduce_reads_a_timedelta_axis_without_units():
    daily = _td_forecast(step_hours=24, with_init=False)
    out = lead_window_reduce(daily, {"week1": (1, 7)})
    assert float(out.sel(window="week1").isel(member=0, lat=0, lon=0)) == pytest.approx(4.0)


def test_sub_daily_steps_belong_to_the_day_they_end_in():
    """3-hourly steps 3h..24h are all day 1; a (1, 1) window must take all eight."""
    fc = _td_forecast(step_hours=3, with_init=False)
    out = lead_window_reduce(fc, {"d1": (1, 1)}, how="mean")
    assert float(out.sel(window="d1").isel(member=0, lat=0, lon=0)) == pytest.approx(1.0)
    cnt = lead_window_reduce(fc.notnull().astype("float32").assign_attrs(units="1"), {"d1": (1, 1)}, how="sum")
    assert float(cnt.sel(window="d1").isel(member=0, lat=0, lon=0)) == pytest.approx(8.0)


def test_sum_of_sub_daily_rates_integrates_over_step_length():
    """Eight 3-hourly steps of a 1 mm/day rate are 1 mm, not 8."""
    fc = _td_forecast(step_hours=3, with_init=False, units="mm/day")
    out = lead_window_reduce(fc, {"week1": (1, 7)}, how="sum")
    assert float(out.sel(window="week1").isel(member=0, lat=0, lon=0)) == pytest.approx(28.0)  # sum(1..7) mm
    assert out.attrs["units"] == "mm"


def test_sum_of_sub_daily_amounts_adds_plainly():
    fc = _td_forecast(step_hours=3, with_init=False, units="mm")
    out = lead_window_reduce(fc, {"d1": (1, 1)}, how="sum")
    assert float(out.sel(window="d1").isel(member=0, lat=0, lon=0)) == pytest.approx(8.0)
    assert out.attrs["units"] == "mm"


def test_sum_of_sub_daily_steps_without_units_is_refused():
    fc = _td_forecast(step_hours=3, with_init=False)
    fc.attrs.pop("units")
    with pytest.raises(ValueError, match="units"):
        lead_window_reduce(fc, {"d1": (1, 1)}, how="sum")


def test_mean_over_uneven_sub_daily_steps_is_duration_weighted():
    """3-hourly to 144 h then 6-hourly, like IFS-ENS: a window straddling the
    change weights each step by how long it lasts."""
    lead = pd.to_timedelta(list(range(3, 145, 3)) + list(range(150, 361, 6)), unit="h")
    vals = np.where(lead <= pd.Timedelta(hours=144), 1.0, 3.0).astype("float32")   # day 6 = 1, day 7 = 3
    da = xr.DataArray(vals, dims=("lead_time",), coords={"lead_time": lead.values}, attrs={"units": "K"})
    out = lead_window_reduce(da, {"d6_7": (6, 7)}, how="mean")
    assert float(out.sel(window="d6_7")) == pytest.approx(2.0)      # one day of 1, one day of 3


def test_sub_daily_require_complete_counts_days_not_steps():
    fc = _td_forecast(step_hours=3, n_days=10, with_init=False)
    with pytest.raises(ValueError, match="days"):
        lead_window_reduce(fc, {"week2": (8, 14)})
    out = lead_window_reduce(fc, {"week2": (8, 14)}, require_complete=False)
    assert float(out.sel(window="week2").isel(member=0, lat=0, lon=0)) == pytest.approx(9.0)  # mean(8, 9, 10)


def test_chirps_gefs_style_day_axis_still_takes_lead_zero_in_a_zero_based_window():
    lead = pd.to_timedelta(np.arange(0, 16), unit="D")
    da = xr.DataArray(np.arange(16, dtype="float32"), dims=("lead_time",), coords={"lead_time": lead.values},
                      attrs={"units": "mm/day"})
    out = lead_window_reduce(da, {"first_week": (0, 6)}, how="sum")
    assert float(out.sel(window="first_week")) == pytest.approx(21.0)      # 0+1+...+6, one day per step


def test_forecast_window_gives_the_member_lat_lon_shape_predict_expects():
    fc = _td_forecast(step_hours=3)
    out = forecast_window(fc, (1, 10), how="sum")
    assert set(out.dims) == {"member", "lat", "lon"}
    assert float(out.isel(member=0, lat=0, lon=0)) == pytest.approx(55.0)   # sum(1..10) mm
    assert out.attrs["units"] == "mm" and out.attrs["lead_window"] == "days 1-10"


def test_forecast_window_accepts_a_named_window_and_a_bare_lead_axis():
    fc = _td_forecast(step_hours=24, with_init=False)
    out = forecast_window(fc, "week2", how="mean")
    assert float(out.isel(member=0, lat=0, lon=0)) == pytest.approx(11.0)   # mean(8..14)


def test_forecast_window_refuses_several_issuances():
    fc = xr.concat([_td_forecast(), _td_forecast()], dim="init_time")
    with pytest.raises(ValueError, match="init_time"):
        forecast_window(fc, (1, 7))


def test_forecast_increments_are_daily_totals_on_calendar_dates():
    fc = _td_forecast(step_hours=3, n_days=5)
    inc = forecast_increments(fc, how="sum")
    assert set(inc.dims) == {"time", "member", "lat", "lon"}
    assert list(pd.DatetimeIndex(inc["time"].values).strftime("%Y-%m-%d")) == [
        "2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01", "2026-10-02"]
    assert inc.isel(member=0, lat=0, lon=0).values.tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])
    assert inc.attrs["units"] == "mm"


def test_forecast_increments_drop_a_partial_last_day():
    fc = _td_forecast(step_hours=3, n_days=5).isel(lead_time=slice(0, 37))   # day 5 has 5 of 8 steps
    inc = forecast_increments(fc, how="sum")
    assert inc.sizes["time"] == 4


def test_forecast_increments_from_a_daily_axis_are_the_values_themselves():
    fc = _td_forecast(step_hours=24, n_days=3)
    inc = forecast_increments(fc, how="sum")
    assert inc.isel(member=0, lat=0, lon=0).values.tolist() == pytest.approx([1.0, 2.0, 3.0])


def test_forecast_increments_keep_a_complete_last_day_after_a_step_change():
    """IFS-ENS steps 3-hourly to 144 h then 6-hourly: day 15 has four steps like
    days 7-14, not eight like days 1-6, and is complete. Found on real data."""
    lead = pd.to_timedelta(list(range(3, 145, 3)) + list(range(150, 361, 6)), unit="h")
    day = np.ceil(np.asarray(lead / pd.Timedelta(days=1))).astype("float32")
    da = xr.DataArray(day, dims=("lead_time",), coords={"lead_time": lead.values,
                      "init_time": np.datetime64("2026-09-28", "ns")}, attrs={"units": "mm/day"})
    inc = forecast_increments(da, how="sum")
    assert inc.sizes["time"] == 15
    assert inc.values.tolist() == pytest.approx(list(range(1, 16)))
