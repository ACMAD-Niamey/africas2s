"""Per-issuance S2S forecast generation orchestrator.

Given a country + issuance date, this script:
  1. fetches the ECMWF S2S forecast and matching reforecast suite via rosetta
  2. fetches the CHIRPS dekadal climatology for the climatology window
  3. iterates over the target dekads covered by the issuance
  4. per dekad, aggregates forecast/reforecast to a dekadal mean, then per
     method in the country's allowlist runs africas2s.downscale(),
     computes tercile probabilities from the ensemble, and writes an
     xr.Dataset to the issuance store

Invocation:
  uv run python -m scripts.s2s.run_issuance \\
      --country kenya --issuance 2026-05-15 --config scripts/s2s/s2s.yml

The function ``run_issuance`` is also importable for tests.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT.parent / "rosetta" / "src"))

import africas2s  # noqa: E402

from scripts.s2s.config import S2SConfig, load_config  # noqa: E402
from scripts.s2s.dekads import dekad_window, dekads_for_issuance  # noqa: E402
from scripts.s2s.issuance_store import write_issuance  # noqa: E402


def rosetta_fetch(product, **kwargs):
    """Thin wrapper around rosetta.fetch — easier to patch in tests."""
    import rosetta
    return rosetta.fetch(product=product, **kwargs)


# Substrings that identify recoverable "upstream doesn't have the data we
# asked for" failures from ECDS. They cover the two distinct paths we hit
# in practice: the on-the-fly reforecast not yet generated for some
# issuance, and the realtime S2S forecast being under embargo for very
# recent dates. Everything else (auth, network, code bugs) propagates.
_NO_DATA_SIGNATURES = (
    "MarsNoDataError",
    "MARS returned no data",
    "Restricted access to S2S data",
    "AccessError",
)


def _is_upstream_no_data(exc: BaseException) -> bool:
    msg = str(exc)
    return any(sig in msg for sig in _NO_DATA_SIGNATURES)


def _aggregate_to_dekad(da: xr.DataArray, issuance: date, target_start: date) -> xr.DataArray:
    """Mean of `da` over the lead-time hours that fall in [target_start, target_end).

    Real rosetta output for c3s/ecmwf-s2s has `lead_time` in hours (float64)
    coming from cfgrib's `step` dim — e.g. 24.0, 48.0, ..., 1104.0. Convert
    the target-dekad day-deltas to hours before slicing.
    """
    _, target_end = dekad_window(target_start)
    lead_min_hours = (target_start - issuance).days * 24
    lead_max_hours = (target_end - issuance).days * 24 - 1  # exclusive end → inclusive hour stop
    sel = da.sel(lead_time=slice(lead_min_hours, lead_max_hours))
    return sel.mean("lead_time")


def _tercile_probs_from_ensemble(member_field: xr.DataArray, obs_clim: xr.DataArray) -> xr.DataArray:
    """Tercile probabilities from an ensemble forecast.

    member_field: (member, lat, lon) downscaled forecast on the obs grid.
    obs_clim: (year, lat, lon) obs climatology used to compute the 33/67
              percentile boundaries per cell.
    Returns: (category, lat, lon) probability triple summing to 1 along category.
    """
    t33 = obs_clim.quantile(1 / 3, dim="year").drop_vars("quantile")
    t67 = obs_clim.quantile(2 / 3, dim="year").drop_vars("quantile")

    n = member_field.sizes["member"]
    below = (member_field < t33).sum("member") / n
    above = (member_field > t67).sum("member") / n
    normal = 1.0 - below - above

    out = xr.concat([below, normal, above], dim="category")
    out = out.assign_coords(category=["below", "normal", "above"]).transpose("category", "lat", "lon")
    return out.astype("float32")


def _dekad_of_year(d: date) -> int:
    """Return 1..36, matching CHIRPS dekadal indexing."""
    return (d.month - 1) * 3 + ((d.day - 1) // 10) + 1


def _obs_climatology_for_dekad(obs_full: xr.DataArray, target: date, climatology_years: tuple[int, int]) -> xr.DataArray:
    """Slice the obs climatology to the target dekad of year and the climatology window.

    Sheerwater's chirps_v2(agg_days=10) LEFT-aligns its 10-day rolling window
    (roll_and_agg subtracts agg-1 days after a right-aligned mean), so the value
    at label T is the mean of days [T, T+9]. The dekad starting on date D is
    therefore labelled D: we pick that day-of-year from each year in the
    climatology window and stack the results on a new `year` dimension.

    Returns (year, lat, lon).
    """
    y0, y1 = climatology_years
    obs = obs_full.sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))
    target_doy = target.timetuple().tm_yday
    mask = obs.time.dt.dayofyear == target_doy
    sel = obs.where(mask, drop=True)
    sel = (
        sel.assign_coords(year=("time", sel.time.dt.year.values))
        .swap_dims({"time": "year"})
        .drop_vars("time")
    )
    return sel


def _bbox_to_region(bbox: dict) -> list[float]:
    """Convert {min_lat, max_lat, min_lon, max_lon} → [lat_s, lat_n, lon_w, lon_e]
    (rosetta's region convention)."""
    return [bbox["min_lat"], bbox["max_lat"], bbox["min_lon"], bbox["max_lon"]]


def _regrid_nearest(da: xr.DataArray, target_lat, target_lon) -> xr.DataArray:
    """Nearest-neighbour regrid via xarray .interp."""
    return da.interp(lat=target_lat, lon=target_lon, method="nearest")


def run_issuance(*, country: str, issuance: date, config_path: Path | str) -> None:
    cfg: S2SConfig = load_config(config_path)
    if country not in cfg.countries:
        raise KeyError(f"country {country!r} not in config; have {sorted(cfg.countries)}")
    cc = cfg.countries[country]

    region = _bbox_to_region(cc.bbox)
    # ECMWF embargoes very recent S2S forecasts (returns "Restricted access
    # to S2S data" via MarsRuntimeError until the embargo lifts, typically
    # a few days after issuance). When the realtime forecast itself isn't
    # available, there's nothing this shard can produce — exit cleanly
    # rather than crash the workflow.
    try:
        fcst = rosetta_fetch(product=cc.forecast, init=issuance.isoformat(),
                             variable=cc.variable, region=region)[cc.variable]
    except Exception as e:  # noqa: BLE001 — narrow on the message below
        if _is_upstream_no_data(e):
            print(f"[run_issuance] realtime forecast for {issuance.isoformat()} "
                  f"is not yet available from ECDS ({type(e).__name__}); "
                  f"skipping this shard.")
            return
        raise

    # The on-the-fly reforecast suite isn't always published in lockstep
    # with the realtime forecast — issuances in the gap between a model
    # cycle changeover, or within ~24h of "now", can have a working
    # realtime forecast but no matching reforecast. Degrade gracefully:
    # skip the methods that need reforecast training (bcsd/cca/rank-analog)
    # but still produce raw + climatology for this issuance.
    try:
        refc = rosetta_fetch(product=cc.forecast, init=issuance.isoformat(),
                             variable=cc.variable, region=region,
                             reforecast=True)[cc.variable]
    except Exception as e:  # noqa: BLE001
        if _is_upstream_no_data(e):
            print(f"[run_issuance] no reforecast suite published for "
                  f"{issuance.isoformat()}; degrading to raw + climatology "
                  f"(skipping bcsd/cca/rank-analog).")
            refc = None
        else:
            raise
    # Pass the climatology window explicitly so sheerwater doesn't default
    # to "last year + this year" — which extends into the future and breaks
    # its internal rolling aggregation on empty time chunks.
    obs_full = rosetta_fetch(
        product=cc.obs, variable=cc.variable, region=region,
        hindcast=cfg.climatology_years,
    )[cc.variable].load()

    targets = dekads_for_issuance(issuance, cfg.lead_days)
    target_lat = obs_full["lat"]
    target_lon = obs_full["lon"]

    for target in targets:
        # Forecast and reforecast aggregated to this dekad.
        fcst_dekad = _aggregate_to_dekad(fcst, issuance, target)
        refc_dekad = (
            _aggregate_to_dekad(refc, issuance, target)
            if refc is not None else None
        )  # (year, member, lat, lon) or None when reforecast wasn't available
        obs_dekad = _obs_climatology_for_dekad(obs_full, target, cfg.climatology_years)

        for method_name in cc.methods:
            if method_name == "raw":
                regridded = _regrid_nearest(fcst_dekad.mean("member"), target_lat, target_lon)
                ds = xr.Dataset({"mean": regridded.astype("float32")})
                ds["mean"] = ds["mean"].fillna(0.0)
                tercile_input = _regrid_nearest(fcst_dekad, target_lat, target_lon).fillna(0.0)
                ds["tercile_probs"] = _tercile_probs_from_ensemble(tercile_input, obs_dekad)
                write_issuance(cfg.store_root, country, issuance, method_name, target, ds)
                continue

            if method_name == "climatology":
                clim_mean = obs_dekad.mean("year").astype("float32")
                ds = xr.Dataset({"mean": clim_mean})
                write_issuance(cfg.store_root, country, issuance, method_name, target, ds)
                continue

            # All other methods go through africas2s.downscale with
            # hindcast=reforecast. BCSD/CCA/rank-analog need the reforecast
            # for training; skip them when it wasn't available.
            if refc_dekad is None:
                continue

            # BCSD/CCA/rank-analog require obs and reforecast to share a
            # year axis — slice both to the intersection of the reforecast's
            # year range and the obs years available at this dekad.
            common_years = sorted(
                set(refc_dekad.year.values.tolist())
                & set(obs_dekad.year.values.tolist())
            )
            if not common_years:
                raise RuntimeError(
                    f"No overlapping years between reforecast "
                    f"({refc_dekad.year.values.min()}..{refc_dekad.year.values.max()}) "
                    f"and obs ({obs_dekad.year.values.min()}..{obs_dekad.year.values.max()}) "
                    f"for target {target}."
                )
            paired_refc = refc_dekad.sel(year=common_years)
            paired_obs = obs_dekad.sel(year=common_years)

            result = africas2s.downscale(
                predictor_hindcast=paired_refc,
                obs=paired_obs,
                forecast=fcst_dekad,
                method=method_name,
                output_type="continuous",
                verbose=False,
            )
            ds = xr.Dataset({"mean": result.mean("member").astype("float32")})
            # Tercile boundaries come from the full obs climatology, not
            # the paired subset — gives a more stable percentile estimate.
            ds["tercile_probs"] = _tercile_probs_from_ensemble(result, obs_dekad)
            write_issuance(cfg.store_root, country, issuance, method_name, target, ds)


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--country", required=True)
    ap.add_argument("--issuance", required=True, help="YYYY-MM-DD")
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    run_issuance(
        country=args.country,
        issuance=date.fromisoformat(args.issuance),
        config_path=args.config,
    )


if __name__ == "__main__":
    _cli()
