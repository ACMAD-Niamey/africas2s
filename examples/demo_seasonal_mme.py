"""
End-to-end demo: seasonal_mme() for East Africa MAM temperature.

Demonstrates the single-call seasonal pipeline: instead of wiring up the
manual downscale/ensemble/to_tercile plumbing by hand, the whole workflow
runs through one `africas2s.seasonal_mme()` call. Uses ERA5 obs + ECMWF
SEAS5 hindcasts for MAM temperature over East Africa.

Run from the repository root:
  uv run python examples/demo_seasonal_mme.py

Prerequisites:
  1. Install Rosetta and AfricaS2S in local virtualenvs.
  2. Configure CDS credentials in ~/.cdsapirc (see rosetta/README.md).
  3. Accept CDS dataset licenses for ERA5 + C3S datasets.

If CDS credentials are absent the script will fail at the fetch step - that is
a pre-existing network/credential requirement, not a bug in the pipeline.
"""
from __future__ import annotations

from pathlib import Path

import xarray as xr
import africas2s as ds

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REGION = [-5, 5, 33, 42]          # East Africa [lat_s, lat_n, lon_w, lon_e]
HINDCAST_YEARS = list(range(2000, 2015))
INIT_MONTH = "02"
TARGET = "MAM"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
CACHE_DIR = OUTPUT_DIR / "demo_cache"
VERBOSE = True
PROGRESS = True

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_or_fetch(cache_path: Path, fetch_fn):
    """Load cached dataset when present; otherwise fetch and cache."""
    if cache_path.exists():
        return xr.open_dataset(cache_path)
    dset = fetch_fn()
    dset.to_netcdf(cache_path)
    return dset


def _era5_to_obs(dset, target_months, years):
    """ERA5 monthly Dataset -> seasonal-mean obs DataArray (year, lat, lon)."""
    da = dset["temp"]
    seasonal = da.sel(time=da.time.dt.month.isin(target_months))
    annual = seasonal.groupby("time.year").mean("time")
    return annual.sel(year=years)


def _seasonal_to_gcm(dset, years):
    """C3S seasonal-monthly Dataset -> GCM DataArray (year, member, lat, lon)."""
    da = dset["temp"]
    keep = {"lat", "lon", "time", "member", "year", "forecast_reference_time", "init_time"}
    for dim in list(da.dims):
        if dim not in keep:
            da = da.mean(dim=dim)
    for dim in ("forecast_reference_time", "init_time", "time"):
        if dim in da.dims:
            da = da.assign_coords(year=(dim, da[dim].dt.year.values))
            da = da.swap_dims({dim: "year"}).drop_vars(dim)
            break
    return da.sel(year=years)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    header = "Seasonal MME: East Africa MAM temperature"
    print(f"\n{header}\n" + "-" * len(header))

    # 1. Observations
    print("\n[1] ERA5 monthly temperature...")
    try:
        import rosetta
        era5_ds = _load_or_fetch(
            CACHE_DIR / "era5_temp_monthly.nc",
            lambda: rosetta.fetch(
                "obs/era5",
                "temp",
                hindcast=(HINDCAST_YEARS[0], HINDCAST_YEARS[-1]),
                region=REGION,
                verbose=VERBOSE,
                progress=PROGRESS,
            ),
        )
    except Exception as exc:
        print(f"\n[BLOCKED] Could not load/fetch ERA5 obs: {exc}")
        print("  Check ~/.cdsapirc credentials and network access.")
        return

    obs = _era5_to_obs(era5_ds, target_months=[3, 4, 5], years=HINDCAST_YEARS)
    print(f"    obs  {dict(obs.sizes)}  (0.25 deg ERA5)")

    # 2. GCM hindcast
    print("\n[2] C3S/ECMWF seasonal hindcast (Feb init -> MAM)...")
    try:
        gcm_ds = _load_or_fetch(
            CACHE_DIR / "c3s_ecmwf_temp_feb_mam.nc",
            lambda: rosetta.fetch(
                "c3s/ecmwf-monthly",
                "temp",
                init=f"2025-{INIT_MONTH}",
                target=TARGET,
                hindcast=(HINDCAST_YEARS[0], HINDCAST_YEARS[-1]),
                region=REGION,
                verbose=VERBOSE,
                progress=PROGRESS,
            ),
        )
    except Exception as exc:
        print(f"\n[BLOCKED] Could not load/fetch GCM hindcast: {exc}")
        print("  Check ~/.cdsapirc credentials and network access.")
        return

    gcm = _seasonal_to_gcm(gcm_ds, years=HINDCAST_YEARS)
    print(f"    gcm  {dict(gcm.sizes)}  (~1 deg ECMWF SEAS5)")

    # 3. Build predictor_tracks and call seasonal_mme()
    # Single-track, single-model example - straightforward API exercise.
    # Extend to {"prcp": {"ECMWF": (gcm, None)}, "sst": {"ECMWF": (sst_gcm, None)}}
    # for the full PyCPT dual-track workflow.
    predictor_tracks = {
        "prcp": {
            "ECMWF": (gcm, None),   # fcst=None -> use last hindcast year as forecast
        }
    }

    print("\n[3] Running seasonal_mme(method='cca', cv='loyo')...")
    print("    (this runs a full LOYO CV loop - may take a few minutes)")

    result = ds.seasonal_mme(
        predictor_tracks,
        obs,
        method="cca",
        cv="loyo",
        verbose=True,
    )

    # 4. Print headline outputs
    summary = "Results"
    print(f"\n{summary}\n" + "-" * len(summary))

    print(f"  Years used       : {result.metadata['years_used']}")
    print(f"  Forecast year    : {result.metadata['forecast_year']}")
    print(f"  CV scheme        : {result.metadata['cv']}")
    print(f"  Method           : {result.metadata['method']}")
    print(f"  Tercile method   : {result.metadata['tercile_method']}")
    print(f"  N members        : {result.metadata['n_members']}")

    print("\n  Skill scores (domain mean):")
    for metric, value in result.skill_report.scores.items():
        if isinstance(value, (int, float)):
            print(f"    {metric:25s}: {value:+.4f}")

    print(f"\n  tercile_forecast shape : {dict(result.tercile_forecast.sizes)}")
    print(f"  tercile_cv shape       : {dict(result.tercile_cv.sizes)}")

    bn = float(result.tercile_forecast.sel(tercile=0).mean())
    nn = float(result.tercile_forecast.sel(tercile=1).mean())
    an = float(result.tercile_forecast.sel(tercile=2).mean())
    print("\n  Headline probabilities (domain mean):")
    print(f"    P(below normal) : {bn:.1%}")
    print(f"    P(normal)       : {nn:.1%}")
    print(f"    P(above normal) : {an:.1%}")

    # 5. Save tercile_forecast to NetCDF
    nc_out = OUTPUT_DIR / "demo_seasonal_mme_tercile.nc"
    result.tercile_forecast.to_netcdf(nc_out)
    print(f"\n  saved tercile_forecast -> {nc_out}")

    # 6. Optional: skill PDF
    pdf_out = OUTPUT_DIR / "demo_seasonal_mme_skill.pdf"
    result.skill_report.metadata = {
        "region": "East Africa (5°S-5°N, 33-42°E)",
        "target": TARGET,
        "init": "February",
        "predictand": "2m air temperature",
        "method": "CCA (seasonal_mme)",
    }
    try:
        result.skill_report.to_pdf(pdf_out)
        print(f"  saved skill PDF     -> {pdf_out}")
    except Exception as exc:
        print(f"  (PDF export skipped: {exc})")

    print("\nseasonal MME demo complete.")


if __name__ == "__main__":
    main()
