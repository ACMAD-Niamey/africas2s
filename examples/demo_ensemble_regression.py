"""
End-to-end demo: ensemble-regression (eReg) seasonal forecast on real data.

Real-data path (default): fetches a multi-model C3S precipitation hindcast
(ECMWF, MeteoFrance, CMCC; Feb-init -> MAM) and ERA5 precipitation observations
over East Africa via Rosetta, regrids each GCM onto the obs grid (eReg is a
per-cell calibration, so predictor and predictand must share a grid), and runs
`africas2s.calibrate(method="ereg")`. The published forecast terciles are the
average of each model's own calibrated-Gaussian terciles.

Always writes a dominant-tercile map PNG to examples/output/.

Run from the repository root:
  uv run python examples/demo_ensemble_regression.py             # real data (needs CDS)
  uv run python examples/demo_ensemble_regression.py --synthetic # offline, deterministic

Prerequisites for the real-data path: CDS credentials in ~/.cdsapirc and accepted
C3S/ERA5 dataset licences. Missing credentials/network fail at the fetch step (a
pre-existing requirement, not a bug). Use --synthetic offline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import africas2s as ds
from africas2s.plotting import plot_tercile_forecast

REGION = [-5, 5, 33, 48]                 # East Africa [lat_s, lat_n, lon_w, lon_e]
HINDCAST_YEARS = list(range(2000, 2015))
INIT_MONTH = "02"
TARGET = "MAM"
MODELS = ["c3s/ecmwf-monthly", "c3s/meteofrance", "c3s/cmcc"]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
CACHE_DIR = OUTPUT_DIR / "demo_cache"
PNG = OUTPUT_DIR / "ensemble_regression_tercile.png"


# --------------------------------------------------------------------------
# Real data
# --------------------------------------------------------------------------
def _load_or_fetch(cache_path, fetch_fn):
    if cache_path.exists():
        return xr.open_dataset(cache_path)
    d = fetch_fn()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    d.to_netcdf(cache_path)
    return d


def _era5_to_obs(dset, months, years):
    da = dset["precip"]
    seasonal = da.sel(time=da.time.dt.month.isin(months))
    annual = seasonal.groupby("time.year").mean("time")
    return annual.sel(year=years)


def _seasonal_to_gcm(dset, years, obs_grid):
    """C3S seasonal monthly -> (year, member, lat, lon), regridded to obs grid."""
    da = dset["precip"]
    keep = {"lat", "lon", "time", "member", "year",
            "forecast_reference_time", "init_time"}
    for dim in list(da.dims):
        if dim not in keep:
            da = da.mean(dim)
    for dim in ("forecast_reference_time", "init_time", "time"):
        if dim in da.dims:
            da = da.assign_coords(year=(dim, da[dim].dt.year.values))
            da = da.swap_dims({dim: "year"}).drop_vars(dim)
            break
    da = da.sel(year=years)
    # eReg per-cell calibration → put the GCM on the obs grid. Interp against
    # plain coordinate *values* so we don't drag in obs's stray scalar 'member'
    # coordinate (CDS attaches one), which would conflict with the GCM's real
    # 25-member ensemble dimension.
    return da.interp(lat=obs_grid["lat"].values, lon=obs_grid["lon"].values)


def build_real():
    import rosetta
    obs_ds = _load_or_fetch(
        CACHE_DIR / f"era5_precip_monthly_ea_{HINDCAST_YEARS[0]}-{HINDCAST_YEARS[-1]}.nc",
        lambda: rosetta.fetch("obs/era5", "precip",
                              hindcast=(HINDCAST_YEARS[0], HINDCAST_YEARS[-1]),
                              region=REGION, verbose=True),
    )
    obs = _era5_to_obs(obs_ds, [3, 4, 5], HINDCAST_YEARS)
    # Drop any stray scalar member/number coordinate CDS attaches to obs.
    obs = obs.drop_vars([c for c in ("member", "number") if c in obs.coords],
                        errors="ignore")
    print(f"  obs (ERA5 precip) {dict(obs.sizes)}")

    tracks = {}
    for product in MODELS:
        tag = product.split("/")[-1]
        gcm_ds = _load_or_fetch(
            CACHE_DIR / f"{tag}_precip_feb_mam_ea.nc",
            lambda product=product: rosetta.fetch(
                product, "precip", init=f"2025-{INIT_MONTH}", target=TARGET,
                hindcast=(HINDCAST_YEARS[0], HINDCAST_YEARS[-1]),
                region=REGION, verbose=True),
        )
        gcm = _seasonal_to_gcm(gcm_ds, HINDCAST_YEARS, obs)
        tracks[tag] = (gcm, None)
        print(f"  {tag} hindcast {dict(gcm.sizes)} (regridded to obs)")
    return {"prcp": tracks}, obs


# --------------------------------------------------------------------------
# Synthetic fallback (deterministic, offline): known calibration + spatial pattern
# --------------------------------------------------------------------------
def build_synthetic(slope=2.0, intercept=20.0, seed=0):
    rng = np.random.default_rng(seed)
    years = np.arange(1991, 2021)
    members, lat, lon = np.arange(8), np.linspace(-5, 12, 8), np.linspace(33, 48, 10)
    pattern = (np.cos(np.linspace(0, np.pi, len(lat)))[:, None]
               * np.sin(np.linspace(0.2, np.pi, len(lon)))[None, :])
    amp = rng.normal(0, 1, len(years))
    truth = amp[:, None, None] * pattern[None] + rng.normal(0, 0.4, (len(years), len(lat), len(lon)))

    def model(bias, mem_noise, sd):
        h = truth[:, None] + rng.normal(0, mem_noise, (len(years), len(members), len(lat), len(lon))) + bias
        return xr.DataArray(h, dims=["year", "member", "lat", "lon"],
                            coords={"year": years, "member": members, "lat": lat, "lon": lon})
    obs = xr.DataArray(slope * truth + intercept + rng.normal(0, 0.3, truth.shape),
                       dims=["year", "lat", "lon"], coords={"year": years, "lat": lat, "lon": lon})
    tracks = {"model_a": (model(3.0, 0.4, 1), None), "model_b": (model(-2.0, 0.5, 1), None)}
    return {"prcp": tracks}, obs


def save_png(tercile, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plot_tercile_forecast(tercile, title=title)
    fig.savefig(PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {PNG}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true",
                    help="use offline synthetic data instead of fetching real data")
    args = ap.parse_args()

    header = "Ensemble regression (eReg): " + ("synthetic" if args.synthetic else "real data")
    print(f"\n{header}\n" + "-" * len(header))

    if args.synthetic:
        tracks, obs = build_synthetic()
        fyear, title = 2020, "eReg MME: dominant tercile (synthetic)"
    else:
        tracks, obs = build_real()
        fyear, title = HINDCAST_YEARS[-1], f"eReg MME: dominant tercile (MAM {HINDCAST_YEARS[-1]}, ERA5/C3S)"

    # eReg is a calibrate-family method: per-model OLS calibration -> parametric
    # terciles -> cross-model average, via the calibrate() verb.
    models = tracks["prcp"]
    print(f"\n[run] calibrate(method='ereg') over {len(models)} model(s)...")
    tf = ds.calibrate(models, obs, method="ereg", forecast_year=fyear, verbose=False)

    assert tf.sizes["tercile"] == 3
    # skipna=False so uncalibratable (all-NaN) cells stay NaN rather than
    # summing to 0; validate the sum only where a forecast exists.
    s = tf.sum("tercile", skipna=False).values
    fin = np.isfinite(s)
    assert fin.any() and np.allclose(s[fin], 1.0, atol=1e-6)
    print(f"  {fin.sum()}/{fin.size} cells calibrated "
          f"({100 * (~fin).mean():.0f}% uncalibratable -> NaN)")
    bn, nn, an = (float(tf.sel(tercile=t).mean()) for t in (0, 1, 2))
    print(f"  forecast year {fyear}: domain-mean P(below/normal/above) = "
          f"{bn:.0%} / {nn:.0%} / {an:.0%}")
    print("PASS: valid per-model-averaged calibrated-error tercile forecast")

    save_png(tf, title)
    print("\nensemble-regression demo complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
