"""
End-to-end demo: the WVG / logistic seasonal-forecast stream on real data.

Real-data path (default): fetches ERA5 sea-surface temperature over the tropical
Pacific (for the Western-V Gradient index) and ERA5 precipitation over East
Africa (the predictand) via Rosetta, builds a custom WVG index with
`africas2s.Index.custom(...)`, and runs `africas2s.calibrate(...)` with
`africas2s.LogitConfig` on gridded SST hindcast/forecast inputs. Always writes
a dominant-tercile PNG and NetCDF to examples/output/.

Run from the repository root:
  uv run python examples/demo_logistic_wvg.py             # real data (needs CDS)
  uv run python examples/demo_logistic_wvg.py --synthetic # offline, deterministic

Prerequisites for the real-data path: CDS credentials in ~/.cdsapirc and accepted
ERA5 licences. Missing credentials/network fail at the fetch step (a pre-existing
requirement, not a bug). Use --synthetic offline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import africas2s as ds
from africas2s.plotting import plot_tercile_forecast

PACIFIC = [-20, 40, 120, 240]            # WVG boxes live here (0-360 lon)
EAST_AFRICA = [-5, 5, 33, 48]
HINDCAST_YEARS = list(range(2000, 2021))
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
CACHE_DIR = OUTPUT_DIR / "demo_cache"
PNG = OUTPUT_DIR / "logistic_wvg_tercile.png"
NC = OUTPUT_DIR / "logistic_wvg_tercile.nc"


def _load_or_fetch(cache_path, fetch_fn):
    if cache_path.exists():
        return xr.open_dataset(cache_path)
    d = fetch_fn()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    d.to_netcdf(cache_path)
    return d


def build_real():
    import rosetta
    span = f"{HINDCAST_YEARS[0]}-{HINDCAST_YEARS[-1]}"
    sst_ds = _load_or_fetch(
        CACHE_DIR / f"era5_sst_monthly_pacific_{span}.nc",
        lambda: rosetta.fetch("obs/era5", "sst",
                              hindcast=(HINDCAST_YEARS[0], HINDCAST_YEARS[-1]),
                              region=PACIFIC, verbose=True),
    )
    pr_ds = _load_or_fetch(
        CACHE_DIR / f"era5_precip_monthly_ea_{span}.nc",
        lambda: rosetta.fetch("obs/era5", "precip",
                              hindcast=(HINDCAST_YEARS[0], HINDCAST_YEARS[-1]),
                              region=EAST_AFRICA, verbose=True),
    )
    # MAM-seasonal SST field (year, lat, lon) for the index. Drop any stray
    # scalar member/number coordinate CDS attaches.
    sst = sst_ds["sst"]
    sst = sst.sel(time=sst.time.dt.month.isin([3, 4, 5])).groupby("time.year").mean("time")
    sst = sst.sel(year=HINDCAST_YEARS)
    sst = sst.drop_vars([c for c in ("member", "number") if c in sst.coords], errors="ignore")
    # MAM precipitation predictand, coarsened to ~1deg to keep the per-cell
    # logistic snappy for a demo.
    pr = pr_ds["precip"]
    pr = pr.sel(time=pr.time.dt.month.isin([3, 4, 5])).groupby("time.year").mean("time")
    pr = pr.sel(year=HINDCAST_YEARS).interp(
        lat=np.arange(-5, 5.01, 1.0), lon=np.arange(33, 48.01, 1.0))
    pr = pr.drop_vars([c for c in ("member", "number") if c in pr.coords], errors="ignore")
    return sst, pr


def build_synthetic():
    rng = np.random.default_rng(0)
    years = np.arange(1991, 2021)
    lat, lon = np.arange(-40, 41, 5.0), np.arange(0, 360, 5.0)
    driver = rng.normal(0, 1, len(years))
    box = (((xr.DataArray(lat, dims="lat") >= -5) & (xr.DataArray(lat, dims="lat") <= 5))
           & ((xr.DataArray(lon, dims="lon") >= 190) & (xr.DataArray(lon, dims="lon") <= 240))).astype(float)
    sst = xr.DataArray(27.0 + driver[:, None, None] * box.values[None]
                       + rng.normal(0, 0.3, (len(years), len(lat), len(lon))),
                       dims=["year", "lat", "lon"], coords={"year": years, "lat": lat, "lon": lon})
    nlat, nlon = 6, 8
    olat, olon = np.linspace(-5, 12, nlat), np.linspace(33, 48, nlon)
    responsive = np.zeros((nlat, nlon)); responsive[:3, :] = 1.0   # low-lat half responds
    rain = 120.0 - 30.0 * driver[:, None, None] * responsive[None] + rng.normal(0, 12.0, (len(years), nlat, nlon))
    obs = xr.DataArray(rain, dims=["year", "lat", "lon"],
                       coords={"year": years, "lat": olat, "lon": olon})
    return sst, obs


def save_png(tercile, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plot_tercile_forecast(tercile, title=title)
    fig.savefig(PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {PNG}")


def save_netcdf(tercile):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tercile.rename("tercile_probability").to_netcdf(NC)
    print(f"  saved {NC}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true",
                    help="use offline synthetic data instead of fetching real data")
    args = ap.parse_args()

    header = "WVG logistic calibration: " + ("synthetic" if args.synthetic else "real data")
    print(f"\n{header}\n" + "-" * len(header))

    sst, obs = build_synthetic() if args.synthetic else build_real()

    # Custom WVG definition. Region inputs follow Rosetta's bbox convention:
    # [lat_s, lat_n, lon_w, lon_e].
    wvg = ds.Index.custom(
        name="wvg_candidate",
        regions={
            "nino34": [-5, 5, 190, 240],
            "wnp": [20, 35, 160, 210],
            "wep": [-15, 20, 120, 160],
            "wsp": [-30, -15, 155, 210],
        },
        combine=lambda z: z["nino34"] - (z["wnp"] + z["wep"] + z["wsp"]) / 3,
    )
    train_years = [int(y) for y in obs.year.values[:-1]]
    fcst_year = int(obs.year.values[-1])
    sst_train = sst.sel(year=train_years)
    sst_fcst = sst.sel(year=[fcst_year])
    obs_train = obs.sel(year=train_years)
    idx_fcst = float(wvg.reduce(sst_fcst, climatology=sst_train).values.reshape(-1)[0])
    print(f"  custom WVG index: {len(train_years)} training yrs; "
          f"forecasting {fcst_year} (its WVG = {idx_fcst:+.2f})")

    p = ds.calibrate(
        predictor_hindcast=sst_train,
        obs=obs_train,
        predictor_forecast=sst_fcst,
        method=ds.LogitConfig(
            index=wvg,
            model="independent_binomial",
            predictor_level="model_mean",
            detrend=False,
            significance=None,
            regularization=None,
        ),
    )
    assert p.dims[0] == "tercile" and p.sizes["tercile"] == 3
    # skipna=False so masked/uncalibratable cells stay NaN (not summed to 0).
    s = p.sum("tercile", skipna=False).values
    finite = np.isfinite(s)
    assert finite.any() and np.allclose(s[finite], 1.0, atol=1e-9)
    bn, nn, an = (float(p.sel(tercile=t).mean()) for t in (0, 1, 2))
    print(f"  domain-mean P(below/normal/above) = {bn:.0%} / {nn:.0%} / {an:.0%}")

    if args.synthetic:
        # The low-latitude half is index-responsive; with a known driver it
        # should be more decisive there than in the noise half.
        resp = float(p.isel(lat=slice(0, 3)).sel(tercile=0).mean())
        noise = float(p.isel(lat=slice(3, 6)).sel(tercile=0).mean())
        assert resp != noise
        print(f"  responsive P(below)={resp:.2f} vs noise={noise:.2f}")
    print("PASS: valid per-cell logistic tercile forecast")

    save_png(p, f"WVG logistic: dominant tercile ({fcst_year})"
             if not args.synthetic else "WVG logistic: dominant tercile (synthetic)")
    save_netcdf(p)
    print("\nWVG/logistic demo complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
