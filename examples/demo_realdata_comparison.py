"""
Real-data comparison of downscaling methods: East Africa MAM precip.

Loads real CHIRPS observations + C3S/ECMWF seasonal-hindcast precip via Rosetta
(served from the local cache - no network needed once fetched), fits every
registered statistical method on 1993-2016, holds out a target year, and renders:
  - a map panel comparing each method's downscaled MAM precip to the CHIRPS truth
  - a skill table (area-mean, bias, spatial pattern correlation vs truth)

Covers qm / dqm / delta alongside climatology / bcsd / cca.

Run from the repo root:

    uv run python examples/demo_realdata_comparison.py

Prereqs: Rosetta importable + CDS creds for the first fetch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import africas2s as ds

OBS_REGION = [-12, 15, 22, 52]       # cached CHIRPS box (S, N, W, E)
GCM_REGION = [-20, 20, 10, 75]       # cached C3S predictor box (subset below)
YEARS = (1993, 2016)
MAM = [3, 4, 5]
TARGET_YEAR = 2015
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUT = OUTPUT_DIR / "realdata_eastafrica_mam_methods.png"


def load_obs():
    import rosetta
    dset = rosetta.fetch("obs/chirps-v2", "precip", hindcast=YEARS, region=OBS_REGION)
    da = dset["precip"].where(dset["precip"] >= 0)
    seasonal = da.sel(time=da.time.dt.month.isin(MAM))
    annual = seasonal.groupby("time.year").mean("time")     # (year, lat, lon)
    annual = annual.coarsen(lat=4, lon=4, boundary="trim").mean()  # ~0.05 -> ~0.2 deg
    return annual.sel(year=slice(YEARS[0], YEARS[1]))


def load_gcm():
    import rosetta
    dset = rosetta.fetch("c3s/ecmwf-monthly", "precip", init="2025-02", target="MAM",
                         hindcast=YEARS, region=GCM_REGION)
    da = dset["precip"]
    for dim in ("lead_time", "forecastMonth"):
        if dim in da.dims:
            da = da.mean(dim)
    if "number" in da.dims:
        da = da.rename({"number": "member"})
    for tdim in ("init_time", "time", "forecast_reference_time"):
        if tdim in da.dims:
            da = da.assign_coords(year=(tdim, da[tdim].dt.year.values))
            da = da.swap_dims({tdim: "year"}).drop_vars(tdim)
            break
    # Co-locate with the obs region for the local bias-correction methods.
    return da.sel(lat=slice(OBS_REGION[0], OBS_REGION[1]),
                  lon=slice(OBS_REGION[2], OBS_REGION[3]))


def pattern_corr(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main():
    header = "Real-data method comparison: East Africa MAM precip (CHIRPS + C3S)"
    print(f"\n{header}\n" + "-" * len(header))

    print("\nloading real data (Rosetta cache)...")
    obs = load_obs()
    gcm = load_gcm()
    print(f"  obs (CHIRPS): {dict(obs.sizes)}  precip[{float(obs.min()):.0f},{float(obs.max()):.0f}] mm/mo")
    print(f"  gcm (C3S):    {dict(gcm.sizes)}  precip[{float(gcm.min()):.2g},{float(gcm.max()):.2g}]")

    train_years = [y for y in obs.year.values if y != TARGET_YEAR]
    gcm_train = gcm.sel(year=train_years)
    obs_train = obs.sel(year=train_years)
    forecast = gcm.sel(year=TARGET_YEAR)
    truth = obs.sel(year=TARGET_YEAR)

    methods = [
        ("Climatology", "climatology", {}),
        ("Delta", "delta", {}),
        ("QM (emp)", "qm", {"variant": "empirical"}),
        ("QM (param)", "qm", {"variant": "parametric"}),
        ("DQM", "dqm", {"variant": "empirical"}),
        ("BCSD", "bcsd", {}),
        ("CCA", "cca", {"n_modes": 3}),
    ]

    print(f"\nfit on {len(train_years)} years, hold out {TARGET_YEAR}, predict...\n")
    preds = {}
    truth_mean = float(truth.mean())
    print(f"  {'method':12s} {'area-mean':>10s} {'bias':>8s} {'patt.corr':>10s}")
    print(f"  {'CHIRPS truth':12s} {truth_mean:10.1f} {'--':>8s} {'--':>10s}")
    for name, method_name, kwargs in methods:
        pred = ds.downscale(gcm_train, obs_train, method=method_name,
                            forecast=forecast, verbose=False, **kwargs).mean("member")
        preds[name] = pred
        am = float(pred.mean())
        print(f"  {name:12s} {am:10.1f} {am - truth_mean:+8.1f} "
              f"{pattern_corr(pred.values, truth.values):10.2f}")

    print("\nrendering map panel...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        panels = [("CHIRPS truth", truth)] + [(n, preds[n]) for n, _, _ in methods]
        vmax = float(np.nanpercentile(truth.values, 98))
        ncols = 4
        nrows = int(np.ceil(len(panels) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows),
                                 constrained_layout=True)
        for ax, (name, da) in zip(axes.ravel(), panels):
            im = ax.pcolormesh(da.lon, da.lat, da.values, cmap="YlGnBu",
                               vmin=0, vmax=vmax, shading="auto")
            ax.set_title(name, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
        for ax in axes.ravel()[len(panels):]:
            ax.axis("off")
        fig.colorbar(im, ax=axes, shrink=0.6, label="MAM precip (mm/month)")
        fig.suptitle(f"East Africa MAM {TARGET_YEAR}: downscaled precip by method "
                     f"(real CHIRPS + C3S/ECMWF)", fontsize=13, fontweight="bold")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {OUT}")
    except ImportError:
        print("  (matplotlib not installed - skipping the figure)")


if __name__ == "__main__":
    main()
