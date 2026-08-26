"""
Demo: Delta scaling (Hay et al. 2000).

The simplest baseline: prediction = obs_clim + (GCM forecast - GCM hist clim).
The GCM's *absolute* bias (mean ~12 here vs obs ~5) cancels out; only its
anomaly is kept and laid on top of the observed climatology.

Network-free - synthetic data on a shared grid. Run from the repo root:

    uv run python examples/demo_delta_scaling.py
"""
from __future__ import annotations

import numpy as np
import xarray as xr
import africas2s as ds


def _data():
    rng = np.random.default_rng(3)
    years = np.arange(2000, 2020)
    members = np.arange(4)
    lat, lon = np.linspace(0, 4, 4), np.linspace(30, 34, 4)
    gcm = xr.DataArray(
        rng.normal(12.0, 2.0, (len(years), len(members), 4, 4)),  # biased +7
        dims=["year", "member", "lat", "lon"],
        coords={"year": years, "member": members, "lat": lat, "lon": lon},
    )
    obs = xr.DataArray(
        rng.normal(5.0, 1.0, (len(years), 4, 4)),
        dims=["year", "lat", "lon"],
        coords={"year": years, "lat": lat, "lon": lon},
    )
    return gcm, obs


def main() -> None:
    header = "Delta scaling baseline"
    print(f"\n{header}\n" + "-" * len(header))

    gcm, obs = _data()
    forecast = gcm.isel(year=-1)

    out = ds.downscale(gcm, obs, method="delta", forecast=forecast, verbose=False)

    raw_anom = float(forecast.mean()) - float(gcm.mean(["member", "year"]).mean())
    print(f"  obs climatology mean      = {float(obs.mean()):.2f}")
    print(f"  raw GCM forecast mean     = {float(forecast.mean()):.2f}  (biased high)")
    print(f"  GCM anomaly vs hist clim  = {raw_anom:+.2f}")
    print(f"  delta-scaled output mean  = {float(out.mean()):.2f}  "
          f"(obs clim + GCM anomaly)")
    print("\n  -> GCM's absolute bias drops out; only its anomaly survives.")


if __name__ == "__main__":
    main()
