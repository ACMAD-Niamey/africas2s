"""
Demo: Quantile Mapping bias correction + downscaling.

Shows both variants correcting a deliberately biased GCM (mean 10, wide) toward
the observed climatology (mean 5, narrow):
  - empirical : nonparametric CDF matching, F_obs^-1(F_gcm(x))
  - parametric: Gaussian z-score rescaling

Network-free: synthetic data on a shared grid. Run from the repo root:

    uv run python examples/demo_quantile_mapping.py
"""
from __future__ import annotations

import numpy as np
import xarray as xr
import africas2s as ds


def _data():
    rng = np.random.default_rng(1)
    years = np.arange(2000, 2030)
    members = np.arange(5)
    lat, lon = np.linspace(0, 4, 4), np.linspace(30, 34, 4)
    gcm = xr.DataArray(
        rng.normal(10.0, 3.0, (len(years), len(members), 4, 4)),
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
    header = "Quantile mapping bias correction"
    print(f"\n{header}\n" + "-" * len(header))

    gcm, obs = _data()
    forecast = gcm.isel(year=-1)
    obs_mean = float(obs.mean())

    print(f"  obs climatology mean  = {obs_mean:.2f}")
    print(f"  raw GCM forecast mean = {float(forecast.mean()):.2f}  "
          f"(bias {float(forecast.mean()) - obs_mean:+.2f})")

    for variant in ("empirical", "parametric"):
        out = ds.downscale(gcm, obs, method="qm", variant=variant,
                           forecast=forecast, verbose=False)
        bias = float(out.mean()) - obs_mean
        print(f"  {variant:10s} QM mean   = {float(out.mean()):.2f}  "
              f"(bias {bias:+.2f})")

    print("\n  -> both variants pull the forecast onto the obs distribution.")


if __name__ == "__main__":
    main()
