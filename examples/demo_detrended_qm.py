"""
Demo: Detrended Quantile Mapping (Cannon et al. 2015).

Contrasts DQM with plain QM on a strongly warming GCM (trend +0.5/yr) against a
trendless obs record. Plain QM absorbs the trend into the CDF and maps the hot
final year back down toward the obs range; DQM removes the trend before mapping
and re-adds the GCM trend, preserving the warming signal.

Network-free - synthetic data on a shared grid. Run from the repo root:

    uv run python examples/demo_detrended_qm.py
"""
from __future__ import annotations

import numpy as np
import xarray as xr
import africas2s as ds


def _data():
    rng = np.random.default_rng(2)
    years = np.arange(1990, 2020)
    members = np.arange(4)
    lat, lon = np.linspace(0, 4, 4), np.linspace(30, 34, 4)
    trend = 0.5 * np.arange(len(years))           # strong GCM warming
    gcm = xr.DataArray(
        5.0 + trend[:, None, None, None]
        + rng.normal(0, 0.5, (len(years), len(members), 4, 4)),
        dims=["year", "member", "lat", "lon"],
        coords={"year": years, "member": members, "lat": lat, "lon": lon},
    )
    obs = xr.DataArray(
        5.0 + rng.normal(0, 0.5, (len(years), 4, 4)),  # trendless
        dims=["year", "lat", "lon"],
        coords={"year": years, "lat": lat, "lon": lon},
    )
    return gcm, obs


def main() -> None:
    header = "Detrended quantile mapping vs plain QM"
    print(f"\n{header}\n" + "-" * len(header))

    gcm, obs = _data()
    forecast = gcm.isel(year=-1)  # the warmest year

    print(f"  obs climatology mean    = {float(obs.mean()):.2f}  (no trend)")
    print(f"  raw GCM final-year mean = {float(forecast.mean()):.2f}  (warmed)")

    qm_out = float(ds.downscale(gcm, obs, method="qm", variant="empirical",
                                forecast=forecast, verbose=False).mean())
    dqm_out = float(ds.downscale(gcm, obs, method="dqm", variant="empirical",
                                 forecast=forecast, verbose=False).mean())

    print(f"\n  plain QM  output mean   = {qm_out:.2f}  (trend absorbed)")
    print(f"  DQM       output mean   = {dqm_out:.2f}  (trend preserved)")
    print(f"\n  -> DQM keeps {dqm_out - qm_out:+.2f} of warming signal QM discards.")


if __name__ == "__main__":
    main()
