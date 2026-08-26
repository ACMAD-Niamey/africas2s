"""
Demo: checkpoint a fitted downscaling method, reload it, and confirm the
reloaded model reproduces the original prediction exactly.

Network-free - uses small synthetic data. Run from the repository root:

    uv run python examples/demo_checkpoint_roundtrip.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from africas2s.methods.cca import CCAMethod


def _synthetic_data():
    rng = np.random.default_rng(0)
    years = np.arange(2000, 2015)
    members = np.arange(3)
    c_lat, c_lon = np.linspace(-4, 4, 5), np.linspace(30, 38, 5)
    f_lat, f_lon = np.linspace(-4, 4, 20), np.linspace(30, 38, 20)

    sig = np.sin(np.arange(len(years)) * 0.5)[:, None, None]
    gcm = xr.DataArray(
        sig[:, None] * np.outer(np.sin(c_lat * 0.5), np.cos(c_lon * 0.3))[None, None]
        + rng.standard_normal((len(years), len(members), 5, 5)) * 0.3 + 5.0,
        dims=["year", "member", "lat", "lon"],
        coords={"year": years, "member": members, "lat": c_lat, "lon": c_lon},
    )
    obs = xr.DataArray(
        sig * np.outer(np.sin(f_lat * 0.5), np.cos(f_lon * 0.3))[None]
        + rng.standard_normal((len(years), 20, 20)) * 0.2 + 5.0,
        dims=["year", "lat", "lon"],
        coords={"year": years, "lat": f_lat, "lon": f_lon},
    )
    forecast = gcm.isel(year=-1, drop=True)
    return gcm, obs, forecast


def main() -> None:
    header = "Checkpoint round-trip (CCA)"
    print(f"\n{header}\n" + "-" * len(header))

    gcm, obs, forecast = _synthetic_data()

    print("\n[1] Fit CCA on synthetic hindcast...")
    m1 = CCAMethod(n_modes=2)
    m1.fit(gcm, obs)
    print(f"    is_trained = {m1.is_trained}")
    expected = m1.predict(forecast)

    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "cca_checkpoint.pkl"
        print(f"\n[2] Save checkpoint -> {ckpt.name}")
        m1.save(ckpt)
        print(f"    {ckpt.stat().st_size} bytes on disk")

        print("\n[3] Load into a brand-new (untrained) instance...")
        m2 = CCAMethod()
        print(f"    before load: is_trained = {m2.is_trained}")
        m2.load(ckpt)
        print(f"    after  load: is_trained = {m2.is_trained}")

        print("\n[4] Predict with the reloaded model and compare...")
        got = m2.predict(forecast)
        np.testing.assert_array_equal(got.values, expected.values)
        print("    OK - reloaded prediction is bit-identical to the original.")

    print("\ncheckpoint round-trip demo complete.")


if __name__ == "__main__":
    main()
