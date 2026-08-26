"""
Demo: train / inference separation.

Pattern for expensive (DL) methods - train once, checkpoint, then run inference
any number of times without refitting:

    ds.train(method, hindcast, obs, save_to=PATH)            # expensive, once
    ds.downscale(forecast, method=method, weights_path=PATH) # cheap, many

Here we use CCA (cheap) to demonstrate the mechanics end-to-end, and show the
guard that stops a `requires_training=True` method from being fit inline by
`downscale()`.

Network-free - synthetic data. Run from the repo root:

    uv run python examples/demo_train_inference.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
import africas2s as ds
from africas2s.methods.base import MethodBase
from africas2s.registry import register_method


@register_method("demo_dl_stub")
class _DLStub(MethodBase):
    """Stand-in for a slow DL method: flagged requires_training=True."""

    requires_training = True

    def fit(self, hindcast, obs, **kwargs):
        self.obs_coords_ = {"lat": obs.lat, "lon": obs.lon}

    def predict(self, forecast, **kwargs):
        if "year" in forecast.dims and forecast.sizes.get("year") == 1:
            forecast = forecast.isel(year=0, drop=True)
        return forecast.interp(
            lat=self.obs_coords_["lat"], lon=self.obs_coords_["lon"]
        )


def _data():
    rng = np.random.default_rng(7)
    years = np.arange(2000, 2012)
    members = np.arange(3)
    c_lat, c_lon = np.linspace(-4, 4, 5), np.linspace(30, 38, 5)
    f_lat, f_lon = np.linspace(-4, 4, 12), np.linspace(30, 38, 12)
    sig = np.sin(np.arange(len(years)) * 0.5)[:, None, None]
    gcm = xr.DataArray(
        sig[:, None] * np.outer(np.sin(c_lat * 0.5), np.cos(c_lon * 0.3))[None, None]
        + rng.standard_normal((len(years), len(members), 5, 5)) * 0.3 + 5.0,
        dims=["year", "member", "lat", "lon"],
        coords={"year": years, "member": members, "lat": c_lat, "lon": c_lon},
    )
    obs = xr.DataArray(
        sig * np.outer(np.sin(f_lat * 0.5), np.cos(f_lon * 0.3))[None]
        + rng.standard_normal((len(years), 12, 12)) * 0.2 + 5.0,
        dims=["year", "lat", "lon"],
        coords={"year": years, "lat": f_lat, "lon": f_lon},
    )
    return gcm, obs


def main() -> None:
    header = "Train / inference separation"
    print(f"\n{header}\n" + "-" * len(header))

    gcm, obs = _data()
    forecast = gcm.isel(year=-1, drop=True)

    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "cca.pkl"

        print("\n[1] train() once -> checkpoint")
        m = ds.train(
            "cca", gcm.isel(year=slice(None, -1)), obs.isel(year=slice(None, -1)),
            save_to=ckpt, n_modes=2, verbose=False,
        )
        expected = m.predict(forecast)
        print(f"    wrote {ckpt.name} ({ckpt.stat().st_size} bytes)")

        print("\n[2] downscale(weights_path=...) -> inference only, no refit")
        result = ds.downscale(
            predictor_hindcast=forecast, method="cca",
            weights_path=str(ckpt), verbose=False,
        )
        np.testing.assert_array_equal(result.values, expected.values)
        print("    inference matches the trained model bit-for-bit.")

    print("\n[3] guard: a requires_training=True method can't be fit inline")
    try:
        ds.downscale(gcm, obs, method="demo_dl_stub", verbose=False)
        print("    (unexpected: no error raised)")
    except RuntimeError as e:
        print(f"    RuntimeError -> {str(e)[:72]}...")

    print("\ntrain/inference demo complete.")


if __name__ == "__main__":
    main()
