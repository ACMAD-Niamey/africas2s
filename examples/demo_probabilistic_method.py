"""
Demo: the ProbabilisticMethodBase path.

Probabilistic methods (CorrDiff draws, XGBoost-quantile, EMOS) natively produce
an ensemble/distribution. They subclass ProbabilisticMethodBase, which gives:
  - predict_distribution(forecast) -> the full ensemble
  - predict(forecast)              -> ensemble mean (deterministic best guess)
and a tercile path that *counts members* instead of fitting a Gaussian to a
single deterministic forecast.

Network-free - uses small synthetic data + a minimal illustrative method.
Run from the repository root:

    uv run python examples/demo_probabilistic_method.py
"""
from __future__ import annotations

import numpy as np
import xarray as xr
import africas2s as ds
from africas2s.methods.base import ProbabilisticMethodBase
from africas2s.registry import register_method
from africas2s.tercile import to_tercile


@register_method("demo_spread")
class _SpreadMethod(ProbabilisticMethodBase):
    """Illustrative probabilistic method: regrid the coarse forecast to the obs
    grid and treat its ensemble members as the predictive distribution."""

    def fit(self, hindcast, obs, **kwargs):
        self.obs_coords_ = {"lat": obs.lat, "lon": obs.lon}

    def predict_distribution(self, forecast, **kwargs):
        if "year" in forecast.dims and forecast.sizes.get("year") == 1:
            forecast = forecast.isel(year=0, drop=True)
        return forecast.interp(
            lat=self.obs_coords_["lat"], lon=self.obs_coords_["lon"], method="linear"
        )


def _synthetic_data():
    rng = np.random.default_rng(0)
    years = np.arange(2000, 2012)
    members = np.arange(8)
    c_lat, c_lon = np.linspace(-4, 4, 5), np.linspace(30, 38, 5)
    f_lat, f_lon = np.linspace(-4, 4, 16), np.linspace(30, 38, 16)
    sig = np.sin(np.arange(len(years)) * 0.5)[:, None, None]
    gcm = xr.DataArray(
        sig[:, None] * np.outer(np.sin(c_lat * 0.5), np.cos(c_lon * 0.3))[None, None]
        + rng.standard_normal((len(years), len(members), 5, 5)) * 0.6 + 5.0,
        dims=["year", "member", "lat", "lon"],
        coords={"year": years, "member": members, "lat": c_lat, "lon": c_lon},
    )
    obs = xr.DataArray(
        sig * np.outer(np.sin(f_lat * 0.5), np.cos(f_lon * 0.3))[None]
        + rng.standard_normal((len(years), 16, 16)) * 0.2 + 5.0,
        dims=["year", "lat", "lon"],
        coords={"year": years, "lat": f_lat, "lon": f_lon},
    )
    return gcm, obs


def main() -> None:
    header = "Probabilistic method (counting terciles)"
    print(f"\n{header}\n" + "-" * len(header))

    gcm, obs = _synthetic_data()
    forecast = gcm.isel(year=-1, drop=True)

    m = _SpreadMethod()
    m.fit(gcm.isel(year=slice(None, -1)), obs.isel(year=slice(None, -1)))

    print("\n[1] predict_distribution() -> full ensemble")
    dist = m.predict_distribution(forecast)
    print(f"    dims={dict(dist.sizes)}  ('member' preserved)")

    print("\n[2] predict() -> deterministic ensemble mean")
    det = m.predict(forecast)
    print(f"    dims={dict(det.sizes)}  ('member' collapsed)")
    assert "member" not in det.dims

    print("\n[3] terciles BY COUNTING members (no Gaussian fit)")
    terc = to_tercile(dist, obs, method="counting")
    sums = terc.sum("tercile")
    print(f"    P sums to 1 everywhere: {bool(np.allclose(sums.values, 1.0))}")
    print(f"    P(below)={float(terc.sel(tercile=0).mean()):.2f}  "
          f"P(normal)={float(terc.sel(tercile=1).mean()):.2f}  "
          f"P(above)={float(terc.sel(tercile=2).mean()):.2f}")

    print("\n[4] same path via the public downscale() API")
    terc2 = ds.downscale(
        gcm, obs, method="demo_spread", output_type="tercile", verbose=False
    )
    print(f"    downscale(output_type='tercile') -> dims={dict(terc2.sizes)}")
    assert "tercile" in terc2.dims

    print("\nprobabilistic method demo complete.")


if __name__ == "__main__":
    main()
