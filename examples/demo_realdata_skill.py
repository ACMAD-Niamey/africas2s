"""
Real cross-validated skill (RPSS) of downscaling methods: East Africa MAM precip.

The honest test: leave-one-year-out cross-validation on real CHIRPS + C3S/ECMWF
data, convert each method's held-out predictions to tercile probabilities, and
score with RPSS. RPSS is referenced to climatology, so:

    RPSS  > 0  ->  beats "just predict the average"  (real skill)
    RPSS == 0  ->  no better than climatology
    RPSS  < 0  ->  worse than climatology

Boundaries are computed leave-one-out (no leakage of the held-out year).

Run from the repo root:

    uv run python examples/demo_realdata_skill.py

Prereqs: Rosetta importable + CDS creds for the first fetch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
import africas2s as ds
from africas2s.cv import loyo
from africas2s.skill import skill
from africas2s.tercile import to_tercile_cv

OBS_REGION = [-12, 15, 22, 52]
GCM_REGION = [-20, 20, 10, 75]
YEARS = (1993, 2016)
MAM = [3, 4, 5]


def load_obs(coarsen=8):
    import rosetta
    dset = rosetta.fetch("obs/chirps-v2", "precip", hindcast=YEARS, region=OBS_REGION)
    da = dset["precip"].where(dset["precip"] >= 0)
    annual = da.sel(time=da.time.dt.month.isin(MAM)).groupby("time.year").mean("time")
    annual = annual.coarsen(lat=coarsen, lon=coarsen, boundary="trim").mean()
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
    return da.sel(lat=slice(OBS_REGION[0], OBS_REGION[1]),
                  lon=slice(OBS_REGION[2], OBS_REGION[3]))


# loyo() yields (train_years, test_year) folds; each fold is downscaled with the
# public africas2s.downscale() verb (fit on the train years, predict the held-out
# year) and the held-out predictions are scored together.
METHODS = [
    ("Climatology", "climatology", {}),
    ("Delta", "delta", {}),
    ("QM (emp)", "qm", {"variant": "empirical"}),
    ("QM (param)", "qm", {"variant": "parametric"}),
    ("DQM", "dqm", {"variant": "empirical"}),
    ("BCSD", "bcsd", {}),
    ("CCA", "cca", {"n_modes": 3}),
]


def cv_rpss(method, kwargs, gcm, obs, years):
    cv = []
    for train_years, test_year in loyo(years):
        pred = ds.downscale(
            gcm.sel(year=train_years), obs.sel(year=train_years),
            method=method, forecast=gcm.sel(year=test_year), verbose=False,
            **kwargs,
        ).mean("member")
        cv.append(pred.expand_dims(year=[test_year]))
    cv = xr.concat(cv, dim="year")
    terc = to_tercile_cv(cv, obs, method="bootstrap")
    report = skill(terc, obs, metrics=["rpss"], loo_boundaries=True, cv_window=1)
    return float(report.scores["rpss"])


def main():
    header = "Real CV skill (RPSS): East Africa MAM precip, 1993-2016"
    print(f"\n{header}\n" + "-" * len(header))

    obs = load_obs()
    gcm = load_gcm()
    years = [int(y) for y in obs.year.values]
    print(f"\n  obs {dict(obs.sizes)} | gcm {dict(gcm.sizes)} | {len(years)} LOYO folds\n")

    rows = []
    for name, method, kw in METHODS:
        rpss = cv_rpss(method, kw, gcm, obs, years)
        rows.append((name, rpss))
        print(f"  {name:12s}  RPSS = {rpss:+.3f}  "
              f"{'(beats climatology)' if rpss > 0.005 else '(~ climatology)' if rpss > -0.005 else '(worse)'}")

    print("\n  ranked:")
    for name, rpss in sorted(rows, key=lambda r: -r[1]):
        bar = "#" * max(0, int(round(rpss * 200)))
        print(f"  {name:12s} {rpss:+.3f} {bar}")


if __name__ == "__main__":
    main()
