"""Regenerate the ``tests/data/cccma/*.npz`` parity fixtures for
``calibrate(method="smoothed_regression", engine="cccma")`` by running the reference
package itself (ECCC/CCCma ``s2d_calibration`` as deployed at ICPAC).

The reference lives at https://gitlab.com/climate-prediction-public/gcm_calibration_ICPAC
(v0.1.0-icpac, wrapping CCCma_s2d_postprocessing/Calibration @ f728ce17). It needs its
own environment — dask[distributed], lmoments3, netCDF4, PyYAML on top of xarray/scipy
— and is not pip-installable (incomplete pyproject), so it is imported from a checkout:

    git clone https://gitlab.com/climate-prediction-public/gcm_calibration_ICPAC ref
    uv venv refenv --python 3.12
    VIRTUAL_ENV=$PWD/refenv uv pip install "xarray==2025.6.1" "numpy==2.3.1" \\
        "scipy==1.16.0" "dask==2025.5.1" "distributed==2025.5.1" "lmoments3==1.0.8" \\
        "netCDF4==1.7.2" "PyYAML==6.0.2" "pandas==2.3.0"
    PYTHONPATH=$PWD/ref refenv/bin/python scripts/reproduce_cccma.py --work /tmp/cccma \\
        --out tests/data/cccma

Three synthetic cases are written: ``precip_1m`` (one model, 6 members, gamma
transform), ``precip_2m`` (two models with 6 and 4 members, NaN-padded), ``temp_1m``
(no transform). Each bundle holds the inputs (``X``, ``Y``, ``F``), the reference's
rescaling coefficients, smoothed variances, in-sample and forecast probabilities and
ensemble means, with dims and coords in the ``__meta__`` JSON entry.

Note: the reference's season-0 smoothed values depend on dask task placement (see
``tests/test_cccma_parity.py``), so two regenerations can differ there by an ulp.
"""
import argparse
import json
import logging
import pathlib
import shutil
import sys

import numpy as np
import xarray as xr
import yaml

NMON = 12
MONTHS = [f"{m:02d}" for m in range(1, 13)]
INIT_MONTH = "03"
INIT_YEAR = 2024

CASES = {
    "precip_1m": dict(kind="precip", members=[6], gamma=True, smallic=-1e-9, seed=1),
    "precip_2m": dict(kind="precip", members=[6, 4], gamma=True, smallic=-1e-9, seed=2),
    "temp_1m": dict(kind="temp", members=[6], gamma=False, smallic=0, seed=3),
}
EXPORT = (["X", "Y", "F", "XAVG", "XANOU", "YSDT", "YSDF", "XVARAU", "XVARTS", "XVARAS",
           "XSDEMS", "SCGC", "SCGA", "SCGF", "XSDEMC", "XSDEMA", "XSDEMF",
           "RPSGU", "RPSGC", "RPSGA", "RPSGF", "YAVC", "YPARA"]
          + [f"PXG{s}-TR" for s in "UCAF"]
          + [f"Fcst_PXG{s}-TR" for s in "UCAF"] + [f"Fcst_EMG{s}" for s in "UCAF"])


def synthetic(spec, nlat, nlon, years):
    rng = np.random.default_rng(spec["seed"])
    n_models, members = len(spec["members"]), spec["members"]
    nmax, ny = max(members), len(years)
    rho = 0.6
    X = np.full((nlat, nlon, NMON, ny, n_models, nmax), np.nan)
    F = np.full((nlat, nlon, n_models, nmax), np.nan)
    if spec["kind"] == "precip":
        base = rng.gamma(3.0, 40.0, (nlat, nlon, NMON))
        sig = rng.standard_normal((nlat, nlon, NMON, ny))
        for m, nm in enumerate(members):
            noise = rng.standard_normal((nlat, nlon, NMON, ny, nm))
            z = rho * sig[..., None] + np.sqrt(1 - rho ** 2) * noise
            X[..., m, :nm] = base[..., None, None] * np.exp(0.45 * z - 0.1) * (1.0 + 0.15 * m)
        zo = 0.5 * sig + np.sqrt(0.75) * rng.standard_normal(sig.shape)
        Y = base[..., None] * 1.1 * np.exp(0.5 * zo - 0.12)
        zf = rng.standard_normal()
        for m, nm in enumerate(members):
            z = rho * zf + np.sqrt(1 - rho ** 2) * rng.standard_normal((nlat, nlon, nm))
            F[..., m, :nm] = base[:, :, 2][..., None] * np.exp(0.45 * z - 0.1) * (1.0 + 0.15 * m)
    else:
        base = rng.normal(25.0, 3.0, (nlat, nlon, NMON))
        sig = rng.standard_normal((nlat, nlon, NMON, ny))
        for m, nm in enumerate(members):
            noise = rng.standard_normal((nlat, nlon, NMON, ny, nm))
            X[..., m, :nm] = base[..., None, None] + 0.8 * (rho * sig[..., None] + np.sqrt(1 - rho ** 2) * noise) + 0.3 * m
        Y = base[..., None] + 0.3 + (0.5 * sig + np.sqrt(0.75) * rng.standard_normal(sig.shape))
        zf = rng.standard_normal()
        for m, nm in enumerate(members):
            noise = rng.standard_normal((nlat, nlon, nm))
            F[..., m, :nm] = base[:, :, 2][..., None] + 0.8 * (rho * zf + np.sqrt(1 - rho ** 2) * noise) + 0.3 * m
    return X, Y, F


def write_case(name, spec, work, nlat, nlon, years):
    X, Y, F = synthetic(spec, nlat, nlon, years)
    lat = np.linspace(-1.5, -0.5, nlat)
    lon = np.linspace(36.5, 36.5 + nlon - 1, nlon)
    models = [f"M{i + 1}" for i in range(len(spec["members"]))]
    ens = np.arange(1, max(spec["members"]) + 1)
    d = work / name / "Data"
    if (work / name).exists():
        shutil.rmtree(work / name)
    d.mkdir(parents=True)
    # the reference's canonical layouts (so no transpose happens on load)
    xr.DataArray(X, dims=("latitude", "longitude", "month", "year", "model", "ensemble_member"),
                 coords=dict(latitude=lat, longitude=lon, month=MONTHS, year=years, model=models,
                             ensemble_member=ens), name="prate").to_dataset().to_netcdf(d / "X_lead1.nc")
    xr.DataArray(Y, dims=("latitude", "longitude", "month", "year"),
                 coords=dict(latitude=lat, longitude=lon, month=MONTHS, year=years),
                 name="PCP").to_dataset().to_netcdf(d / "Y.nc")
    xr.DataArray(F, dims=("latitude", "longitude", "model", "ensemble_member"),
                 coords=dict(latitude=lat, longitude=lon, model=models, ensemble_member=ens,
                             init_year=INIT_YEAR, init_month=INIT_MONTH),
                 name="prate").to_dataset().to_netcdf(d / f"X_lead1_{INIT_YEAR}-{INIT_MONTH}.nc")
    cfg = dict(
        CanSIPS_Datamart_format=False, CCCma_DCP_format=False, domain_subsetting_by_coords=False,
        preprocess_seasonal_obs_init_aligned=False, obs_path="unused",
        output_path=str(work) + "/", output_dir=name,
        variable_name_obs="PCP", variable_name_fcst="prate", variable_name_hcst="prate",
        leadtime=1, length_of_season=3, first_hindcast_yr=int(years[0]), last_hindcast_yr=int(years[-1]),
        fcst_initialization_yr=INIT_YEAR, fcst_initialization_month=INIT_MONTH,
        gamma_transform=spec["gamma"], Variance_smoothing="Gaussian", Coeff_smoothing="Gaussian",
        Gaussian_width=dict(calib=2, obsVar=1, fcstVar=1),
        Golden_tolerance=1e-3, SPVAL=1e38, SPVALT=1e32, SMALL=1e-6, SMALLIC=spec["smallic"],
        terciles=dict(fcst=True, hcst=True),
        probability_of_exceedance=dict(fcst=False, hcst=False, list_of_percentiles=[0.2, 0.5, 0.8]),
        save_empirical_boundaries=True, save_parametric_boundaries=True, save_gamma_obs=True,
        # SCGC/SCGA/SCGF/XPARA are held open lazily by the reference; listing them here
        # makes its final save loop fail with a permission error.
        additional_output=[k for k in EXPORT if k not in ("X", "Y", "F", "SCGC", "SCGA", "SCGF")
                           and not k.startswith(("PXG", "Fcst_"))],
        save_fcst_ensemble_mean=True, memory_limit="4GB",
    )
    with open(work / name / "config.yml", "w") as fh:
        yaml.safe_dump(cfg, fh)
    return cfg


def run_reference(cfg):
    from s2d_calibration.MakeCalib import calibrate
    from s2d_calibration.MakeFcst import forecast
    calibrate(cfg)
    forecast(cfg)


def export(name, work, out, cfg):
    d = work / name / "Data"
    tag = f"{INIT_YEAR}-{INIT_MONTH}"
    files = {"X": ("X_lead1.nc", "prate"), "Y": ("Y.nc", "PCP"), "F": (f"X_lead1_{tag}.nc", "prate")}
    for k in EXPORT:
        if k in files:
            continue
        files[k] = (f"{k}_{tag}_lead1.nc" if k.startswith("Fcst_") else f"{k}_lead1.nc", "prate")
    arrays, meta = {}, {"config": {k: cfg[k] for k in ["gamma_transform", "Gaussian_width", "Golden_tolerance",
                                                        "SMALL", "SMALLIC", "first_hindcast_yr",
                                                        "last_hindcast_yr", "fcst_initialization_month"]},
                        "coords": {}}
    for key, (fname, var) in files.items():
        path = d / fname
        if not path.exists():
            print(f"  {name}: {fname} not produced, skipped")
            continue
        da = xr.open_dataset(path, decode_cf=False)[var].load()
        arrays[key] = da.values
        meta[key] = list(da.dims)
        for c in da.coords:
            if c not in meta["coords"] and da[c].ndim == 1:
                meta["coords"][c] = da[c].values.tolist()
    arrays["__meta__"] = np.array(json.dumps(meta))
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{name}.npz"
    np.savez_compressed(target, **arrays)
    print(f"  wrote {target} ({target.stat().st_size / 1024:.0f} KB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, type=pathlib.Path, help="scratch directory for the reference's files")
    ap.add_argument("--out", default=pathlib.Path(__file__).resolve().parents[1] / "tests" / "data" / "cccma",
                    type=pathlib.Path, help="where the .npz bundles go")
    ap.add_argument("--cases", nargs="*", default=list(CASES), choices=list(CASES))
    ap.add_argument("--nlat", type=int, default=2)
    ap.add_argument("--nlon", type=int, default=2)
    ap.add_argument("--years", nargs=2, type=int, default=[1997, 2016])
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    years = np.arange(args.years[0], args.years[1] + 1)
    for name in args.cases:
        print(f"== {name}")
        cfg = write_case(name, CASES[name], args.work, args.nlat, args.nlon, years)
        run_reference(cfg)
        export(name, args.work, args.out, cfg)


if __name__ == "__main__":
    main()
