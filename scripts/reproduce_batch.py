#!/usr/bin/env python3
"""
reproduce_batch.py
==================
End-to-end pipeline comparison: AfricaS2S/Rosetta  vs  pycpt/CPT/IRI.

Runs all 3 regions × 3 seasons for two validated GCM configurations
and saves four plots into ./output/ (relative to this script):

  ecmwf_matrix.png   — 9-row × 10-col spatial comparison (ECMWF SEAS51)
  ecmwf_table.png    — summary metrics table (ECMWF SEAS51)
  cfsv2_matrix.png   — 9-row × 10-col spatial comparison (CFSv2 PSF)
  cfsv2_table.png    — summary metrics table (CFSv2 PSF)

Configurations
--------------
ECMWF:  SEAS51c monthly, 1993-2016, EOF modes (8, 6, 3)  — default, well-determined.
CFSv2:  PENTAD_SAMPLES_FULL, 1993-2010, EOF modes (4, 4, 2)  — reduced modes required
        because 18 training years / 13 per fold makes CCA underdetermined at (8,6,3).
        PSF (PENTAD_SAMPLES_FULL) aligns with the IRI endpoint pycpt uses.
        See AFRICAS2S_USAGE_NOTES.md §§9-10 for the full diagnosis.

Usage
-----
  cd africas2s
  conda run -n accord python scripts/reproduce_batch.py
"""

import sys, io, time, shutil, traceback
from pathlib import Path

import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.gridspec import GridSpec
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

REPO    = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO / "rosetta"   / "src"))
sys.path.insert(0, str(REPO / "africas2s" / "src"))

import rosetta
from africas2s.methods.cca      import CCAMethod
from africas2s.cv                import loyo as ds_loyo
from africas2s.tercile           import to_tercile_cv
from africas2s.metrics.rpss      import RPSSMetric
from africas2s.metrics.pearson   import PearsonMetric
import pycpt

# ---------------------------------------------------------------------------
# Run configurations
# ---------------------------------------------------------------------------

# Season months map — used for CHIRPS obs seasonal aggregation and PSF URL building.
SEASON_MONTHS = {
    "MAM": [3, 4, 5], "SON": [9, 10, 11], "DJF": [12, 1, 2],
}

REGIONS = {
    "kenya":    {"predictand": [-5,  5,  33, 42], "predictor": [-20, 20,  30, 100]},
    "ethiopia": {"predictand": [ 3, 15,  33, 48], "predictor": [-20, 25,  30, 100]},
    "nigeria":  {"predictand": [ 4, 14,   3, 15], "predictor": [-20, 25, -40,  40]},
}
SEASONS = {
    "MAM": {"init_month": 2,  "target": "Mar-May", "init_str": "2025-02"},
    "SON": {"init_month": 8,  "target": "Sep-Nov",  "init_str": "2025-08"},
    "DJF": {"init_month": 11, "target": "Dec-Feb",  "init_str": "2025-11"},
}
ALL_REGIONS = ["kenya", "ethiopia", "nigeria"]
ALL_SEASONS = ["MAM", "SON", "DJF"]

# Two validated configurations.
CONFIGS = {
    "ecmwf": {
        "label":       "ECMWF SEAS51 monthly",
        "pycpt_model": "SEAS51c.PRCP",
        "rosetta_gcm": "c3s/ecmwf-monthly",
        "years":       (1993, 2016),
        "x_eof":       8, "y_eof": 6, "cca_modes": 3,
        "cv_window":   5, "obs_coarsen": 5,
        "use_psf":     False,
    },
    "cfsv2": {
        # PSF = PENTAD_SAMPLES_FULL: same IRI endpoint pycpt/CPT uses.
        # Modes reduced to (4,4,2): x_eof ≤ (n_train-2)/2 ≈ 5 for 13-sample folds.
        "label":       "CFSv2 PENTAD_SAMPLES_FULL (4/4/2 modes)",
        "pycpt_model": "CFSv2.PRCP",
        "rosetta_gcm": "nmme/cfsv2",
        "years":       (1993, 2010),
        "x_eof":       4, "y_eof": 4, "cca_modes": 2,
        "cv_window":   5, "obs_coarsen": 5,
        "use_psf":     True,
    },
}

# IRI INGRID base URL for CFSv2 PENTAD_SAMPLES_FULL (PSF).
_PSF_BASE    = ("https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME"
                "/.NCEP-CFSv2/.HINDCAST/.PENTAD_SAMPLES_FULL/.prec")
_MONTH_ABBR  = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}


# ---------------------------------------------------------------------------
# Data fetching helpers
# ---------------------------------------------------------------------------

def _box(region_list):
    """Convert [S, N, W, E] list to pycpt extent dict."""
    return {"south": region_list[0], "north": region_list[1],
            "west":  region_list[2], "east":  region_list[3]}


def fetch_obs_ros(season_name, region, years, obs_coarsen):
    """Fetch CHIRPS obs via Rosetta, aggregate seasonally, coarsen to match IRI grid."""
    months = SEASON_MONTHS[season_name]
    cross  = months[0] > months[-1]            # True for DJF (Dec crosses year boundary)

    obs_ds  = rosetta.fetch("obs/chirps-v2-monthly", "precip",
                            hindcast=(years[0], years[-1]), region=region)
    obs_raw = obs_ds["precip"].where(obs_ds["precip"] >= 0)
    obs_sea = obs_raw.sel(time=obs_raw.time.dt.month.isin(months))

    if cross:
        # Months after the year-turn (Jan, Feb for DJF) belong to the season that
        # started in the prior December.  Shift those year labels back by one so
        # groupby aggregates them with the correct December.
        yr_vals = np.where(obs_sea.time.dt.month.values < months[0],
                           obs_sea.time.dt.year.values - 1,
                           obs_sea.time.dt.year.values)
        obs_sea = xr.DataArray(obs_sea.values, dims=list(obs_sea.dims),
                               coords={"time": yr_vals, "lat": obs_sea.lat.values,
                                       "lon": obs_sea.lon.values})
        obs_ros = (obs_sea.groupby("time").mean("time")
                          .rename(time="year")
                          .sel(year=slice(years[0], years[-1])))
        # Relabel to IRI convention: Rosetta init-year N → Jan-year N+1
        # (IRI labels DJF by the January year; Rosetta labels by the December year).
        obs_ros = obs_ros.assign_coords(year=obs_ros.year.values + 1)
    else:
        obs_ros = obs_sea.groupby("time.year").mean("time")

    return obs_ros.coarsen(lat=obs_coarsen, lon=obs_coarsen, boundary="trim").mean()


def fetch_gcm_ros(season_name, region, cfg):
    """Fetch GCM hindcast via Rosetta (ECMWF) or INGRID PSF (CFSv2)."""
    months = SEASON_MONTHS[season_name]
    cross  = months[0] > months[-1]
    years  = cfg["years"]
    sea    = SEASONS[season_name]

    if cfg["use_psf"]:
        gcm_ros = _fetch_psf(season_name, region, years[0], years[-1])
    else:
        gcm_ds  = rosetta.fetch(cfg["rosetta_gcm"], "precip",
                                init=sea["init_str"], target=season_name,
                                hindcast=years, region=region)
        gcm_ros = gcm_ds["precip"]
        # Collapse any residual lead dimension (different adapters name it differently)
        for dim in ("lead_time", "forecastMonth", "L"):
            if dim in gcm_ros.dims:
                gcm_ros = gcm_ros.mean(dim)
        if "number" in gcm_ros.dims:
            gcm_ros = gcm_ros.rename({"number": "member"})
        elif "member" not in gcm_ros.dims:
            gcm_ros = gcm_ros.expand_dims("member")
        # Promote init-time dim to integer year coord
        for tdim in ("init_time", "time", "forecast_reference_time"):
            if tdim in gcm_ros.dims:
                gcm_ros["year"] = (tdim, gcm_ros[tdim].dt.year.values)
                gcm_ros = gcm_ros.swap_dims({tdim: "year"}).drop_vars(tdim)
                break

    if cross:
        gcm_ros = gcm_ros.assign_coords(year=gcm_ros.year.values + 1)
    return gcm_ros


def _fetch_psf(season_name, region, first_year, final_year):
    """Fetch CFSv2 hindcast from PENTAD_SAMPLES_FULL (same endpoint as pycpt/IRI).

    Builds an IRI INGRID URL that:
      - selects init months spanning first_year..final_year
      - averages over the three target-season lead months ([L]//keepgrids/average)
      - preserves the member (M) dimension so CCA can average internally
    Returns DataArray with dims (year, member, lat, lon).
    """
    months  = SEASON_MONTHS[season_name]
    cross   = months[0] > months[-1]
    init_m  = SEASONS[season_name]["init_month"]
    mon_str = _MONTH_ABBR[init_m]
    y0      = first_year - 1 if cross else first_year  # init year (Nov for DJF)
    y1      = final_year - 1 if cross else final_year
    lat_s, lat_n, lon_w, lon_e = region

    # Lead bounds: L = lead_month - 0.5 (all three-month seasons use 1.5–3.5)
    url = (f"{_PSF_BASE}"
           f"/S/%280000%201%20{mon_str}%20{y0}-{y1}%29/VALUES"
           f"/L/1.5/3.5/RANGEEDGES/%5BL%5D//keepgrids/average"
           f"/Y/{lat_s}/{lat_n}/RANGEEDGES"
           f"/X/{lon_w}/{lon_e}/RANGEEDGES"
           f"/data.nc")

    resp   = requests.get(url, timeout=360); resp.raise_for_status()
    # decode_times=False: INGRID uses 360-day calendar which xarray can't decode
    ds_raw = xr.open_dataset(io.BytesIO(resp.content), engine="scipy", decode_times=False)
    var    = "prec" if "prec" in ds_raw else list(ds_raw.data_vars)[0]
    da     = ds_raw[var].rename({k: v for k, v in
                                 [("S","year"),("M","member"),("Y","lat"),("X","lon")]
                                 if k in ds_raw[var].dims})

    # Squeeze the degenerate L dim that INGRID leaves after [L]//keepgrids/average
    if "L" in da.dims and da.sizes["L"] == 1:
        da = da.squeeze("L", drop=True)

    # Decode year coordinate from "months since <epoch>" float values
    s_units = ds_raw["S"].attrs.get("units", "") if "S" in ds_raw else ""
    if "months since" in s_units:
        from rosetta.normalize import decode_months_since
        yrs, _ = decode_months_since(s_units, ds_raw["S"].values)
        da = da.assign_coords(year=yrs)
    else:
        da = da.assign_coords(year=np.arange(y0, y0 + da.sizes["year"]))

    if "member" not in da.dims:
        da = da.expand_dims("member")
    return da


def _adapt_pycpt(gcm_iri, obs_iri, years):
    """Rename IRI/pycpt dims (T, Y, X) to (year, lat, lon) and assign year coords."""
    def _ren(da, rmap, yr):
        da = da.rename({k: v for k, v in rmap.items() if k in da.dims})
        if "year" in da.dims:
            n, n_yr = da.sizes["year"], len(yr)
            if n == n_yr:
                da["year"] = yr
            elif n > n_yr and n % n_yr == 0:
                # Monthly obs (e.g. DJF T=72 = 3 months × 24 years): reshape and mean
                k    = n // n_yr
                vals = da.values.reshape((n_yr, k) + da.values.shape[1:]).mean(axis=1)
                sp   = list(da.dims[1:])
                da   = xr.DataArray(vals, dims=["year"] + sp,
                                    coords={"year": yr, **{d: da.coords[d].values
                                                           for d in sp if d in da.coords}})
        return da
    gcm = _ren(gcm_iri, {"T": "year", "Y": "lat", "X": "lon"}, years)
    if "member" not in gcm.dims:
        gcm = gcm.expand_dims("member")
    obs = _ren(obs_iri, {"T": "year", "Y": "lat", "X": "lon"}, years)
    return gcm, obs


def _corrcoef(a, b):
    """Pearson r between two arrays, ignoring NaN."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    m    = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() >= 3 else np.nan


# ---------------------------------------------------------------------------
# Single-slot pipeline
# ---------------------------------------------------------------------------

def run_slot(region_name, season_name, cfg):
    """Run one region × season comparison slot.

    Returns a dict with scalar metrics and in-memory xarray arrays, or None on error.
    """
    reg    = REGIONS[region_name]
    sea    = SEASONS[season_name]
    years  = list(range(cfg["years"][0], cfg["years"][-1] + 1))
    cross  = SEASON_MONTHS[season_name][0] > SEASON_MONTHS[season_name][-1]
    t0_all = time.perf_counter()

    print(f"\n{'='*60}\n  {region_name.upper()} / {season_name}\n{'='*60}")

    # -- 1. IRI download (GCM + obs together via pycpt) ---------------
    case_dir = OUT_DIR / f"_pycpt_tmp_{region_name}_{season_name}"
    domain   = pycpt.setup(case_dir, _box(reg["predictor"]))
    t0 = time.perf_counter()
    try:
        import datetime as _dt
        Y_iri, hcast, fcast = pycpt.download_data(
            "UCSB.PRCP", None, [cfg["pycpt_model"]],
            {"fdate":               _dt.datetime(2025, sea["init_month"], 1),
             "target_first_year":   cfg["years"][0],
             "target_final_year":   cfg["years"][-1],
             "target":              sea["target"],
             "predictand_extent":   _box(reg["predictand"]),
             "predictor_extent":    _box(reg["predictor"])},
            domain, False)
    except Exception as e:
        print(f"  ERROR IRI download: {e}"); traceback.print_exc(); return None
    iri_dl_s = time.perf_counter() - t0
    print(f"  IRI dl: {iri_dl_s:.1f}s")

    # -- 2. CPT Fortran CCA -------------------------------------------
    cpt_args = {
        "transform_predictand": None, "tailoring": None,
        "cca_modes":   (cfg["cca_modes"],   cfg["cca_modes"]),
        "x_eof_modes": (cfg["x_eof"],       cfg["x_eof"]),
        "y_eof_modes": (cfg["y_eof"],       cfg["y_eof"]),
        "validation":  "crossvalidation", "drymask_threshold": None,
        "skillmask_threshold": None, "crossvalidation_window": cfg["cv_window"],
        "synchronous_predictors": True,
    }
    t0 = time.perf_counter()
    try:
        hcsts, _, skill_cpt, _, _ = pycpt.evaluate_models(
            hcast, "CCA", Y_iri, fcast, cpt_args, domain,
            [cfg["pycpt_model"]], interactive=False)
    except Exception as e:
        print(f"  ERROR CPT CCA: {e}"); traceback.print_exc(); return None
    cpt_cca_s = time.perf_counter() - t0

    skill       = skill_cpt[0]
    cpt_det_raw = hcsts[0]["deterministic"]
    cpt_prob_raw= hcsts[0]["probabilistic"]

    # Normalise CPT output dims to (year, [tercile,] lat, lon)
    cpt_det  = cpt_det_raw.rename({k: v for k, v in
                                   [("T","year"),("Y","lat"),("X","lon")]
                                   if k in cpt_det_raw.dims})
    cpt_det["year"] = years
    cpt_prob = cpt_prob_raw.rename({k: v for k, v in
                                    [("C","tercile"),("T","year"),
                                     ("Y","lat"),("X","lon")]
                                    if k in cpt_prob_raw.dims})
    cpt_prob["tercile"] = [0, 1, 2]; cpt_prob["year"] = years
    cpt_prob  = cpt_prob.transpose("year", "tercile", "lat", "lon") / 100.0

    rpss_da   = skill.get("rank_probability_skill_score")
    cpt_rpss  = float(rpss_da.values[np.isfinite(rpss_da.values) &
                                     (rpss_da.values > -900)].mean()) \
                if rpss_da is not None else np.nan
    r_da      = skill.get("pearson")
    cpt_r     = float(r_da.values[np.isfinite(r_da.values)].mean()) \
                if r_da is not None else np.nan
    print(f"  CPT: {cpt_cca_s:.1f}s  r={cpt_r:+.4f}  RPSS={cpt_rpss:+.2f}%")

    # Adapt IRI arrays to standard dims
    gcm_iri, obs_iri = _adapt_pycpt(hcast[0], Y_iri, years)

    # -- 3. Rosetta CHIRPS obs ----------------------------------------
    t0 = time.perf_counter()
    try:
        obs_ros = fetch_obs_ros(season_name, reg["predictand"],
                                cfg["years"], cfg["obs_coarsen"])
    except Exception as e:
        print(f"  ERROR Rosetta obs: {e}"); traceback.print_exc(); return None
    ros_obs_s = time.perf_counter() - t0
    # QC: area-mean r between Rosetta obs and IRI obs
    qc_yr = (np.intersect1d(obs_iri.year.values, obs_ros.year.values)
             if cross else None)
    r_obs = _corrcoef(
        obs_iri.sel(year=qc_yr).mean(["lat","lon"]).values if qc_yr is not None
        else obs_iri.mean(["lat","lon"]).values,
        obs_ros.sel(year=qc_yr).mean(["lat","lon"]).values if qc_yr is not None
        else obs_ros.mean(["lat","lon"]).values)
    print(f"  Ros obs: {ros_obs_s:.1f}s  r(IRI,Ros)={r_obs:.4f}")

    # -- 4. GCM fetch -------------------------------------------------
    t0 = time.perf_counter()
    try:
        gcm_ros = fetch_gcm_ros(season_name, reg["predictor"], cfg)
    except Exception as e:
        print(f"  ERROR GCM fetch: {e}"); traceback.print_exc(); return None
    ros_gcm_s = time.perf_counter() - t0

    # For cross-year seasons: restrict to years present in all three arrays
    if cross:
        common = sorted(set(gcm_iri.year.values) &
                        set(gcm_ros.year.values) &
                        set(obs_ros.year.values))
        years   = common
        gcm_ros = gcm_ros.sel(year=years)
        obs_ros = obs_ros.sel(year=years)

    r_gcm = _corrcoef(gcm_iri.mean(["member","lat","lon"]).values,
                      gcm_ros.mean(["member","lat","lon"]).values)
    print(f"  Ros GCM: {ros_gcm_s:.1f}s  r(IRI,Ros)={r_gcm:.4f}")

    # -- 5. AfricaS2S LOYO CV ----------------------------------------
    t0 = time.perf_counter()
    preds, levs = [], []
    for train_yrs, test_yr in ds_loyo(years, window=cfg["cv_window"]):
        m = CCAMethod(x_eof_modes=cfg["x_eof"], y_eof_modes=cfg["y_eof"],
                      cca_modes=cfg["cca_modes"], standardize=True)
        m.fit(gcm_ros.sel(year=train_yrs), obs_ros.sel(year=train_yrs))
        fcst = gcm_ros.sel(year=[test_yr]).isel(year=0, drop=True)
        preds.append(m.predict(fcst).mean("member"))
        levs.append(m.leverage(fcst))
    cv_ros      = xr.concat(preds, dim="year"); cv_ros["year"] = years
    levs_cv     = np.array(levs)
    ds_loyo_s   = time.perf_counter() - t0

    ds_r        = float(PearsonMetric().compute(cv_ros, obs_ros, spatial=True).mean())
    r_ds_cpt    = _corrcoef(
        cv_ros.sel(year=np.intersect1d(cv_ros.year.values, cpt_det.year.values))
              .mean(["lat","lon"]).values,
        cpt_det.sel(year=np.intersect1d(cv_ros.year.values, cpt_det.year.values))
               .mean(["lat","lon"]).values)
    print(f"  DS LOYO: {ds_loyo_s:.1f}s  r={ds_r:+.4f}  r(DS,CPT)={r_ds_cpt:.4f}")

    # -- 6. Tercile + RPSS (CV and in-sample) -------------------------
    t0 = time.perf_counter()
    ds_terc_cv = to_tercile_cv(cv_ros, obs_ros, method="cpt",
                               leverages=levs_cv, n_modes=cfg["x_eof"],
                               cpt_boundaries=True)
    _rpss_cv_v = RPSSMetric().compute(ds_terc_cv, obs_ros, spatial=True,
                                      loo_boundaries=False, bounded=True).values
    rpss_cv    = float(_rpss_cv_v[~np.isnan(_rpss_cv_v)].mean()) * 100

    m_is = CCAMethod(x_eof_modes=cfg["x_eof"], y_eof_modes=cfg["y_eof"],
                     cca_modes=cfg["cca_modes"], standardize=True)
    m_is.fit(gcm_ros, obs_ros)
    is_preds, is_levs = [], []
    for yr in years:
        fcst = gcm_ros.sel(year=[yr]).isel(year=0, drop=True)
        is_preds.append(m_is.predict(fcst).mean("member"))
        is_levs.append(m_is.leverage(fcst))
    cv_is     = xr.concat(is_preds, dim="year"); cv_is["year"] = years
    ds_terc_is = to_tercile_cv(cv_is, obs_ros, method="cpt",
                                leverages=np.array(is_levs), n_modes=cfg["x_eof"],
                                cpt_boundaries=True)
    _rpss_is_v = RPSSMetric().compute(ds_terc_is, obs_ros, spatial=True,
                                      loo_boundaries=False, bounded=True).values
    ds_rpss_is = float(_rpss_is_v[~np.isnan(_rpss_is_v)].mean()) * 100
    terc_s = time.perf_counter() - t0
    print(f"  RPSS IS: DS={ds_rpss_is:+.2f}%  CPT={cpt_rpss:+.2f}%  CV DS={rpss_cv:+.2f}%")

    # Clean up pycpt temp dir
    shutil.rmtree(case_dir, ignore_errors=True)

    return {
        "region": region_name, "season": season_name,
        "ds_r":        ds_r,       "cpt_r":     cpt_r,
        "ds_rpss_is":  ds_rpss_is, "cpt_rpss_is": cpt_rpss,
        "ds_rpss_cv":  rpss_cv,    "r_ds_cpt_det": r_ds_cpt,
        "r_iri_ros_obs": r_obs,    "r_iri_ros_gcm": r_gcm,
        "timing": {
            "iri_dl_s":  round(iri_dl_s, 1),  "cpt_cca_s": round(cpt_cca_s, 1),
            "ros_obs_s": round(ros_obs_s, 1),  "ros_gcm_s": round(ros_gcm_s, 1),
            "ds_loyo_s": round(ds_loyo_s, 1),  "total_s":
                round(time.perf_counter() - t0_all, 1),
        },
        # In-memory arrays (no disk I/O)
        "_arrays": {
            "cv_ros":       cv_ros,
            "cpt_det":      cpt_det,
            "ds_terc_cv":   ds_terc_cv,
            "cpt_prob":     cpt_prob,
            "obs_ros":      obs_ros,
            "obs_iri":      obs_iri,
            "gcm_ros_mean": gcm_ros.mean("member"),
            "gcm_iri_mean": gcm_iri.mean("member"),
        },
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COL_TITLES = [
    "DS det\n(CV mean)",   "CPT det\n(IS mean)",
    "DS AN prob\n(CV)",    "CPT AN prob\n(IS)",
    "Ros GCM\n(yr/mbr mean)", "IRI GCM\n(yr/mbr mean)",
    "Ros obs\n(yr mean)",  "IRI obs\n(yr mean)",
    "Metrics", "Timing (s)",
]


def _sym_norm(arr):
    v = max(float(np.nanpercentile(np.abs(arr), 97)), 0.01)
    return TwoSlopeNorm(vcenter=0, vmin=-v, vmax=v)

def _pos_norm(arr):
    v = max(float(np.nanpercentile(arr[np.isfinite(arr)], 97)), 0.01)
    return Normalize(vmin=0, vmax=v)

def _panel(ax, lon, lat, data, cmap, norm):
    ax.pcolormesh(lon, lat, data, cmap=cmap, norm=norm, shading="auto")
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

def _metrics_text(ax, res):
    ax.set_axis_off()
    lines = [("r GCM match",  f"{res['r_iri_ros_gcm']:+.4f}"),
             ("r Obs match",  f"{res['r_iri_ros_obs']:+.4f}"),
             ("r DS / CPT",   f"{res['r_ds_cpt_det']:+.4f}"),
             ("RPSS DS IS",   f"{res['ds_rpss_is']:+.1f}%"),
             ("RPSS CPT IS",  f"{res['cpt_rpss_is']:+.1f}%"),
             ("RPSS DS CV",   f"{res['ds_rpss_cv']:+.1f}%")]
    y = 0.92
    for lbl, val in lines:
        ax.text(0.05, y, lbl, transform=ax.transAxes, fontsize=7, va="top", color="#444")
        ax.text(0.95, y, val, transform=ax.transAxes, fontsize=7, va="top",
                ha="right", fontweight="bold")
        y -= 0.155

def _timing_bar(ax, res):
    t = res["timing"]
    vals   = [t["iri_dl_s"], t["ros_obs_s"], t["ros_gcm_s"],
              t["ros_obs_s"] + t["ros_gcm_s"]]
    labels = ["IRI\n(gcm+obs)", "Ros obs", "Ros GCM", "Ros\n(obs+gcm)"]
    colors = ["#3498db", "#e67e22", "#27ae60", "#8e44ad"]
    bars   = ax.barh(labels, vals, color=colors, height=0.5)
    ax.set_xlim(0, max(vals) * 1.25)
    ax.tick_params(labelsize=6); ax.set_xlabel("seconds", fontsize=6)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + max(vals) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.0f}s", va="center", fontsize=6)


def make_matrix_plot(results, title, path):
    """9-row × 10-col spatial comparison figure."""
    valid = [r for r in results if r is not None]
    if not valid:
        print("[matrix] no valid results — skipping"); return
    nrows = len(results)
    col_w = [2.6] * 8 + [2.2, 2.6]
    fig   = plt.figure(figsize=(sum(col_w) + 0.5, nrows * 2.8 + 0.8))
    gs    = GridSpec(nrows, 10, figure=fig, width_ratios=col_w,
                     hspace=0.45, wspace=0.25)
    row_axes = {}

    for row, res in enumerate(results):
        if res is None:
            for c in range(10):
                fig.add_subplot(gs[row, c]).set_visible(False)
            continue
        arr = res["_arrays"]
        def ll(da):
            return (da.lat.values if "lat" in da.dims else da.Y.values,
                    da.lon.values if "lon" in da.dims else da.X.values)

        ds_m   = arr["cv_ros"].mean("year")
        cpt_m  = arr["cpt_det"].mean("year")
        ds_an  = arr["ds_terc_cv"].isel(tercile=2).mean("year")
        cpt_an = arr["cpt_prob"].isel(tercile=2).mean("year")
        gr_m   = arr["gcm_ros_mean"].mean("year")
        gi_m   = arr["gcm_iri_mean"].mean("year")
        or_m   = arr["obs_ros"].mean("year")
        oi_m   = arr["obs_iri"].mean("year")

        panels = [
            (ds_m,  "BrBG", _sym_norm(ds_m.values)),
            (cpt_m, "BrBG", _sym_norm(cpt_m.values)),
            (ds_an, "YlGnBu", Normalize(0, 0.7)),
            (cpt_an,"YlGnBu", Normalize(0, 0.7)),
            (gr_m,  "BrBG", _sym_norm(gr_m.values)),
            (gi_m,  "BrBG", _sym_norm(gi_m.values)),
            (or_m,  "YlGnBu", _pos_norm(or_m.values)),
            (oi_m,  "YlGnBu", _pos_norm(oi_m.values)),
        ]
        axs = []
        for c, (da, cmap, norm) in enumerate(panels):
            lat, lon = ll(da)
            ax = fig.add_subplot(gs[row, c])
            _panel(ax, lon, lat, da.values, cmap, norm)
            axs.append(ax)
        axs[0].set_ylabel(f"{res['region']}\n{res['season']}", fontsize=8, labelpad=3)
        ax8 = fig.add_subplot(gs[row, 8]); _metrics_text(ax8, res); axs.append(ax8)
        ax9 = fig.add_subplot(gs[row, 9]); _timing_bar(ax9, res);   axs.append(ax9)
        row_axes[row] = axs

    # Column headers on first non-None row
    fr = next(i for i, r in enumerate(results) if r is not None)
    for c, t in enumerate(COL_TITLES):
        row_axes[fr][c].set_title(t, fontsize=8, fontweight="bold", pad=3)

    fig.suptitle(f"{title}\nDS/Rosetta vs CPT/IRI  |  CV det & in-sample terciles",
                 fontsize=9, y=1.002)
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"[matrix] saved → {path}")


def make_summary_table(results, title, path):
    """Text metrics table — one row per region × season slot."""
    rows = []
    for res in results:
        if res is None:
            rows.append(["—"] * 10); continue
        t = res["timing"]
        rows.append([
            res["region"].capitalize(), res["season"],
            f"{res['ds_r']:+.3f}",      f"{res['cpt_r']:+.3f}",
            f"{res['ds_rpss_is']:+.1f}", f"{res['cpt_rpss_is']:+.1f}",
            f"{t['ros_obs_s']:.0f} / {t['iri_dl_s']:.0f}",
            f"{t['ros_gcm_s']:.0f} / {t['iri_dl_s']:.0f}",
            f"{res['r_iri_ros_gcm']:.4f}",
            f"{res['r_iri_ros_obs']:.4f}",
        ])
    col_labels = ["Region", "Season", "r DS", "r CPT",
                  "RPSS IS\nDS %", "RPSS IS\nCPT %",
                  "t obs\nRos/IRI s", "t GCM\nRos/IRI s",
                  "r GCM\nmatch", "r Obs\nmatch"]
    fig, ax = plt.subplots(figsize=(16, 0.5 * len(rows) + 2.0))
    ax.set_axis_off()
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.auto_set_column_width(list(range(len(col_labels))))
    for c in range(len(col_labels)):
        tbl[0, c].set_facecolor("#2c3e50")
        tbl[0, c].set_text_props(color="white", fontweight="bold")
    for r in range(1, len(rows) + 1):
        bg = "#f2f2f2" if r % 2 == 0 else "white"
        for c in range(len(col_labels)):
            tbl[r, c].set_facecolor(bg)
    ax.set_title(f"{title}\nt obs / t GCM: Rosetta (s) / IRI (s)", fontsize=10, pad=8)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[table]  saved → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    for cfg_key, cfg in CONFIGS.items():
        print(f"\n{'#'*70}")
        print(f"  CONFIG: {cfg['label']}")
        print(f"  years={cfg['years']}  modes={cfg['x_eof']}/{cfg['y_eof']}/{cfg['cca_modes']}"
              f"  PSF={'yes' if cfg['use_psf'] else 'no'}")
        print(f"{'#'*70}")

        results = []
        for region in ALL_REGIONS:
            for season in ALL_SEASONS:
                res = run_slot(region, season, cfg)
                results.append(res)

        valid = [r for r in results if r is not None]
        print(f"\n  {len(valid)}/{len(results)} slots succeeded — building plots")

        make_matrix_plot(results, cfg["label"],
                         OUT_DIR / f"{cfg_key}_matrix.png")
        make_summary_table(results, cfg["label"],
                           OUT_DIR / f"{cfg_key}_table.png")

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
