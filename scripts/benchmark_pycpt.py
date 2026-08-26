"""Side-by-side benchmark: PyCPT (CPT binary + IRI data) vs AfricaS2S (our CCA + Rosetta).

Calls PyCPT step by step, inspects its outputs at every stage, and compares them
to our implementations in africas2s.methods.cca / africas2s.tercile / africas2s.metrics.rpss.

Stages compared:
  1. Input data:  PyCPT (IRI) vs Rosetta (CDS) — grid, values, units
  2. EOF loadings: CPT's x/y_eof_loadings vs CCAMethod(standardize=True)
  3. CCA predictions: CPT's deterministic CV hindcasts vs our LOYO CV
  4. Tercile probabilities: CPT's in-sample probabilistic vs our reproduction
  5. Skill scores: CPT's RPSS vs our RPSSMetric

Last verified results (2026-03-20, SEAS51c East Africa MAM):
  EOF explained var:    exact match (6 decimal places)
  EOF loadings:         |r| > 0.9999 all modes
  CCA predictions:      r = 0.9996 (most years r = 1.000000)
  Tercile probs:        r = 0.9994, MAD = 0.004
  RPSS:                 7.72% vs 7.77% (per-gridpoint r = 0.9998)

== HOW CPT/PyCPT ACTUALLY WORKS (algorithm reference) ==

1. PREPROCESSING (confirmed by matching EOF explained variances to 6 dp):
   - Center each column (subtract training mean)
   - Standardize: divide by per-column sample std (ddof=1)
     AfricaS2S: CCAMethod(standardize=True)
   - Apply sqrt(cos(lat)) area weighting per gridpoint
   - SVD of weighted+standardized anomaly matrix → EOF loadings + PC scores
   Source: CPT get_pcs() in cca.F95; stdize in settings.F95

2. CCA (canonical correlation analysis):
   - SVD of cross-product matrix (Y_pcs.T @ X_pcs) → canonical correlations (mu),
     CCA rotations (r for Y, s for X)
   - Predict: project X → normalize by svx → rotate by s → scale by mu →
     rotate by r → scale by svy → project through eofy → undo weighting → add mean
   Source: cca.F95 L111-112 (CCA), L605-638 (predict)

3. MODE AUTO-SELECTION:
   - Outer loop: LOYO CV folds. Inner loop: all (x_eof, y_eof, cca) combos.
   - Goodness index: average Kendall tau across gridpoints (igood=3, default).
   - Overfitting guard: if any canonical corr >= 1-tol, skip that (x_eof, y_eof) pair.
   - CPT selected (8, 6, 3) for SEAS51c with ranges x=(1,8), y=(1,6), cca=(1,3).
   Source: cca.F95 L223-539 (cv_cca), scores.F95 L3675-3679 (goodness)

4. LEVERAGE (per forecast case, cca.F95 L602-620):
   xvp = 1/n + (Sum(prjc(1:ncc)))^2
   prjc = CCA projections BEFORE scaling by mu.
   NOTE: square of sum, NOT sum of squares.

5. PyCPT's PROBABILISTIC HINDCASTS ARE IN-SAMPLE (not cross-validated):
   pycpt/notebook.py:396-403 calls CPT a second time, fitting on ALL years and
   passing the hindcast data as "forecast" (dates shifted +48yr to trick CPT).
   - 'deterministic' field = genuine LOYO cross-validated predictions
   - 'probabilistic' field = in-sample predictions (trained on all data)
   - 'prediction_error_variance' = PEV from this in-sample context
   The comment in PyCPT warns: "this produces unrealistically optimistic values"

6. PEV (prediction error variance):
   PEV(gridpoint, year) = s2_cv(gridpoint) * (1 + h_insample(year))
   Where:
   - s2_cv = sum(cv_residuals^2) / (n - x_eof - 1), per gridpoint, from CPT's
     internal CV (which runs for mode selection even in the "forecast" call)
   - h_insample = leverage from the in-sample (all-data) CCA fit
   Source: distribs.F95 L253-296 (get_errvar), regression.F95 L1656

7. TERCILE PROBABILITIES (regression.F95 L1753-1782):
   t1 = (threshold_33 - forecast) / PESD
   P(below) = StudentT_CDF(t1, dofr)        dofr = n - x_eof - 1
   P(above) = 1 - StudentT_CDF(t2, dofr)
   P(normal) = 1 - P(below) - P(above)
   Boundaries: full-climatology CPT q_empirical (rndx = n*p + 0.5, distribs.F95 L1007)
   AfricaS2S: to_tercile_cv(method="cpt", cpt_boundaries=True)

8. RPSS SCORING (scores.F95 L2309-2340):
   - Obs categorized against LOO tercile boundaries (year ± hcw excluded)
   - Bounded formula: when rps > rps_clim, RPSS = (rps_clim - rps) / (1 - rps_clim)
   - PyCPT's skill file stores per-gridpoint RPSS (spatial map, in percentage 0-100)
   AfricaS2S: RPSSMetric(loo_boundaries=False, bounded=True, spatial=True)
   Note: PyCPT's per-gridpoint RPSS uses full-sample boundaries, not LOO.

== KEY AFRICAS2S FLAGS TO MATCH CPT ==

  CCAMethod(standardize=True)           # center + divide by std ddof=1
  to_tercile_cv(cpt_boundaries=True)    # CPT q_empirical boundaries
  RPSSMetric(bounded=True)              # CPT's bounded RPSS formula

Prerequisites:
  bash scripts/install_pycpt.sh
  source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
  conda activate pycpt

Usage:
  python scripts/benchmark_pycpt.py
"""
import datetime as dt
import sys
from pathlib import Path
import numpy as np
import xarray as xr

try:
    import pycpt
except ImportError:
    print("ERROR: Run this in the pycpt conda env. See scripts/install_pycpt.sh")
    sys.exit(1)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "rosetta" / "src"))
sys.path.insert(0, str(REPO / "africas2s" / "src"))

from africas2s.methods.cca import CCAMethod
from africas2s.cv import loyo
from africas2s.tercile import to_tercile_cv
from africas2s.metrics.rpss import RPSSMetric, _cpt_boundaries

CASE_DIR = Path(__file__).parent / "output" / "benchmark"

# --- Config (matches PyCPT East Africa MAM exactly) ---
PREDICTOR_EXTENT = {"west": 10, "east": 75, "south": -20, "north": 20}
PREDICTAND_EXTENT = {"west": 22, "east": 52, "south": -12, "north": 15}
FORECAST_DATE = dt.datetime(2025, 2, 1)
TARGET_SEASON = "Mar-May"
FIRST_YEAR = 1993
FINAL_YEAR = 2016
CV_WINDOW = 5

PREDICTOR_NAMES = ["SEAS51c.PRCP"]
PREDICTAND_NAME = "UCSB.PRCP"

CPT_ARGS = {
    "transform_predictand": None,
    "tailoring": None,
    "cca_modes": (1, 3),
    "x_eof_modes": (1, 8),
    "y_eof_modes": (1, 6),
    "validation": "crossvalidation",
    "drymask_threshold": None,
    "skillmask_threshold": None,
    "crossvalidation_window": CV_WINDOW,
    "synchronous_predictors": True,
}


def _extract_years(da):
    """Extract integer years from a PyCPT DataArray's T coordinate."""
    t = da.coords["T"].values
    try:
        import pandas as pd
        return pd.DatetimeIndex(t).year.astype(int).tolist()
    except Exception:
        return list(range(FIRST_YEAR, FINAL_YEAR + 1))


def _adapt_pycpt_dims(gcm_iri, obs_iri):
    """Rename PyCPT dims (T/Y/X) to AfricaS2S dims (year/lat/lon)."""
    years = _extract_years(gcm_iri)
    renames = {}
    if "Y" in gcm_iri.dims:
        renames["Y"] = "lat"
    if "X" in gcm_iri.dims:
        renames["X"] = "lon"
    if "T" in gcm_iri.dims:
        renames["T"] = "year"
    gcm = gcm_iri.rename(renames)
    gcm["year"] = years
    if "member" not in gcm.dims:
        gcm = gcm.expand_dims("member")

    obs_renames = {}
    if "Y" in obs_iri.dims:
        obs_renames["Y"] = "lat"
    if "X" in obs_iri.dims:
        obs_renames["X"] = "lon"
    if "T" in obs_iri.dims:
        obs_renames["T"] = "year"
    obs = obs_iri.rename(obs_renames)
    obs["year"] = years
    return gcm, obs, years


def corrcoef_valid(a, b):
    """Pearson r between two arrays, ignoring NaN in either."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def compare(label, a, b, units=""):
    """Print comparison stats between two arrays."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        print(f"  {label}: no valid points to compare")
        return
    r = np.corrcoef(a[m], b[m])[0, 1]
    diff = np.abs(a[m] - b[m])
    print(f"  {label}: r={r:.6f}, MAD={diff.mean():.4f}{units}, "
          f"MaxAD={diff.max():.4f}{units}, n={m.sum()}")


# =========================================================================
# STEP 0: Run PyCPT
# =========================================================================

def run_pycpt():
    """Call PyCPT to fetch IRI data and run CPT. Returns all outputs."""
    print(f"PyCPT v{pycpt.__version__}")
    print(f"Config: predictor={PREDICTOR_EXTENT}, predictand={PREDICTAND_EXTENT}")
    print(f"  {FIRST_YEAR}-{FINAL_YEAR}, target={TARGET_SEASON}, cv_window={CV_WINDOW}")

    domain_dir = pycpt.setup(CASE_DIR, PREDICTOR_EXTENT)

    print("\nFetching data from IRI Data Library...")
    Y, hindcast_data, forecast_data = pycpt.download_data(
        PREDICTAND_NAME, None, PREDICTOR_NAMES,
        {"fdate": FORECAST_DATE, "target_first_year": FIRST_YEAR,
         "target_final_year": FINAL_YEAR, "target": TARGET_SEASON,
         "predictand_extent": PREDICTAND_EXTENT,
         "predictor_extent": PREDICTOR_EXTENT},
        domain_dir, False,
    )

    print("Running CCA via CPT binary...")
    hcsts, fcsts, skill, pxs, pys = pycpt.evaluate_models(
        hindcast_data, "CCA", Y, forecast_data, CPT_ARGS,
        domain_dir, PREDICTOR_NAMES, interactive=False,
    )

    return {
        "gcm": hindcast_data[0],   # X: GCM hindcast from IRI
        "obs": Y,                   # Y: CHIRPS obs from IRI
        "hcst": hcsts[0],          # deterministic + probabilistic + PEV
        "fcst": fcsts[0],          # real-time forecast
        "skill": skill[0],         # per-gridpoint skill scores
        "x_patterns": pxs[0],     # X EOF/CCA loadings, scores, explained var
        "y_patterns": pys[0],     # Y EOF/CCA loadings, scores, explained var
    }


# =========================================================================
# STEP 1: Compare input data
# =========================================================================

def step1_data(cpt):
    print("\n" + "=" * 70)
    print("STEP 1: Input Data — PyCPT (IRI) vs Rosetta (CDS)")
    print("=" * 70)

    gcm_iri = cpt["gcm"]
    obs_iri = cpt["obs"]
    print(f"\n  PyCPT GCM: {dict(gcm_iri.sizes)}")
    print(f"  PyCPT Obs: {dict(obs_iri.sizes)}")

    # Fetch same config via Rosetta
    import rosetta
    print("\n  Fetching via Rosetta...")
    gcm_ds = rosetta.fetch("c3s/ecmwf-monthly", "precip", init="2025-02",
                           target="MAM", hindcast=(FIRST_YEAR, FINAL_YEAR),
                           region=[-20, 20, 10, 75])
    obs_ds = rosetta.fetch("obs/chirps-v2-monthly", "precip",
                           hindcast=(FIRST_YEAR, FINAL_YEAR),
                           region=[-12, 15, 22, 52])

    gcm_ros = gcm_ds["precip"]
    obs_ros = obs_ds["precip"].where(obs_ds["precip"] >= 0)

    print(f"  Rosetta GCM: {dict(gcm_ros.sizes)}")
    print(f"  Rosetta Obs: {dict(obs_ros.sizes)}")

    # Compare area means over time
    iri_obs_ts = obs_iri.mean(["Y", "X"]).values if "Y" in obs_iri.dims else obs_iri.mean(["lat", "lon"]).values
    ros_obs_ts = obs_ros.sel(time=obs_ros.time.dt.month.isin([3, 4, 5])).groupby("time.year").mean("time").mean(["lat", "lon"]).values

    m = min(len(iri_obs_ts), len(ros_obs_ts))
    if m > 2:
        r = corrcoef_valid(iri_obs_ts[:m], ros_obs_ts[:m])
        print(f"\n  Obs area-mean temporal r: {r:.6f}")
        print(f"  IRI mean: {np.nanmean(iri_obs_ts):.2f}, Rosetta mean: {np.nanmean(ros_obs_ts):.4f}")
        print(f"  (IRI is mm/season, Rosetta is mm/day — ratio ~90)")

    return gcm_ros, obs_ros


# =========================================================================
# STEP 2: Compare EOF decomposition
# =========================================================================

def step2_eofs(cpt):
    print("\n" + "=" * 70)
    print("STEP 2: EOF Decomposition — CPT vs AfricaS2S (standardize=True)")
    print("=" * 70)

    gcm_iri = cpt["gcm"]
    obs_iri = cpt["obs"]
    xp = cpt["x_patterns"]

    # Extract CPT's explained variance
    cpt_x_expvar = xp["x_explained_variance"].values
    print(f"\n  CPT x_explained_var[:5]: {cpt_x_expvar[:5]}")

    # Run AfricaS2S CCA with standardize=True on PyCPT's data
    # Adapt dims
    gcm, obs, years = _adapt_pycpt_dims(gcm_iri, obs_iri)

    # Fit with max modes to compare EOF structure
    m = CCAMethod(x_eof_modes=len(years), y_eof_modes=len(years),
                  cca_modes=len(years), standardize=True)
    m.fit(gcm, obs)

    ds_x_expvar = m.svx_ ** 2 / np.sum(m.svx_ ** 2) * 100
    print(f"  DS  x_explained_var[:5]: {ds_x_expvar[:5]}")
    print(f"  Match: {np.allclose(ds_x_expvar[:5], cpt_x_expvar[:5], atol=0.001)}")

    # Compare EOF loadings
    cpt_x_eof = xp["x_eof_loadings"].values  # (Mode, Y, X)
    n_modes = min(5, cpt_x_eof.shape[0], m.eofx_.shape[1])
    print(f"\n  EOF spatial correlation (|r|, allowing sign flip):")
    for i in range(n_modes):
        ds_eof = np.full(len(m.x_valid_), np.nan)
        ds_eof[m.x_valid_] = m.eofx_[:, i]
        cpt_flat = cpt_x_eof[i].ravel()
        ds_flat = ds_eof
        mask = np.isfinite(cpt_flat) & np.isfinite(ds_flat)
        if mask.sum() > 10:
            r = abs(np.corrcoef(cpt_flat[mask], ds_flat[mask])[0, 1])
            print(f"    X EOF {i+1}: |r| = {r:.6f}")

    return gcm, obs


# =========================================================================
# STEP 3: Compare CCA predictions
# =========================================================================

def step3_cca(cpt, gcm, obs):
    print("\n" + "=" * 70)
    print("STEP 3: CCA Cross-Validated Predictions")
    print("=" * 70)

    cpt_det_raw = cpt["hcst"]["deterministic"]
    years = list(range(FIRST_YEAR, FINAL_YEAR + 1))
    # Adapt PyCPT det dims
    det_renames = {}
    if "T" in cpt_det_raw.dims:
        det_renames["T"] = "year"
    if "Y" in cpt_det_raw.dims:
        det_renames["Y"] = "lat"
    if "X" in cpt_det_raw.dims:
        det_renames["X"] = "lon"
    cpt_det = cpt_det_raw.rename(det_renames)
    cpt_det["year"] = years

    # Run AfricaS2S LOYO with standardize=True, modes 8/6/3
    print("\n  Running AfricaS2S LOYO CV (standardize=True, modes=8/6/3)...")
    preds, leverages = [], []
    for train_yrs, test_yr in loyo(years, window=CV_WINDOW):
        m = CCAMethod(x_eof_modes=8, y_eof_modes=6, cca_modes=3, standardize=True)
        m.fit(gcm.sel(year=train_yrs), obs.sel(year=train_yrs))
        forecast = gcm.sel(year=[test_yr]).isel(year=0, drop=True)
        preds.append(m.predict(forecast).mean("member"))
        leverages.append(m.leverage(forecast))
    cv = xr.concat(preds, dim="year")
    cv["year"] = years
    leverages = np.array(leverages)

    # Compare
    compare("Overall", cv.values, cpt_det.values, " mm")

    print(f"\n  {'Year':>6} {'r':>10} {'MAD':>10}")
    for yr in years:
        a = cv.sel(year=yr).values.ravel()
        b = cpt_det.sel(year=yr).values.ravel()
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 10:
            continue
        r = np.corrcoef(a[m], b[m])[0, 1]
        print(f"  {yr:>6} {r:>10.6f} {np.abs(a[m]-b[m]).mean():>10.4f}")

    return cv, leverages


# =========================================================================
# STEP 4: Compare tercile probabilities
# =========================================================================

def step4_tercile(cpt, cv, leverages, obs):
    print("\n" + "=" * 70)
    print("STEP 4: Tercile Probabilities")
    print("=" * 70)

    years = list(range(FIRST_YEAR, FINAL_YEAR + 1))

    # PyCPT's probabilistic is IN-SAMPLE (see CPT_TERCILE_AND_RPSS.md Section 0)
    cpt_prob = cpt["hcst"]["probabilistic"]
    print(f"\n  PyCPT probabilistic dims: {cpt_prob.dims}, shape: {cpt_prob.shape}")
    print(f"  NOTE: PyCPT's probabilistic field is IN-SAMPLE (not CV)")

    # Method A: AfricaS2S CV tercile (using library with cpt_boundaries=True)
    ds_terc_cv = to_tercile_cv(cv, obs, method="cpt", leverages=leverages,
                               n_modes=8, cpt_boundaries=True)

    # Adapt PyCPT prob dims for comparison
    renames = {}
    if "C" in cpt_prob.dims:
        renames["C"] = "tercile"
    if "T" in cpt_prob.dims:
        renames["T"] = "year"
    if "Y" in cpt_prob.dims:
        renames["Y"] = "lat"
    if "X" in cpt_prob.dims:
        renames["X"] = "lon"
    cpt_p = cpt_prob.rename(renames)
    cpt_p["tercile"] = [0, 1, 2]
    cpt_p["year"] = years
    cpt_p = cpt_p.transpose("year", "tercile", "lat", "lon") / 100.0

    print("\n  CV tercile (our method) vs PyCPT in-sample probabilistic:")
    compare("Overall", ds_terc_cv.values, cpt_p.values)
    print("  (Low r expected — comparing CV probs vs in-sample probs)")

    # Method B: Reproduce PyCPT's in-sample approach
    # Fit on all data, predict all years, use s2_cv for PEV
    print("\n  Reproducing PyCPT's in-sample approach...")
    from scipy.stats import t as t_dist

    n = len(years)
    residuals = cv - obs
    dofr = n - 8 - 1
    s2_cv = (residuals ** 2).sum("year").values / dofr

    # For the in-sample fit we need the original GCM data
    gcm_adapted, _, _ = _adapt_pycpt_dims(cpt["gcm"], cpt["obs"])

    m_insample = CCAMethod(x_eof_modes=8, y_eof_modes=6, cca_modes=3, standardize=True)
    m_insample.fit(gcm_adapted, obs)

    insample_preds = []
    insample_levs = []
    for yr in years:
        fcast = gcm_adapted.sel(year=[yr]).isel(year=0, drop=True)
        insample_preds.append(m_insample.predict(fcast).mean("member"))
        insample_levs.append(m_insample.leverage(fcast))
    insample_cv = xr.concat(insample_preds, dim="year")
    insample_cv["year"] = years
    insample_levs = np.array(insample_levs)

    # Tercile from in-sample preds + s2_cv*(1+h)
    t33, t67 = _cpt_boundaries(obs.values)
    terc_insample = np.full_like(cpt_p.values, np.nan)
    for i in range(n):
        loc = insample_cv.sel(year=years[i]).values
        pesd = np.sqrt(np.maximum(s2_cv * (1 + insample_levs[i]), 1e-24))
        terc_insample[i, 0] = t_dist.cdf(t33, df=dofr, loc=loc, scale=pesd)
        terc_insample[i, 2] = t_dist.sf(t67, df=dofr, loc=loc, scale=pesd)
        terc_insample[i, 1] = 1.0 - terc_insample[i, 0] - terc_insample[i, 2]

    print("\n  In-sample reproduction vs PyCPT probabilistic:")
    compare("Overall", terc_insample, cpt_p.values)
    for c, name in enumerate(["below", "normal", "above"]):
        compare(f"  {name}", terc_insample[:, c], cpt_p.values[:, c])

    return ds_terc_cv, terc_insample, cpt_p


# =========================================================================
# STEP 5: Compare RPSS
# =========================================================================

def step5_rpss(cpt, ds_terc_cv, terc_insample, cpt_p, obs):
    print("\n" + "=" * 70)
    print("STEP 5: RPSS Skill Scores")
    print("=" * 70)

    years = list(range(FIRST_YEAR, FINAL_YEAR + 1))

    # PyCPT's RPSS
    cpt_rpss = cpt["skill"]["rank_probability_skill_score"]
    cpt_rpss_vals = cpt_rpss.values
    cpt_valid = cpt_rpss_vals[(~np.isnan(cpt_rpss_vals)) & (cpt_rpss_vals > -900)]
    print(f"\n  PyCPT spatial mean RPSS: {cpt_valid.mean():+.4f}%")

    # Our RPSS from CV tercile
    rpss_cv = RPSSMetric().compute(ds_terc_cv, obs, spatial=True,
                                   loo_boundaries=False, bounded=True)
    cv_valid = rpss_cv.values[~np.isnan(rpss_cv.values)]
    print(f"  AfricaS2S CV RPSS:       {cv_valid.mean()*100:+.4f}%")

    # Our RPSS from in-sample tercile reproduction
    terc_da = xr.DataArray(terc_insample, dims=["year", "tercile", "lat", "lon"],
                           coords=cpt_p.coords)
    rpss_is = RPSSMetric().compute(terc_da, obs, spatial=True,
                                   loo_boundaries=False, bounded=True)
    is_valid = rpss_is.values[~np.isnan(rpss_is.values)]
    print(f"  AfricaS2S in-sample RPSS: {is_valid.mean()*100:+.4f}%")

    # Per-gridpoint comparison
    if "Y" in cpt_rpss.dims:
        cpt_rpss_flat = cpt_rpss.values.ravel()
    else:
        cpt_rpss_flat = cpt_rpss.values.ravel()
    is_rpss_flat = (rpss_is.values * 100).ravel()
    m = np.isfinite(is_rpss_flat) & np.isfinite(cpt_rpss_flat) & (cpt_rpss_flat > -900)
    if m.sum() > 0:
        r = np.corrcoef(is_rpss_flat[m], cpt_rpss_flat[m])[0, 1]
        print(f"\n  Per-gridpoint RPSS (in-sample): r={r:.6f}, "
              f"MAD={np.abs(is_rpss_flat[m]-cpt_rpss_flat[m]).mean():.4f}%")


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 70)
    print("BENCHMARK: PyCPT (CPT binary + IRI) vs AfricaS2S")
    print("=" * 70)

    # Step 0: Run PyCPT
    cpt = run_pycpt()
    print("  PyCPT complete.\n")

    # Step 1: Data comparison
    gcm_ros, obs_ros = step1_data(cpt)

    # Step 2: EOF comparison (using PyCPT's IRI data in both)
    gcm, obs = step2_eofs(cpt)

    # Step 3: CCA predictions
    cv, leverages = step3_cca(cpt, gcm, obs)

    # Step 4: Tercile probabilities
    ds_terc_cv, terc_insample, cpt_p = step4_tercile(cpt, cv, leverages, obs)

    # Step 5: RPSS
    step5_rpss(cpt, ds_terc_cv, terc_insample, cpt_p, obs)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
