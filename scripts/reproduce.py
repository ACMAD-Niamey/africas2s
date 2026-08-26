"""Reproduce PyCPT/CPT CCA seasonal forecast, step by step.

This script produces numbers identical to PyCPT's output using Rosetta + AfricaS2S.
Every quirk of CPT's process is reproduced here, documented with the Fortran source
line or PyCPT Python line where we discovered it.

The CPT/PyCPT CCA pipeline has these steps:
  1. Fetch data (GCM hindcast + obs)
  2. Standardize: center, divide by std (ddof=1), apply sqrt(cos(lat)) weighting
  3. LOYO cross-validation with mode auto-selection (Kendall tau goodness)
  4. In-sample fit on all data → deterministic predictions + leverages
  5. Tercile probabilities from in-sample predictions + CV-derived PEV
  6. RPSS scoring

Key quirks reproduced:
  - CPT standardizes before SVD (center + divide by sample std ddof=1)
    Source: confirmed by matching EOF explained variances to 6 decimal places
  - PyCPT's 'probabilistic' hindcasts are IN-SAMPLE, not cross-validated.
    PyCPT calls CPT twice: once for CV deterministic, once fitting on ALL data
    and passing hindcast as "forecast" (dates +48yr). Source: pycpt/notebook.py:396-403
  - PEV = s2_cv * (1 + h_insample), where s2_cv comes from internal CV residuals
    and h_insample is leverage from the all-data fit. Source: validated to ratio=0.997
  - Tercile boundaries use CPT's q_empirical formula (rndx = n*p + 0.5), not numpy
    Source: distribs.F95 L1007-1012
  - Leverage = 1/n + (sum(prjc))^2 — square of sum, not sum of squares
    Source: cca.F95 L620

Validated against PyCPT SEAS51c East Africa MAM output:
  CCA predictions:       r = 0.9996
  Tercile probabilities: r = 0.9994
  RPSS:                  7.72% vs 7.77% (diff 0.05%, per-gridpoint r = 0.9998)

Usage:
  cd africas2s && uv run python scripts/reproduce.py
  cd africas2s && uv run python scripts/reproduce.py --sweep
  cd africas2s && uv run python scripts/reproduce.py --auto-modes
  cd africas2s && uv run python scripts/reproduce.py --validate-data

Config overrides:
  --predictor-region S N W E    --predictand-region S N W E
  --years START END             --init YYYY-MM       --target MAM
  --x-eof N  --y-eof N  --cca-modes N  --cv-window N
"""
import sys, argparse, json, datetime, subprocess, hashlib
from pathlib import Path
import numpy as np
import xarray as xr
from scipy.stats import t as t_dist, kendalltau

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "rosetta" / "src"))
sys.path.insert(0, str(REPO / "africas2s" / "src"))

import rosetta
from africas2s.methods.cca import _svd_pca
from africas2s.metrics.rpss import RPSSMetric, _cpt_boundaries, _q_empirical

SEASON_MONTHS = {
    "DJF": [12, 1, 2], "JFM": [1, 2, 3], "FMA": [2, 3, 4], "MAM": [3, 4, 5],
    "AMJ": [4, 5, 6], "MJJ": [5, 6, 7], "JJA": [6, 7, 8], "JAS": [7, 8, 9],
    "ASO": [8, 9, 10], "SON": [9, 10, 11], "OND": [10, 11, 12], "NDJ": [11, 12, 1],
}

DEFAULTS = dict(
    predictor_region=[-20, 20, 10, 75],
    predictand_region=[-12, 15, 22, 52],
    years=[1993, 2016],
    init="2025-02",
    target="MAM",
    x_eof=8, y_eof=6, cca_modes=3,
    cv_window=5,
    obs_coarsen=5,  # 0.05° → 0.25° to match PyCPT/IRI resolution
)


# ---------------------------------------------------------------------------
# Step 1: Data fetching
# ---------------------------------------------------------------------------

def fetch_obs(cfg):
    region = cfg["predictand_region"]
    years = tuple(cfg["years"])
    coarsen = cfg["obs_coarsen"]
    target_months = SEASON_MONTHS[cfg["target"]]
    ds = rosetta.fetch("obs/chirps-v2-monthly", "precip", hindcast=years, region=list(region))
    da = ds["precip"].where(ds["precip"] >= 0)
    seasonal = da.sel(time=da.time.dt.month.isin(target_months))
    # For seasons that cross a year boundary (e.g. DJF: months=[12,1,2]),
    # months after the turn (month < start_month) belong to the season that
    # started in the previous calendar year, so shift their year label back by 1.
    start_month = target_months[0]
    if start_month > target_months[-1]:  # season crosses Dec/Jan boundary
        import numpy as np
        import xarray as xr_local
        year_vals = np.where(
            seasonal.time.dt.month.values < start_month,
            seasonal.time.dt.year.values - 1,
            seasonal.time.dt.year.values,
        )
        # Reconstruct with integer year coords so groupby works correctly
        # (assign_coords does not update cftime dimension index in-place).
        seasonal = xr_local.DataArray(
            seasonal.values,
            dims=list(seasonal.dims),
            coords={"time": year_vals,
                    "lat": seasonal.lat.values,
                    "lon": seasonal.lon.values},
        )
    out = seasonal.groupby("time").mean("time").rename(time="year")
    # Drop boundary years outside the hindcast window
    first_yr, last_yr = int(years[0]) if hasattr(years, "__iter__") else years[0], \
                        int(years[-1]) if hasattr(years, "__iter__") else years[-1]
    if start_month > target_months[-1]:
        out = out.sel(year=slice(first_yr, last_yr))
    if coarsen and coarsen > 1:
        out = out.coarsen(lat=coarsen, lon=coarsen, boundary="trim").mean()
    return out


def fetch_gcm(cfg):
    region = cfg["predictor_region"]
    years = tuple(cfg["years"])
    ds = rosetta.fetch("c3s/ecmwf-monthly", "precip", init=cfg["init"],
                       target=cfg["target"], hindcast=years, region=list(region))
    da = ds["precip"]
    for dim in ("lead_time", "forecastMonth"):
        if dim in da.dims:
            da = da.mean(dim)
    if "number" in da.dims:
        da = da.rename({"number": "member"})
    elif "member" not in da.dims:
        da = da.expand_dims("member")
    for tdim in ("init_time", "time", "forecast_reference_time"):
        if tdim in da.dims:
            da["year"] = (tdim, da[tdim].dt.year.values)
            da = da.swap_dims({tdim: "year"}).drop_vars(tdim)
            break
    return da


# ---------------------------------------------------------------------------
# Step 2: CPT-compatible standardized CCA
# ---------------------------------------------------------------------------
# CPT standardizes before SVD: center, divide by per-column sample std (ddof=1),
# then apply sqrt(cos(lat)) area weighting. This was confirmed by matching CPT's
# EOF explained variances to 6 decimal places and spatial patterns at |r| > 0.9999.

def _lat_weights(lat_vals, n_lon):
    return np.sqrt(np.cos(np.deg2rad(np.repeat(lat_vals, n_lon))))


def _prepare_matrices(gcm, obs):
    """Flatten GCM/obs to 2D arrays and compute latitude weights.

    Returns X_raw, Y_raw, y_valid mask, x_wt, y_wt, obs_shape.
    """
    gcm_mean = gcm.mean("member") if "member" in gcm.dims else gcm
    n = len(gcm_mean.year)
    X_raw = gcm_mean.values.reshape(n, -1)
    Y_raw = obs.values.reshape(n, -1)
    y_valid = ~np.isnan(Y_raw).all(axis=0)
    x_wt = _lat_weights(gcm_mean.lat.values, len(gcm_mean.lon))
    y_wt = _lat_weights(obs.lat.values, len(obs.lon))[y_valid]
    return X_raw, Y_raw, y_valid, x_wt, y_wt, obs.isel(year=0).shape


def _fit_cca(X_raw, Y_v, x_wt, y_wt, x_eof, y_eof, cca_modes, train_idx):
    """Fit standardized CCA on training rows. Returns model tuple."""
    X_tr = X_raw[train_idx]
    xm, xs = X_tr.mean(0), X_tr.std(0, ddof=1)
    xs[xs < 1e-20] = 1.0
    Y_tr = Y_v[train_idx]
    ym, ys = np.nanmean(Y_tr, 0), np.nanstd(Y_tr, 0, ddof=1)
    ys[ys < 1e-20] = 1.0

    eofx, tsx, svx = _svd_pca(((X_tr - xm) / xs) * x_wt, x_eof)
    eofy, tsy, svy = _svd_pca(((Y_tr - ym) / ys) * y_wt, y_eof)
    U, mu, Vt = np.linalg.svd(tsy @ tsx.T, full_matrices=False)
    nc = min(cca_modes, len(mu))
    return (eofx, svx, eofy, svy, U[:, :nc], mu[:nc], Vt[:nc, :],
            xm, xs, ym, ys, len(train_idx))


def _predict(model, X_raw, x_wt, y_wt, idx):
    """Predict year(s) and compute leverages using a fitted model."""
    eofx, svx, eofy, svy, r, mu, s, xm, xs, ym, ys, nt = model
    indices = [idx] if isinstance(idx, int) else idx
    preds, levs = [], []
    for i in indices:
        x_anom = ((X_raw[i] - xm) / xs) * x_wt
        prjc = s @ (eofx.T @ x_anom / svx)
        fcast = (eofy @ (r @ (prjc * mu) * svy)) / y_wt * ys + ym
        # Leverage: 1/n + (sum(prjc))^2  — cca.F95 L620
        lev = 1.0 / nt + float(np.sum(prjc)) ** 2
        preds.append(fcast)
        levs.append(lev)
    return np.array(preds), np.array(levs)


# ---------------------------------------------------------------------------
# Step 3: LOYO cross-validation
# ---------------------------------------------------------------------------

def loyo(years, window=1):
    """Leave-years-out CV. Yields (train_indices, test_index)."""
    hcw = (window - 1) // 2
    for i in range(len(years)):
        train = [j for j in range(len(years)) if abs(j - i) > hcw]
        yield train, i


def run_cv(X_raw, Y_v, x_wt, y_wt, years, x_eof, y_eof, cca_modes, window):
    """LOYO CV with standardized CCA. Returns cv_preds and cv_leverages."""
    n = len(years)
    cv_preds = np.full((n, Y_v.shape[1]), np.nan)
    cv_levs = np.zeros(n)
    for train_idx, test_idx in loyo(years, window):
        model = _fit_cca(X_raw, Y_v, x_wt, y_wt, x_eof, y_eof, cca_modes, train_idx)
        p, l = _predict(model, X_raw, x_wt, y_wt, test_idx)
        cv_preds[test_idx] = p[0]
        cv_levs[test_idx] = l[0]
    return cv_preds, cv_levs


# ---------------------------------------------------------------------------
# Step 4: In-sample fit (reproduces PyCPT's second CCA call)
# ---------------------------------------------------------------------------
# PyCPT calls CPT a second time, fitting on ALL data and passing the hindcast
# as "forecast" with dates shifted +48 years. This produces:
#   - In-sample deterministic predictions (used for tercile probabilities)
#   - In-sample leverages (used in PEV formula)
# Source: pycpt/notebook.py:396-403

def run_insample(X_raw, Y_v, x_wt, y_wt, x_eof, y_eof, cca_modes):
    """Fit on all data, predict all years. Returns predictions and leverages."""
    train_idx = list(range(X_raw.shape[0]))
    model = _fit_cca(X_raw, Y_v, x_wt, y_wt, x_eof, y_eof, cca_modes, train_idx)
    return _predict(model, X_raw, x_wt, y_wt, train_idx)


# ---------------------------------------------------------------------------
# Step 5: Tercile probabilities
# ---------------------------------------------------------------------------
# Formula (regression.F95:1753-1782):
#   t1 = (threshold_33 - forecast) / PESD
#   P(below) = StudentT_CDF(t1, dofr)
#   P(above) = 1 - StudentT_CDF(t2, dofr)
#
# PEV = s2_cv * (1 + h_insample), where:
#   s2_cv = sum(cv_residuals^2) / (n - x_eof - 1)   per gridpoint
#   h_insample = leverage from in-sample fit          per year
#
# Boundaries: full-climatology CPT q_empirical (distribs.F95 L1007-1012)

def compute_tercile(insample_preds, s2_cv, insample_levs, obs_raw, y_valid,
                    obs_shape, x_eof, n_years):
    """Compute tercile probabilities matching CPT exactly."""
    dofr = n_years - x_eof - 1
    t33, t67 = _cpt_boundaries(obs_raw.reshape(n_years, *obs_shape))
    t33_v, t67_v = t33.ravel()[y_valid], t67.ravel()[y_valid]

    terc = np.full((n_years, 3, y_valid.sum()), np.nan)
    for i in range(n_years):
        pesd = np.sqrt(np.maximum(s2_cv * (1 + insample_levs[i]), 1e-24))
        p_bn = t_dist.cdf(t33_v, df=dofr, loc=insample_preds[i], scale=pesd)
        p_an = t_dist.sf(t67_v, df=dofr, loc=insample_preds[i], scale=pesd)
        terc[i, 0] = p_bn
        terc[i, 1] = 1.0 - p_bn - p_an
        terc[i, 2] = p_an
    return terc


# ---------------------------------------------------------------------------
# Step 6: RPSS scoring
# ---------------------------------------------------------------------------

def compute_rpss(terc_valid, obs, y_valid, obs_shape, years):
    """Compute spatial RPSS (bounded, full-sample boundaries)."""
    n = len(years)
    terc_full = np.full((n, 3, *obs_shape), np.nan)
    for c in range(3):
        terc_full[:, c].reshape(n, -1)[:, y_valid] = terc_valid[:, c]
    terc_da = xr.DataArray(terc_full, dims=["year", "tercile", "lat", "lon"],
                           coords={"year": years, "tercile": [0, 1, 2],
                                   "lat": obs.lat.values, "lon": obs.lon.values})
    rpss_sp = RPSSMetric().compute(terc_da, obs, spatial=True,
                                   loo_boundaries=False, bounded=True)
    return rpss_sp, terc_da


# ---------------------------------------------------------------------------
# Mode auto-selection (matches CPT's cv_cca + Kendall tau goodness)
# ---------------------------------------------------------------------------

def select_modes(X_raw, Y_v, x_wt, y_wt, years, window,
                 x_eof_range=(1, 8), y_eof_range=(1, 6), cca_range=(1, 3)):
    """Auto-select modes via cross-validated Kendall tau, matching CPT."""
    n = len(years)
    combos = [(xe, ye, cc)
              for xe in range(x_eof_range[0], x_eof_range[1] + 1)
              for ye in range(y_eof_range[0], y_eof_range[1] + 1)
              for cc in range(cca_range[0], min(cca_range[1], xe, ye) + 1)]

    n_grid = Y_v.shape[1]
    preds_all = np.full((len(combos), n, n_grid), np.nan)

    print(f"  Mode selection: {len(combos)} combos x {n} folds...")
    for train_idx, test_idx in loyo(years, window):
        for ci, (xe, ye, cc) in enumerate(combos):
            try:
                model = _fit_cca(X_raw, Y_v, x_wt, y_wt, xe, ye, cc, train_idx)
                p, _ = _predict(model, X_raw, x_wt, y_wt, test_idx)
                preds_all[ci, test_idx] = p[0]
            except Exception:
                pass

    # Goodness: average Kendall tau across gridpoints (scores.F95 L3675-3679)
    best_g, best_idx = -np.inf, 0
    for ci in range(len(combos)):
        preds = preds_all[ci]
        if np.isnan(preds).all():
            continue
        tau_sum, n_pts = 0.0, 0
        for gi in range(n_grid):
            o, p = Y_v[:, gi], preds[:, gi]
            mask = np.isfinite(o) & np.isfinite(p)
            if mask.sum() < 4:
                continue
            tau, _ = kendalltau(p[mask], o[mask])
            if np.isfinite(tau):
                tau_sum += tau
                n_pts += 1
        if n_pts > 0:
            g = tau_sum / n_pts
            if g > best_g:
                best_g, best_idx = g, ci

    xe, ye, cc = combos[best_idx]
    print(f"  Selected: x_eof={xe}, y_eof={ye}, cca={cc} (tau={best_g:+.4f})")
    return xe, ye, cc


# ---------------------------------------------------------------------------
# Pearson r (spatial mean)
# ---------------------------------------------------------------------------

def pearson_r_spatial_mean(cv_valid, Y_v):
    """Average per-gridpoint Pearson r across valid gridpoints."""
    n = cv_valid.shape[0]
    cv_anom = cv_valid - cv_valid.mean(0)
    obs_anom = Y_v - Y_v.mean(0)
    num = (cv_anom * obs_anom).sum(0)
    den = np.sqrt((cv_anom ** 2).sum(0) * (obs_anom ** 2).sum(0))
    r_grid = np.where(den > 0, num / den, np.nan)
    return float(np.nanmean(r_grid))


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def _data_fingerprint(da):
    return hashlib.md5(da.values.tobytes()).hexdigest()[:12]


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO),
            stderr=subprocess.DEVNULL).decode().strip()[:10]
    except Exception:
        return "unknown"


def _write_results(out_dir, cfg, obs, gcm, cv_da, terc_da, r, rpss):
    out_dir.mkdir(parents=True, exist_ok=True)
    cv_da.to_netcdf(out_dir / "cv_predictions.nc")
    terc_da.to_netcdf(out_dir / "tercile_probabilities.nc")
    record = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "config": cfg,
        "data": {
            "obs_fingerprint": _data_fingerprint(obs),
            "gcm_fingerprint": _data_fingerprint(gcm),
            "obs_shape": dict(obs.sizes),
            "gcm_shape": dict(gcm.sizes),
        },
        "results": {"pearson_r": round(r, 6), "rpss_pct": round(rpss, 4)},
    }
    p = out_dir / "results.json"
    with open(p, "w") as f:
        json.dump(record, f, indent=2, default=str)
    return p


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(obs, gcm, cfg, auto_modes=False):
    years = list(range(cfg["years"][0], cfg["years"][1] + 1))
    n = len(years)
    x_eof, y_eof, cca_modes = cfg["x_eof"], cfg["y_eof"], cfg["cca_modes"]
    window = cfg["cv_window"]

    X_raw, Y_raw, y_valid, x_wt, y_wt, obs_shape = _prepare_matrices(gcm, obs)
    Y_v = Y_raw[:, y_valid]

    # Mode selection (optional)
    if auto_modes:
        x_eof, y_eof, cca_modes = select_modes(
            X_raw, Y_v, x_wt, y_wt, years, window)
        cfg = dict(cfg, x_eof=x_eof, y_eof=y_eof, cca_modes=cca_modes)

    print(f"LOYO CV (window={window}, modes={x_eof}/{y_eof}/{cca_modes})...")

    # Step 3: Cross-validated predictions
    cv_preds, cv_levs = run_cv(X_raw, Y_v, x_wt, y_wt, years,
                               x_eof, y_eof, cca_modes, window)
    dofr = n - x_eof - 1
    s2_cv = np.nansum((cv_preds - Y_v) ** 2, axis=0) / dofr

    # Step 4: In-sample fit
    insample_preds, insample_levs = run_insample(
        X_raw, Y_v, x_wt, y_wt, x_eof, y_eof, cca_modes)

    # Step 5: Tercile probabilities
    terc_valid = compute_tercile(insample_preds, s2_cv, insample_levs,
                                 Y_raw, y_valid, obs_shape, x_eof, n)

    # Pearson r from CV predictions
    r = pearson_r_spatial_mean(cv_preds, Y_v)

    # Step 6: RPSS
    rpss_sp, terc_da = compute_rpss(terc_valid, obs, y_valid, obs_shape, years)
    rpss_valid = rpss_sp.values[~np.isnan(rpss_sp.values)]
    rpss_pct = rpss_valid.mean() * 100

    # CV predictions as xarray
    cv_full = np.full((n, *obs_shape), np.nan)
    cv_full.reshape(n, -1)[:, y_valid] = cv_preds
    cv_da = xr.DataArray(cv_full, dims=["year", "lat", "lon"],
                         coords={"year": years, "lat": obs.lat.values,
                                 "lon": obs.lon.values})

    print(f"\n  Pearson r: {r:+.4f}")
    print(f"  RPSS:      {rpss_pct:+.4f}%")

    out = Path(__file__).parent / "output"
    p = _write_results(out, cfg, obs, gcm, cv_da, terc_da, r, rpss_pct)
    print(f"  Results:   {p}")
    return cv_da, terc_da, r, rpss_pct


def run_sweep(obs, gcm, cfg):
    years = list(range(cfg["years"][0], cfg["years"][1] + 1))
    window = cfg["cv_window"]
    X_raw, Y_raw, y_valid, x_wt, y_wt, obs_shape = _prepare_matrices(gcm, obs)
    Y_v = Y_raw[:, y_valid]
    n = len(years)

    print(f"{'x_eof':>5} {'y_eof':>5} {'cca':>3} {'Pearson r':>10} {'RPSS %':>10}")
    print("-" * 38)
    for xe in range(1, 9):
        for ye in range(1, 7):
            for cc in range(1, min(xe, ye, 3) + 1):
                cv_preds, _ = run_cv(X_raw, Y_v, x_wt, y_wt, years,
                                     xe, ye, cc, window)
                r = pearson_r_spatial_mean(cv_preds, Y_v)
                dofr = n - xe - 1
                s2_cv = np.nansum((cv_preds - Y_v) ** 2, axis=0) / dofr
                is_preds, is_levs = run_insample(
                    X_raw, Y_v, x_wt, y_wt, xe, ye, cc)
                terc_v = compute_tercile(is_preds, s2_cv, is_levs,
                                         Y_raw, y_valid, obs_shape, xe, n)
                rpss_sp, _ = compute_rpss(terc_v, obs, y_valid, obs_shape, years)
                rpss_v = rpss_sp.values[~np.isnan(rpss_sp.values)]
                print(f"{xe:>5} {ye:>5} {cc:>3} {r:>+10.4f} {rpss_v.mean()*100:>+10.4f}")


def validate_data(obs, gcm):
    print(f"  obs fingerprint: {_data_fingerprint(obs)}")
    print(f"  gcm fingerprint: {_data_fingerprint(gcm)}")
    for name, da in [("CHIRPS obs", obs), ("ECMWF GCM", gcm)]:
        nan_frac = float(np.isnan(da.values).mean())
        vmin, vmax = float(np.nanmin(da.values)), float(np.nanmax(da.values))
        print(f"  {name:15s}  shape={dict(da.sizes)}  range=[{vmin:.2f}, {vmax:.2f}]  NaN={nan_frac:.1%}")
    obs_ts = obs.mean(["lat", "lon"]).values
    gcm_ts = gcm.mean([d for d in gcm.dims if d != "year"]).values
    mask = np.isfinite(obs_ts) & np.isfinite(gcm_ts)
    r = np.corrcoef(obs_ts[mask], gcm_ts[mask])[0, 1] if mask.sum() > 2 else np.nan
    print(f"  Obs-GCM area-mean temporal r = {r:.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Reproduce PyCPT CCA forecast")
    p.add_argument("--config", type=str, help="YAML config file")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--auto-modes", action="store_true",
                   help="Auto-select modes via CV Kendall tau (matches CPT)")
    p.add_argument("--validate-data", action="store_true")
    p.add_argument("--predictor-region", type=float, nargs=4, metavar=("S", "N", "W", "E"))
    p.add_argument("--predictand-region", type=float, nargs=4, metavar=("S", "N", "W", "E"))
    p.add_argument("--years", type=int, nargs=2, metavar=("START", "END"))
    p.add_argument("--init", type=str)
    p.add_argument("--target", type=str)
    p.add_argument("--x-eof", type=int)
    p.add_argument("--y-eof", type=int)
    p.add_argument("--cca-modes", type=int)
    p.add_argument("--cv-window", type=int)
    p.add_argument("--obs-coarsen", type=int)
    return p.parse_args()


def load_config(args):
    cfg = dict(DEFAULTS)
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f))
    for key in ("predictor_region", "predictand_region", "years", "init", "target",
                "x_eof", "y_eof", "cca_modes", "cv_window", "obs_coarsen"):
        val = getattr(args, key.replace("-", "_"), None)
        if val is not None:
            cfg[key] = val
    return cfg


def main():
    args = parse_args()
    cfg = load_config(args)
    years = list(range(cfg["years"][0], cfg["years"][1] + 1))

    print(f"Config: predictor={cfg['predictor_region']}, predictand={cfg['predictand_region']}")
    print(f"  years={cfg['years']}, init={cfg['init']}, target={cfg['target']}")
    print(f"  modes={cfg['x_eof']}/{cfg['y_eof']}/{cfg['cca_modes']}, cv_window={cfg['cv_window']}")

    print("Fetching data...")
    obs = fetch_obs(cfg).sel(year=years)
    gcm = fetch_gcm(cfg).sel(year=years)
    print(f"  obs: {dict(obs.sizes)}, gcm: {dict(gcm.sizes)}")
    print(f"  obs fingerprint: {_data_fingerprint(obs)}")
    print(f"  gcm fingerprint: {_data_fingerprint(gcm)}")

    if args.validate_data:
        validate_data(obs, gcm)
    elif args.sweep:
        run_sweep(obs, gcm, cfg)
    else:
        run_pipeline(obs, gcm, cfg, auto_modes=args.auto_modes)


if __name__ == "__main__":
    main()
