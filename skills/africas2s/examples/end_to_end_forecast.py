"""End-to-end: method selection -> forecast -> honest CV verification -> report.

Inputs (build them however you like; rosetta shown for concreteness):
  gcm: (year, member, lat, lon) hindcast   obs: (year, lat, lon) observations
Years must be consecutive integers.
"""

import xarray as xr

import africas2s as ds
from africas2s.cv import loyo
from africas2s.tercile import to_tercile, to_tercile_cv

# --- Data via rosetta (optional; any xarray source works) ----------------
import rosetta

REGION = [-5, 15, 33, 48]  # Horn of Africa [lat_s, lat_n, lon_w, lon_e]
YEARS = (1993, 2016)

gcm = rosetta.fetch("c3s/ecmwf-monthly", "precip", init="2024-02", target="MAM",
                    region=REGION, hindcast=YEARS, year_index=True)["precip"]
obs = rosetta.fetch("obs/era5", "precip", region=REGION, hindcast=YEARS,
                    target="MAM", seasonal="mean")["precip"]
gcm = gcm.interp(lat=obs.lat, lon=obs.lon)  # only needed for calibrate/skill_compare,
                                            # downscale methods regrid themselves

# --- 1. Pick the best method under cross-validation ----------------------
best = ds.optimize(gcm, obs, methods=["bcsd", "cca"], cv="loyo", primary_metric="rpss")
print(f"winner: {best.method}  CV RPSS: {best.score:.4f}")

# --- 2. Production tercile forecast (full obs is fine here) --------------
tercile_fc = to_tercile(best.forecast, obs)  # (tercile, lat, lon)
ds.write_terciles(tercile_fc, "forecast.nc", title="MAM precip", method=best.method)

# --- 3. Honest CV verification -------------------------------------------
# Manual LOYO with train()/predict(). Do NOT nest optimize() in this loop
# (double CV + non-consecutive inner years).
cv_preds = []
for train_years, test_year in loyo(gcm.year.values):
    model = ds.train(best.method, gcm.sel(year=train_years), obs.sel(year=train_years))
    pred = model.predict(gcm.sel(year=test_year))
    cv_preds.append(pred.expand_dims(year=[test_year]))
cv_fcst = xr.concat(cv_preds, dim="year")            # (year, member, lat, lon)

# Held-out tercile conversion — never to_tercile(cv_fcst, obs) here (leakage).
cv_terc = to_tercile_cv(cv_fcst, obs, method="bootstrap")  # (year, tercile, lat, lon)

# --- 4. Score: probabilistic and continuous metrics separately -----------
report_prob = ds.skill(cv_terc, obs,
                       metrics=["rpss", "hss", "roc", "generalized_roc", "reliability"],
                       spatial=True)
report_det = ds.skill(cv_fcst, obs,
                      metrics=["pearson_r", "spearman", "2afc", "rmse",
                               "spread_error_ratio", "spread_error_correlation"],
                      spatial=True)
print(report_prob.to_table())
print(report_det.to_table())

# --- 5. Outputs ----------------------------------------------------------
report_prob.to_pdf("verification.pdf")       # WMO-SVSLRF report (plotting extra)
report_prob.to_geotiff("rpss.tif", "rpss")   # spatial RPSS as GeoTIFF
ds.plot_terciles(tercile_fc, title=f"MAM precip ({best.method})")
