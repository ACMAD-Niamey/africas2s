"""One-call PyCPT-style multi-model ensemble with seasonal_mme().

predictor_tracks maps track name -> {model name: (hindcast, forecast_or_None)}.
Hindcasts are (year, member, lat, lon); forecasts (member, lat, lon) or None
(None = the last hindcast year is held out as the forecast).
"""

import africas2s as ds

# Single-track, single-model (minimal):
result = ds.seasonal_mme(
    {"prcp": {"ECMWF": (gcm, None)}},
    obs,
    method="cca",
    cv="loyo",
)

# Multi-model, dual-track (precip + SST predictors), with CPT-style knobs:
result = ds.seasonal_mme(
    {
        "prcp": {"ECMWF": (ecmwf_prcp_hcst, ecmwf_prcp_fcst),
                 "CFSv2": (cfsv2_prcp_hcst, cfsv2_prcp_fcst)},
        "sst":  {"ECMWF": (ecmwf_sst_hcst, ecmwf_sst_fcst)},
    },
    obs,
    method="cca",
    cv="loyo",
    cpt_args={
        "n_modes": 5,
        "standardize": True,
        "mode_selection": "auto",       # CV Kendall-tau mode search ("cpt" = fixed)
        "crossvalidation_window": 5,
    },
    tercile_method="cpt",                # default for CCA: Student-t w/ leverage PEV
    probability_aggregation="pooled",    # or "cpt_per_model" (CCA only)
    forecast_year=2024,
    skill_metrics=["rpss", "roc", "reliability"],
)

# --- Results -------------------------------------------------------------
result.forecast           # deterministic MME mean (lat, lon)
result.tercile_forecast   # (tercile, lat, lon), sums to 1
result.tercile_cv         # (year, tercile, lat, lon) honest CV terciles
result.skill_report       # SkillReport -> .scores, .spatial, .to_pdf(...)
result.ensemble_result    # EnsembleResult -> .weights, .member_names
result.pev                # (lat, lon) prediction error variance (or None)
result.metadata           # years_used, forecast_year, tercile_method, n_members, ...

print(result.metadata["years_used"], result.skill_report.scores)
result.skill_report.to_pdf("mme_verification.pdf")
ds.write_terciles(result.tercile_forecast, "mme_forecast.nc", title="MAM MME")

# Notes:
# - Requires >= 5 intersection years across obs and all hindcasts.
# - native_years=True calibrates each model on its own obs-overlap years.
# - method="corrdiff" raises NotImplementedError (V1 is deterministic-only).
