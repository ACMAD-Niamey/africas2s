"""Multi-model ensembling with safeguards, method comparison, and reports."""

import africas2s as ds

# --- Downscale several models first --------------------------------------
results = [
    ds.optimize(gcm, obs, methods=["bcsd", "cca"])  # OptimizeResult per model
    for gcm in (ecmwf, cfsv2, geoss2s)              # each (year, member, lat, lon)
]

# --- 1. Simple uniform ensemble ------------------------------------------
ens = ds.ensemble(results, obs)                     # strategy="uniform"

# --- 2. Optimized weights with honest safeguards -------------------------
ens = ds.ensemble(
    results, obs,
    strategy="skill_weighted",
    optimize_ensemble=True,          # requires obs
    primary_metric="rpss",           # must be a LEAF metric (roc_an ok, roc not)
    cv="loyo",
    safeguards={                     # these are the defaults
        "nested_cv": True,           # inner CV keeps reported skill honest
        "shrinkage": 0.5,            # shrink weights toward uniform
        "min_effective_n": 3,        # floor on 1/sum(w^2)
        "gate": True,                # fall back to uniform if it doesn't beat uniform
    },
)
print(dict(zip(ens.member_names, ens.weights)))
print("gate passed:", ens.gate_passed, "effective N:", ens.effective_n)
# ens.pev / ens.member_contributions populate when honest CV predictions exist

# Other strategies:
#   ds.ensemble(results, obs, strategy="bma", hindcasts=[h1, h2, h3])
#   ds.ensemble(results, obs, strategy="drop_worst", n_drop=1)

# --- 3. Compare methods side by side -------------------------------------
# skill_compare does NOT regrid — all forecasts must be on the obs grid.
cmp = ds.skill_compare(
    {"bcsd": fc_bcsd, "cca": fc_cca, "ensemble": ens.forecast},
    obs,
    metrics=["rpss", "hss"],   # tercile forecasts here; continuous metrics
    spatial=True,              # need a separate call with continuous forecasts
)
print(cmp.to_table())                       # methods x metrics DataFrame
cmp.to_heatmap("comparison.png")
cmp.to_pdf("comparison.pdf", spatial_maps=True)

# --- 4. Flexible-threshold (exceedance) forecast -------------------------
pev = ds.prediction_error_variance(cv_predictions, obs)   # same year sets required
flex = ds.flex_forecast(
    det_fcst=ens.forecast.mean("member"),
    pev=pev,
    obs=obs,
    threshold=0.8,             # 80th climatological percentile
    is_percentile=True,        # False -> absolute threshold in data units
)
from africas2s.plotting import plot_exceedance_probability
plot_exceedance_probability(flex.exceedance_prob, threshold=0.8)
