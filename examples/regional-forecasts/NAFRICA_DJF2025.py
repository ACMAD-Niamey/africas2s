# %% [markdown]
# <p><img src="assets/logos/acmad.png" alt="ACMAD — African Centre of Meteorological Applications for Development, which serves the NA-RCC / PRESANORD products" title="ACMAD — African Centre of Meteorological Applications for Development, which serves the NA-RCC / PRESANORD products" height="52"></p>
#
# # NA-RCC DJF 2025/26 — North Africa seasonal forecast with acmadDL and africas2s
#
# The North Africa RCC (NA-RCC, serving the Maghreb–Sahara region and the
# **PRESANORD** forum) December–February 2025/26 rainfall outlook, built end to
# end with [`acmadDL`](https://github.com/ACMAD-Niamey/acmadDL)
# for data, [`africas2s`](https://github.com/ACMAD-Niamey/africas2s) for
# statistics and maps. This is one of five regional demonstration notebooks,
# one per continental RCC, all sharing the same structure:
#
# 1. **Setup and data** — the region, the model roster, the observations, the
#    dry mask and the colour language, and every model's raw forecast;
# 2. **CCA** — each model calibrated by canonical correlation analysis, the
#    equal-weight objective combination, three forecast windows (**DJF 2025/26,
#    JFM 2026, FMA 2026**) at two working resolutions (**0.5° and 0.1°**);
# 3. **Skill and expert adjustment** — cross-validated per-model skill maps,
#    and the objective re-combined without the lowest-skill model;
# 4. **Analogue years** — the past years whose El Niño/La Niña evolution most
#    resembles this one, and what fell in their DJF seasons;
# 5. **Onset** — when the winter rains begin, from the daily forecast streams.
#
# **Season: DJF 2025/26, from the November 2025 initialisations.** North
# Africa's rains are Mediterranean-winter rains: the wet season runs roughly
# October–April along the Maghreb coast and the Atlas, and December–February is
# its core — which makes DJF the region's best-match 3-month outlook window.
# The three forecast windows slide forward from the season: DJF 2025/26, JFM
# and FMA 2026, all from the same 1 November initialisation. November 2025 is
# the most recent complete cycle for this season at the time of writing; change
# `INIT_MONTH` to re-run a later one.

# %% [markdown]
# ### 0. Preliminaries
#
# #### 0.1 Imports
#
# Everything imported once, up front. `acmaddl` is the data layer (one
# `fetch()` over every provider, returning normalised xarray with
# `lat`/`lon`/`year`/`member` coordinates and known units); `africas2s` is the
# statistics and plotting layer. This notebook defines **no plotting functions
# and no colour ramps** of its own — every map is drawn through the packaged
# RCC display styles.
#
# The C3S seasonal datasets are served by the Copernicus CDS
# (`https://cds.climate.copernicus.eu/api`); make sure your `~/.cdsapirc` points
# there (see the README prerequisites).

# %%
import os
import warnings
warnings.filterwarnings("ignore", message="Degrees of freedom")
import datetime as dt
from dataclasses import replace
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

os.environ.setdefault("CDSAPI_URL", "https://cds.climate.copernicus.eu/api")

import acmaddl
import africas2s
from africas2s import FieldScale
from africas2s import aggregations as agg

warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="daily data starts")   # daily fields begin 2 days after init
warnings.filterwarnings("ignore", message="daily data ends")     # dry-spell guard runs to the data edge
try:  # executed-notebook hygiene: don't tight-crop cartopy figures
    get_ipython().run_line_magic("config", "InlineBackend.print_figure_kwargs = {'bbox_inches': None}")
except NameError:
    pass

FIG = Path("outputs/NAFRICA_DJF2025")
FIG.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ### 1. Setup and data
#
# #### 1.1 Parameters
#
# The forecast: **DJF 2025/26 from the November 2025 initialisation**, over
# the NA-RCC domain. The country list is the member states: Algeria,
# Egypt, Libya, Mauritania, Morocco, Tunisia and Western Sahara — the last
# listed twice because Natural Earth spells it "W. Sahara", and the country
# clip matches on any of the spellings. Two working resolutions run through
# the whole notebook: **0.5°** — the operational grid the RCC products use —
# and **0.1° (~10 km)**, the high-resolution leg.
#
# *A note on the year boundary.* DJF crosses the calendar year: **DJF 2025
# means December 2025 – February 2026**, and the stack labels a wraparound
# season by the year of its **starting** month (the wrapped months are shifted
# back a year before grouping, and incomplete seasons at the record ends are
# dropped — behaviour pinned by acmadDL's test suite). Forecast fields are
# year-labelled by their **initialisation** year, which for DJF coincides with
# the season label (a November 2025 init targets DJF 2025). The JFM and FMA
# windows, though, fall wholly in the **next** calendar year — their fields
# are relabelled to the season's own year (2026) in §1.2 so model and observed
# years pair correctly, and every figure says which year its window is in.
#
# Three forecast **windows** slide forward from the season: DJF 2025/26, JFM
# 2026 and FMA 2026, every one from the same `2025-11` initialisation, so the
# reader sees the forecast evolve across the lead times of a single issuance.
#
# *The dry mask is a parameter.* A cell is greyed where the season contributes
# less than `DRY_SHARE` of its annual rainfall — ICPAC's seasonal-share rule,
# inherited by the whole notebook family. A 15 % threshold is used here, in the style of ICPAC's seasonal
# products; change the one number below and every map
# follows. It is a display choice applied by the style, never a change to the
# calculation. For this domain, expect the mask to remove a **large share of
# the map**: much of the interior is hyper-arid, and along the Saharan fringe
# of Mauritania and southern Algeria what little rain falls comes with the
# summer monsoon, not in winter — §1.5 prints the count. The winter-wet
# Mediterranean and Atlantic margins (the Maghreb coast, the Atlas, the
# Cyrenaican and Egyptian coastal strips) are where DJF is the rainy season
# and where the forecast carries its meaning.

# %%
NAFRICA_COUNTRIES = [
    "Algeria", "Egypt", "Libya", "Mauritania", "Morocco", "Tunisia",
    "Western Sahara", "W. Sahara",
]
NAFRICA_REGION = (14.0, 38.0, -18.0, 37.0)   # lat_s, lat_n, lon_w, lon_e
EXTENT         = (NAFRICA_REGION[2], NAFRICA_REGION[3], NAFRICA_REGION[0], NAFRICA_REGION[1])

SEASON       = "DJF"                      # the target season (first window)
WINDOWS      = ["DJF", "JFM", "FMA"]      # the three forecast windows, same init
SEASON_DAYS  = 31 + 31 + 28               # Dec + Jan + Feb
INIT_MONTH   = "2025-11"                  # 1 November 2025 initialisations
YEAR         = 2025                       # DJF 2025 = Dec 2025 - Feb 2026

# JFM/FMA begin after New Year: relabel those windows to their own calendar
# year so model years pair with observed years (DJF needs no shift — both are
# labelled by December's year).
WIN_SHIFT  = {"DJF": 0, "JFM": 1, "FMA": 1}
FCST_YEAR  = {w: YEAR + WIN_SHIFT[w] for w in WINDOWS}
WIN_LABEL  = {"DJF": "DJF 2025/26", "JFM": "JFM 2026", "FMA": "FMA 2026"}

RES_LO = 0.5                              # operational grid
RES_HI = 0.1                              # high-resolution leg (~10 km)

DRY_SHARE = 0.15                          # share of annual rainfall below which a cell is greyed

# %% [markdown]
# #### 1.2 The model roster
#
# The 10-system base roster shared by the whole notebook family: five NMME
# systems (IRI/CCSR, no credentials) and five C3S systems (Copernicus CDS).
# Each row is `(label, acmadDL product, hindcast window)`; the window is the
# model's native reforecast period clipped to **1993–2020**, so every CCA is
# trained on the years that model actually has.
#
# Real-time availability drifts from month to month — a model may have a full
# hindcast and no November 2025 forecast, or serve an all-fill field for one
# target — so the roster is a *request*, and the fetch below reports which
# models actually made it in and why the others did not. Two rows need a word
# for this cycle:
#
# - **GFDL-SPEAR moved to quarterly issuance in 2025.** November is not one of
#   its issuance months, so SPEAR is expected to be *absent* from this
#   forecast — the row stays in the roster and the drop is reported with its
#   reason, exactly as it would be in operations.
# - **JMA is CPS3 here, not CPS4.** C3S retired system 3 in favour of system 4
#   around January 2026; a November **2025** initialisation therefore comes
#   from `c3s/jma` (system 3), with its reforecast clipped to 1993–2020. The
#   2026-init notebooks in this family use `c3s/jma-cps4` instead.
#
# **Why the hindcast windows differ between models.** They differ *by design*, for one of two reasons:
# **what the archive holds** — NASA-GEOSS2S's hindcast ends in 2017 and
# ECMWF's in 2016, so those are hard stops — or **a harmonisation choice**:
# the roster is floored at **1993**, the year every C3S reforecast begins, so
# the NMME systems are not trained on years their C3S counterparts cannot
# match. A model whose range looks short against its neighbour's is showing
# one of those two things, not missing data. The §1.4 coverage bars below show
# the same picture visually.

# %%
MODELS = [
    ("CanSIPS-IC4",      "nmme/cansipsic4",   (1993, 2020)),
    ("NASA-GEOSS2S",     "nmme/geoss2s",      (1993, 2017)),
    ("GFDL-SPEAR",       "nmme/spear",        (1993, 2020)),   # quarterly since 2025; November is off-cycle
    ("COLA-RSMAS-CCSM4", "nmme/ccsm4",        (1993, 2020)),
    ("NCEP-CFSv2",       "nmme/cfsv2",        (1993, 2020)),
    ("ECMWF",            "c3s/ecmwf-monthly", (1993, 2016)),
    ("CMCC",             "c3s/cmcc-sps4",     (1993, 2020)),
    ("DWD",              "c3s/dwd",           (1993, 2020)),
    ("JMA",              "c3s/jma",           (1993, 2020)),   # CPS3 — the system serving Nov 2025
    ("Meteo_France",     "c3s/meteofrance",   (1993, 2020)),
]

# %% [markdown]
# CCA needs a **wide** predictor field, because its value is finding a remote
# pattern that leads local rainfall; the predictand stays the NA-RCC box. For
# this notebook the predictor belt is **30°S–50°N**, the whole longitude
# circle: the tropical half carries ENSO and the tropical Atlantic, and the
# extension to 50°N keeps the North Atlantic / Mediterranean storm-track
# sector — the circulation that actually delivers North Africa's winter rain —
# inside the field the CCA can draw on. `acmaddl.assemble` returns each model
# as a `(hindcast, forecast)` pair of `(year, member, lat, lon)` seasonal
# fields, the shape `africas2s` consumes; one assemble per model per window,
# and the same pairs feed both the raw-anomaly matrix below and the CCA
# section. The JFM/FMA pairs are relabelled to their own calendar year here
# (§1.1's note), so every later step pairs model years with observed years
# without special cases.
#
# A provider can answer a request with a field that is structurally fine but
# empty in substance — every cell a fill value, every cell zero, a few absurd
# numbers — and *which* model does so changes from month to month.
# `acmaddl.usable` is the per-field guard: it strips fill and non-finite
# cells, then requires what remains to be non-empty, not all zero, and within
# the plausible range for precipitation. A model that fails it, or whose
# forecast is not yet available, is dropped **with a stated reason** rather
# than let into the calibration.

# %%
CCA_DOMAIN = (-30.0, 50.0, -180.0, 180.0)   # lat_s, lat_n, lon_w, lon_e

models, dropped = {w: {} for w in WINDOWS}, {w: [] for w in WINDOWS}
for label, product, window in MODELS:
    for w in WINDOWS:
        try:
            pair = acmaddl.assemble([(label, product, window)], "precip",
                                    init=INIT_MONTH, target=w,
                                    region=list(CCA_DOMAIN), seasonal="mean",
                                    boundary="center", verbose=False)[label]
            if not all(acmaddl.usable(x, variable="precip") for x in pair):
                raise ValueError("field is empty or implausible after removing fill")
            if WIN_SHIFT[w]:
                pair = tuple(x.assign_coords(year=x["year"] + WIN_SHIFT[w]) for x in pair)
            models[w][label] = pair
        except Exception as e:
            dropped[w].append(f"{label}: {type(e).__name__}: {str(e)[:70]}")
for w in WINDOWS:
    for line in dropped[w]:
        print(f"dropped [{w}] ", line)
    print(f"{w}: {len(models[w])} of {len(MODELS)} models available: {sorted(models[w])}")

# %% [markdown]
# #### 1.3 Observations
#
# CHIRPS v2 monthly rainfall, averaged over each forecast window for every
# year 1982–2020, on both working grids (`seasonal="mean"` gives the mean
# monthly total for the season, in mm/month; `boundary="cover"` keeps every cell the region
# touches). DJF years are wraparound years — the 2020 entry is Dec 2020 –
# Feb 2021 — labelled by the starting December, as §1.1 describes; the fetch
# therefore runs **through 2021**, so the last wraparound season is complete
# rather than silently dropped at the record edge (incomplete seasons are
# dropped by convention). This is the predictand every CCA is trained against
# and the record the analogue section reads from. The annual total (same
# grid, averaged over years) is what the dry-season share is measured
# against.

# %%
obs_lo, obs_hi = {}, {}
for w in WINDOWS:
    for res, store in ((RES_LO, obs_lo), (RES_HI, obs_hi)):
        store[w] = acmaddl.fetch("obs/chirps-v2-monthly", "precip", target=w,
                                 region=list(NAFRICA_REGION), hindcast=(1982, 2021),
                                 seasonal="mean", grid_res=res, boundary="cover",
                                 verbose=False)["precip"]
print({w: dict(obs_lo[w].sizes) for w in WINDOWS}, "(0.5°)")
print({w: dict(obs_hi[w].sizes) for w in WINDOWS}, "(0.1°)")

annual_lo = acmaddl.fetch("obs/chirps-v2-annual", "precip", region=list(NAFRICA_REGION),
                          hindcast=(1982, 2020), grid_res=RES_LO, boundary="cover",
                          verbose=False)["precip"]
annual_lo = annual_lo.mean([d for d in annual_lo.dims if d not in ("lat", "lon")])

# %% [markdown]
# #### 1.4 The acmadDL inventory
#
# The roster and observations above are a *selection* from acmadDL's
# catalogue, and the catalogue renders itself as a reference sheet — one row
# per product with its source, variables, resolution, cadence, and a
# **temporal-coverage bar** on a shared year axis. Forecast systems on the
# left, observed products on the right.
#
# This is the fastest way to answer the question that bites hardest when
# assembling a multi-model roster: *what years does each product actually
# have?* The bars show it directly — where a hindcast ends, where a real-time
# stream picks up (the arrow caps), and on the observations side the CHIRPS
# family beside the SST records the analogue section reads. Products beyond
# this notebook's roster are visible alongside, which is also how you find the
# next model to add.

# %%
_inv = [acmaddl.show_datasets("forecasts", save=FIG / "_inventory_forecasts.png"),
        acmaddl.show_datasets("observations", save=FIG / "_inventory_observations.png")]
plt.close("all")                       # the sheets are composed side by side below

fig, axes = plt.subplots(1, 2, figsize=(34, 23), layout="constrained")
for ax, (path, title) in zip(axes, [
        (FIG / "_inventory_forecasts.png", "Forecast systems"),
        (FIG / "_inventory_observations.png", "Observed products")]):
    ax.imshow(plt.imread(path))
    ax.set_title(title, fontsize=22, fontweight="bold")
    ax.axis("off")
plt.savefig(FIG / "01b_acmaddl_inventory.png", dpi=170, bbox_inches="tight")
plt.show()

# %% [markdown]
# #### 1.5 The dry mask and the display standard
#
# The mask compares the DJF climatological total against the annual total. It
# is derived **once, at the 0.5° operational grid**, and painted onto both
# working grids, so the 0.5° and 10 km maps are greyed identically.
#
# The **style** is `TercileStyle.named("icpac")` — the packaged house tercile
# standard across this notebook family (the ICPAC rainfall palette and
# 40–100 % probability bands), carrying the NA-RCC country clip, the lakes,
# and the dry mask, so every map below is masked and coloured identically.
#
# Expect the printed count below to be **large**. Interior Algeria, Libya and
# Egypt are hyper-arid; the Saharan fringe of Mauritania and the far south of
# Algeria get their little rain from the summer monsoon tail, so winter
# contributes under 15 % of the annual total across a wide band of the domain.
# That greying is the honest shape of this forecast: a DJF outlook for North
# Africa is an outlook for the Mediterranean and Atlantic margins, and the
# mask says so on every map rather than letting near-zero-rainfall terciles
# masquerade as signal.

# %%
_per_day = str(obs_lo[SEASON].attrs.get("units", "")).lower().startswith("mm/d")   # CHIRPS monthly arrives as mm/month
season_total_lo = (SEASON_DAYS if _per_day else 3) * obs_lo[SEASON].mean("year")
djf_share = season_total_lo / annual_lo.interp(lat=season_total_lo.lat,
                                               lon=season_total_lo.lon)
dry_lo = djf_share < DRY_SHARE                               # THE dry mask (0.5°)
dry_hi = (dry_lo.astype(float)                               # painted onto 0.1°
          .interp(lat=obs_hi[SEASON].lat, lon=obs_hi[SEASON].lon,
                  method="nearest") > 0.5)

STYLE_LO = africas2s.TercileStyle.named(
    "icpac", dry_mask=dry_lo, clip_to=NAFRICA_COUNTRIES, extent=EXTENT,
    lakes=True)
STYLE_HI = replace(STYLE_LO, dry_mask=dry_hi)

dry, outside = africas2s.region_masks(obs_lo[SEASON], STYLE_LO)
print(f"NA-RCC cells (0.5°): {int((~outside).sum())} of {outside.size}; "
      f"of those, {int((dry & ~outside).sum())} are DJF-dry at "
      f"{DRY_SHARE:.0%} of annual")

# %% [markdown]
# `"icpac"` is one of several packaged colour conventions. The library ships a
# reference sheet of all of them — the tercile styles (ICPAC rainfall, ICPAC
# temperature, ICPAC onset, NOAA CPC, ACMAD) with their probability-bin
# edges, and the classified field scales (NOAA CPC anomaly, UCSB CHIRPS
# anomaly and total, ICPAC onset date and spread) with their levels — each
# carrying the provenance note of the operational product it was sampled from.
# §1.6 uses `noaa-cpc-anomaly`; §5 uses the two ICPAC onset scales.

# %%
fig = africas2s.show_schemes()
fig.set_dpi(70)                     # display the reference sheet compactly
plt.show()

# %% [markdown]
# #### 1.6 What the raw models say
#
# Before any calibration, each model's ensemble-mean DJF 2025/26 total is
# shown as an **anomaly against that model's own DJF climatology**, so
# systematic wet or dry bias cancels and what remains is the signal
# calibration will work with — drawn on the packaged NOAA CPC
# precipitation-anomaly scale. Raw totals would mostly redraw the climatology
# once per model; the anomaly is the part that differs between models and
# between years. The fields are the very `(hindcast, forecast)` pairs §2
# calibrates, interpolated from the wide CCA domain onto the NA-RCC grid for
# display only.

# %%
unit  = next(iter(models[SEASON].values()))[0].attrs.get("units")
scale = SEASON_DAYS if unit == "mm/day" else 1.0
print(f"model fields are in {unit!r}; anomalies scaled by {scale:g} to mm over the season")

grid = obs_lo[SEASON]
raw_anomalies = {}
for label, (hind, fcst) in sorted(models[SEASON].items()):
    f = scale * fcst.isel(year=0).mean("member").interp(lat=grid.lat, lon=grid.lon)
    c = scale * hind.mean("member").mean("year").interp(lat=grid.lat, lon=grid.lon)
    raw_anomalies[label] = f - c

africas2s.plot_matrix(
    raw_anomalies, style=STYLE_LO, ncols=5, smooth=3,
    scale=FieldScale.named("noaa-cpc-anomaly"),
    suptitle=f"{WIN_LABEL[SEASON]} raw forecast anomaly, by model (Nov-{YEAR} init)",
    panel_size=(4.4, 3.4))
plt.savefig(FIG / "01_raw_anomaly_by_model.png", dpi=150)
plt.show()

# %% [markdown]
# ### 2. CCA — pattern calibration, model by model
#
# Where a per-cell regression treats every grid point independently, CCA works
# with **whole spatial patterns**. Both fields are reduced to their leading
# EOFs, then CCA finds weight vectors $(\mathbf{a}_k, \mathbf{b}_k)$
# maximising
#
# $$ \rho_k \;=\; \mathrm{corr}\big(\mathbf{a}_k^{\top}\mathbf{X},\;
#    \mathbf{b}_k^{\top}\mathbf{Y}\big), \qquad \rho_1 \ge \rho_2 \ge \dots $$
#
# with $\mathbf{X}$ the predictor — deliberately the **30°S–50°N belt** of
# model rainfall around the whole longitude circle, so both a remote tropical
# pattern (ENSO, the tropical Atlantic) and the North Atlantic /
# Mediterranean storm-track sector that steers North Africa's winter rain can
# be exploited — and $\mathbf{Y}$ observed NA-RCC rainfall. The forecast is
# reconstructed from the retained modes, cross-validated, and turned into
# tercile probabilities through a Student-t error model.
#
# One CCA runs per model, window and resolution, so each model is calibrated
# against its own hindcast record (`native_years=True`) and its probabilities
# are built per model before combining (`probability_aggregation=
# "cpt_per_model"`). The CCA settings follow common CPT practice: 7 X-EOFs, 7 Y-EOFs, 2 CCA modes, a 5-year cross-validation window, no
# standardisation. Probabilities stay 0–1 throughout, on the integer 0/1/2
# tercile coordinate `seasonal_mme` returns.

# %%
CCA_ARGS = dict(x_eof_modes=7, y_eof_modes=7, cca_modes=2,
                crossvalidation_window=5, standardize=False)

cca, cca_full = {}, {}
for w in WINDOWS:
    for rname, obs_w in (("0.5°", obs_lo[w]), ("10 km", obs_hi[w])):
        maps = {}
        for label, pair in sorted(models[w].items()):
            try:
                r = africas2s.seasonal_mme(
                    {"prcp": {label: pair}}, obs_w, method="cca",
                    native_years=True, probability_aggregation="cpt_per_model",
                    cpt_args=CCA_ARGS, forecast_year=FCST_YEAR[w], verbose=False)
            except Exception as e:
                print(f"[CCA dropped] {w} {rname} {label}: "
                      f"{type(e).__name__}: {str(e)[:90]}")
                continue
            maps[label] = r.tercile_forecast
            if w == SEASON and rname == "0.5°":
                cca_full[label] = r          # full result: §3 reads tercile_cv
        cca[(w, rname)] = maps
        print(f"{w} at {rname}: {len(maps)} calibrated models")

# %% [markdown]
# #### 2.1 The calibrated models, first window
#
# The per-model DJF 2025/26 maps at 10 km, after calibration. Comparing these
# panels with the raw anomalies in §1.6 shows what calibration actually does:
# it keeps the part of a model's signal that has historically tracked observed
# rainfall and discards the rest, so the maps are typically quieter and closer to one another.
# White cells are where no category reaches the style's display threshold —
# the forecast there is close to climatology.

# %%
africas2s.plot_matrix(
    {f"{label} (CCA)": probs for label, probs in sorted(cca[(SEASON, "10 km")].items())},
    style=STYLE_HI, ncols=5, smooth=3, legend_detailed=True,
    suptitle=f"{WIN_LABEL[SEASON]} after CCA calibration, by model (10 km)",
    panel_size=(4.4, 3.4))
plt.savefig(FIG / "02_cca_calibrated_by_model.png", dpi=150)
plt.show()

# %% [markdown]
# #### 2.2 The objective forecast
#
# The equal-weight combination of the calibrated models,
#
# $$ P_{\mathrm{obj}}(c \mid x) \;=\; \frac{1}{M}\sum_{m=1}^{M}
#    P_m(c \mid x), \qquad
#    c \in \{\text{below},\,\text{normal},\,\text{above}\} $$
#
# renormalised onto the simplex, so each model gets one vote regardless of how
# many ensemble members it carries. Averaging probabilities rather than
# picking a winner is deliberately conservative: where the models disagree the
# result relaxes toward climatology, and a confident tilt survives only where
# they agree. The first window is shown at both working resolutions — the
# high-resolution leg calibrates against 0.1° CHIRPS, which is precisely the
# axis CCA can exploit: its reconstruction lives on the predictand grid, so
# finer observations sharpen the map (skill remains bounded by the ~1°
# predictors).

# %%
def combine(maps):
    o = xr.concat(list(maps.values()), "m").mean("m")
    return o / o.sum("tercile")

objective = {k: combine(v) for k, v in cca.items()}

africas2s.plot_matrix(
    [(f"Objective {WIN_LABEL[SEASON]} — 0.5°", objective[(SEASON, "0.5°")]),
     (f"Objective {WIN_LABEL[SEASON]} — 10 km", objective[(SEASON, "10 km")], STYLE_HI)],
    style=STYLE_LO, ncols=2, smooth=3, legend_detailed=True,
    suptitle=f"NA-RCC objective rainfall outlook — {WIN_LABEL[SEASON]} (init {INIT_MONTH})",
    panel_size=(7.2, 4.6))
plt.savefig(FIG / "03_objective_two_resolutions.png", dpi=150)
plt.show()

# %% [markdown]
# #### 2.3 Three windows, two resolutions
#
# The same objective for every forecast window of this issuance — DJF 2025/26,
# JFM 2026 and FMA 2026, each at 0.5° and 10 km. Reading down the rows shows
# the forecast evolve with lead time (and cross the year boundary); reading
# across shows what the finer predictand adds.

# %%
panels = []
for w in WINDOWS:
    panels.append((f"{WIN_LABEL[w]} — 0.5°", objective[(w, "0.5°")]))
    panels.append((f"{WIN_LABEL[w]} — 10 km", objective[(w, "10 km")], STYLE_HI))

africas2s.plot_matrix(
    panels, style=STYLE_LO, ncols=2, smooth=3, legend_detailed=True,
    suptitle=f"Objective outlook by forecast window — init {INIT_MONTH}",
    panel_size=(6.4, 4.2))
plt.savefig(FIG / "04_objective_windows_6panel.png", dpi=150)
plt.show()

# %% [markdown]
# ### 3. Skill and expert adjustment
#
# #### 3.1 Cross-validated per-model skill
#
# How much should each model be trusted? The CCA pipeline keeps every model's
# **cross-validated tercile hindcast** — each training year forecast with that
# year (and its 5-year window) held out — and `africas2s.skill` scores it
# against the observations, cell by cell: **GROC** (generalized ROC / 2AFC —
# the probability the forecast ranks a wetter year above a drier one; 0.5 = no
# discrimination) and **RPSS** (ranked-probability skill score against
# climatology; > 0 beats climatology). Skill is assessed on the first window
# at the 0.5° operational grid, where the cross-validated records live.
#
# The model with the **lowest domain-mean GROC** is outlined in red — the
# candidate an operational forecaster would examine first when curating the
# roster.

# %%
skill_groc, skill_rpss, domain_groc = {}, {}, {}
for label, r in sorted(cca_full.items()):
    cv = r.tercile_cv
    o = obs_lo[SEASON].sel(year=cv.year.values)
    sk = africas2s.skill(cv, o, metrics=["rpss", "generalized_roc"], spatial=True)
    skill_groc[label] = sk.spatial["generalized_roc"]
    skill_rpss[label] = sk.spatial["rpss"]
    domain_groc[label] = sk.scores["generalized_roc"]

worst = min(domain_groc, key=domain_groc.get)
for label in sorted(domain_groc):
    flag = "   <- lowest domain-mean GROC" if label == worst else ""
    print(f"{label:18s} GROC {domain_groc[label]:.3f}{flag}")

# %%
fig_groc = africas2s.plot_matrix(
    {f"{l} — GROC {domain_groc[l]:.2f}": m for l, m in sorted(skill_groc.items())},
    style=STYLE_LO, ncols=5, cmap="RdBu", vmin=0, vmax=1,
    cbar_label="Generalized ROC (0.5 = no discrimination)",
    suptitle=f"Cross-validated GROC by model — {SEASON} hindcast (0.5°)",
    panel_size=(4.4, 3.0))
ix = sorted(skill_groc).index(worst)
sp = fig_groc.axes[ix].spines["geo"]
sp.set_edgecolor("#c0392b"); sp.set_linewidth(3.5); sp.set_visible(True)
plt.savefig(FIG / "05_skill_groc_by_model.png", dpi=150)
plt.show()

# %%
fig_rpss = africas2s.plot_matrix(
    {f"{l} (RPSS)": m for l, m in sorted(skill_rpss.items())},
    style=STYLE_LO, ncols=5, cmap="RdBu", vmin=-1, vmax=1,
    cbar_label="RPSS (0 = climatology)",
    suptitle=f"Cross-validated RPSS by model — {SEASON} hindcast (0.5°)",
    panel_size=(4.4, 3.0))
sp = fig_rpss.axes[ix].spines["geo"]
sp.set_edgecolor("#c0392b"); sp.set_linewidth(3.5); sp.set_visible(True)
plt.savefig(FIG / "06_skill_rpss_by_model.png", dpi=150)
plt.show()

# %% [markdown]
# #### 3.2 Re-combining without the lowest-skill model
#
# Operational forecasters curate the roster every cycle. Because §2 kept each
# model's **post-calibration map**, dropping a model is pure arithmetic on
# maps already in memory: re-average the 10 km per-model forecasts without the
# excluded model and renormalise — **no re-download, no re-calibration,
# seconds of work**. The exclusion here follows the skill maps above (lowest
# domain-mean GROC); the choice is the forecaster's. The third panel shows
# where it matters: the change in P(above-normal), in percentage points.

# %%
maps_hi  = cca[(SEASON, "10 km")]
obj_all  = objective[(SEASON, "10 km")]
obj_excl = combine({l: m for l, m in maps_hi.items() if l != worst})
diff_pp  = 100.0 * (obj_excl.sel(tercile=2) - obj_all.sel(tercile=2))

africas2s.plot_matrix(
    [(f"Objective {WIN_LABEL[SEASON]} — all models", obj_all),
     (f"Objective without {worst}", obj_excl),
     (f"Δ P(above), without {worst} − all (pp)", diff_pp)],
    style=STYLE_HI, ncols=3, smooth=3, legend_detailed=True,
    levels=[-10, -7.5, -5, -2.5, -1, 1, 2.5, 5, 7.5, 10], cmap="BrBG",
    extend="both", cbar_label="Δ P(above-normal) (percentage points)",
    suptitle=f"Model selection: the objective re-combined without {worst} (10 km)",
    panel_size=(6.2, 3.8))
plt.savefig(FIG / "07_objective_without_lowest_skill.png", dpi=150)
plt.show()

# %% [markdown]
# ### 4. Analogue years
#
# An independent check on the objective forecast: which past years' El
# Niño/La Niña evolution most resembles the run-up to this season? A
# caveat before reading these maps: **over North Africa ENSO is a weaker
# rainfall driver than the North Atlantic Oscillation**, whose phase sets the
# winter storm track into the Maghreb — the ENSO analogue set is shown here
# because it is the uniform cross-RCC method of this notebook family, and it
# is labelled as exactly that, not as the region's dominant driver.
#
# The Niño3.4 series comes from ERSST v5 (anomalies against the 1991–2020
# monthly climatology, smoothed to overlapping 3-month means — the ONI
# convention); the cell below cuts each year's twelve-season window at the same
# relative point, and `africas2s.analogs_from_evolution` ranks every candidate year
# by temporal correlation and mean absolute difference, composed. Because the
# ranking never sees rainfall, only SST, the observed rainfall in the selected
# years is a genuine out-of-sample check on the outlook above.
#
# **The window is the twelve seasons ending at issuance.** Each year's
# "evolution" is the run of twelve overlapping 3-month means that ends with the
# **last season fully observed before the forecast was issued**, and every
# candidate year is cut at the same relative point. That keeps the comparison
# issuance-anchored and leak-free, and it is always twelve points — which is
# what keeps the correlation meaningful. (A shorter window — only the seasons elapsed in the target's own calendar
# year — can be as few as five points, too few for a correlation to separate
# candidates.)

# %%
ersst = acmaddl.fetch("obs/ersst-v5", "sst", hindcast=(1981, 2025), verbose=False)["sst"]
nino = ersst.sel(lat=slice(-5, 5)).sel(lon=slice(190, 240)).mean(["lat", "lon"])
nino_anom = nino.groupby("time.month") - nino.sel(
    time=slice("1991-01-01", "2020-12-31")).groupby("time.month").mean("time")
oni = nino_anom.rolling(time=3, center=True).mean().dropna("time").rename("oni")

MON = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
N_STEPS, N_ANALOGS = 12, 4

# The anchor: the 3-month season whose LAST month is the month before issuance.
# A season is labelled by its centre month, so the anchor centre is init - 2.
_iy, _im = int(INIT_MONTH[:4]), int(INIT_MONTH[5:7])
ANCHOR_MONTH = (_im - 3) % 12 + 1
ANCHOR_YEAR = _iy if _im - 2 >= 1 else _iy - 1
LABELS = [MON[(ANCHOR_MONTH - 1 - (N_STEPS - 1) + k) % 12] for k in range(N_STEPS)]
ANCHOR_SST_MONTHS = [(ANCHOR_MONTH - 2) % 12 + 1, ANCHOR_MONTH, ANCHOR_MONTH % 12 + 1]

_oni_by_ym = {(int(t.dt.year), int(t.dt.month)): float(v)
              for t, v in zip(oni.time, oni.values)}

def evolution_curves(years, anchor_month, n=N_STEPS):
    """(year, step) ONI evolution: the n consecutive 3-month seasons ending at
    the season centred on `anchor_month` of each year."""
    rows = []
    for y in years:
        vals = []
        for k in range(n - 1, -1, -1):            # chronological, ending at the anchor
            m = anchor_month - k
            yy, mm = (y, m) if m >= 1 else (y - 1, m + 12)
            vals.append(_oni_by_ym.get((yy, mm), np.nan))
        rows.append(vals)
    return xr.DataArray(np.array(rows, dtype=float), dims=("year", "step"),
                        coords={"year": list(years), "step": np.arange(n)})

curves = evolution_curves(range(1982, ANCHOR_YEAR + 1), ANCHOR_MONTH)
print(f"Evolution window: {LABELS[0]} .. {LABELS[-1]}, ending {LABELS[-1]} "
      f"{ANCHOR_YEAR} — the last season fully observed before the "
      f"{INIT_MONTH} issuance ({int(np.isfinite(curves.sel(year=ANCHOR_YEAR)).sum())} "
      f"of {N_STEPS} seasons finite)")

analogs = africas2s.analogs_from_evolution(
    curves, target_year=ANCHOR_YEAR, n=N_ANALOGS,
    candidates=[int(y) for y in curves.year.values if y < ANCHOR_YEAR])
TOPN = [int(y) for y in analogs.years]
for rank, y in enumerate(TOPN, 1):
    print(f"  {rank}. {y}  r={analogs.metadata['r'][y]:+.2f}  "
          f"MAD={analogs.metadata['mad'][y]:.2f}")
# %%
africas2s.plot_index_evolution(
    curves, analogs, labels=LABELS, ylabel="Niño3.4 anomaly (°C)",
    title=f"Niño3.4 evolution over the 12 seasons ending "
          f"{LABELS[-1]} {ANCHOR_YEAR} (the {INIT_MONTH} issuance)")
plt.savefig(FIG / "08_nino34_analogue_plume.png", dpi=150)
plt.show()

# %% [markdown]
# #### The ocean state being matched, and its analogues
#
# The curves above compress the tropical Pacific into one number per season;
# the maps below are the patterns behind them. The **top panel is the state
# being matched** — the anchor season, the last one fully observed before
# issuance. The four beneath are the analogue years' patterns for the same
# season, in rank order, each readable against the target directly: the
# ranking is a correlation on the Niño3.4 *index*, and these maps show how
# much of the broader pattern comes with it.

# %%
def season_sst_anom(months):
    """ERSST anomaly composite for a set of calendar months, vs 1991-2020."""
    f = ersst.sel(time=ersst.time.dt.month.isin(months)).groupby("time.year").mean("time")
    return f - f.sel(year=slice(1991, 2020)).mean("year")

sst_anchor = season_sst_anom(ANCHOR_SST_MONTHS)
SST_LEVELS = [-2.5, -1.5, -1.0, -0.6, -0.3, 0.3, 0.6, 1.0, 1.5, 2.5]

# Target on top at full width (height_ratios gives row 0 the room a 2:1
# global map needs to fill it), the four analogues beneath.
fig = plt.figure(figsize=(16, 18), layout="constrained")
gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1])
proj = ccrs.PlateCarree(central_longitude=180)

def _sst_panel(ax, fld, title, target=False):
    cs = ax.contourf(fld.lon, fld.lat, fld.values, transform=ccrs.PlateCarree(),
                     levels=SST_LEVELS, cmap="RdBu_r", extend="both")
    ax.coastlines(lw=0.5); ax.set_global()
    ax.set_title(title, fontsize=14 if target else 12,
                 fontweight="bold" if target else "normal",
                 color="#c0392b" if target else "black")
    if target:
        for sp in ax.spines.values():
            sp.set_edgecolor("#c0392b"); sp.set_linewidth(3.0); sp.set_visible(True)
    return cs

ax_t = fig.add_subplot(gs[0, :], projection=proj)
ax_t.set_anchor("C")                      # centre it within the spanned row
cs = _sst_panel(ax_t, sst_anchor.sel(year=ANCHOR_YEAR),
                f"TARGET — {LABELS[-1]} {ANCHOR_YEAR} (anchor season, at issuance)",
                target=True)
for k, y in enumerate(TOPN):
    ax = fig.add_subplot(gs[1 + k // 2, k % 2], projection=proj)
    _sst_panel(ax, sst_anchor.sel(year=y),
               f"#{k + 1} analogue — {LABELS[-1]} {y}  "
               f"(r={analogs.metadata['r'][y]:+.2f})")
fig.colorbar(cs, ax=fig.axes, shrink=0.45, label="SST anomaly (°C)")
fig.suptitle(f"The ocean state being matched (top, red) and its {N_ANALOGS} "
             f"closest analogues", fontsize=15, fontweight="bold")
plt.savefig(FIG / "09_analogue_sst_patterns.png", dpi=150)
plt.show()

# %% [markdown]
# #### Observed outcomes in the analogue years, beside the objective forecast
#
# For each analogue year, the observed rainfall tercile at every cell against
# the 1991–2020 boundaries. A year either fell in a tercile or it did not — a
# categorical field. Each observed panel is drawn at one representative shade
# from its RCC category ramp, held at mid intensity so the outcomes sit
# alongside the forecast rather than shouting past it.
#
# The **first panel is this notebook's objective forecast**, shown as its
# leading tercile so all five panels are read the same way. Agreement is corroboration from a line of
# evidence that never saw rainfall; disagreement is a question worth asking
# before the forecast goes out.

# %%
obs_rec = obs_hi[SEASON]
q1 = obs_rec.sel(year=slice(1991, 2020)).quantile(1 / 3, "year").drop_vars("quantile")
q2 = obs_rec.sel(year=slice(1991, 2020)).quantile(2 / 3, "year").drop_vars("quantile")

extra = [y for y in TOPN if y not in set(int(v) for v in obs_rec["year"].values)]
if extra:
    try:
        more = acmaddl.fetch("obs/chirps-v2-monthly", "precip", target=SEASON,
                             region=list(NAFRICA_REGION), hindcast=(min(extra), max(extra)),
                             seasonal="mean", grid_res=RES_HI, boundary="cover",
                             verbose=False)["precip"]
        obs_rec = xr.concat([obs_rec, more.interp(lat=obs_rec.lat, lon=obs_rec.lon)], "year")
        print(f"fetched CHIRPS {SEASON} for the analogue year(s) {extra}")
    except Exception as e:
        print(f"NOTE: could not fetch CHIRPS for analogue year(s) {extra}: "
              f"{type(e).__name__}: {str(e)[:80]}")

have_years = set(int(v) for v in obs_rec["year"].values)
# One representative shade per category (the 60-70 % band of each RCC ramp).
# A one-hot observed field is 100 % in its category, so on the full probability
# ramp it would land in the top band and read as an extreme forecast rather
# than a recorded outcome.
CAT_STYLE = replace(STYLE_HI, below_colors=["#FF7A00"], normal_colors=["#8FEFEA"],
                    above_colors=["#39AB06"], prob_bins=[33.3, 100.01])

# The forecast rendered the way an observed outcome is: its leading tercile,
# one-hot. That makes all five panels categorical and directly comparable —
# "which way does the forecast lean" against "which way did it actually go".
# The probabilistic view, with its no-dominant-category white, is earlier in
# the notebook.
def dominant_onehot(probs):
    valid = np.isfinite(probs).all("tercile")
    lead = probs.fillna(-1.0).argmax("tercile")
    oh = xr.concat([(lead == k).astype(float) for k in range(3)], "tercile")
    return oh.assign_coords(tercile=[0, 1, 2]).where(valid)

panels = [(f"THIS FORECAST — objective {WIN_LABEL[SEASON]}",
           dominant_onehot(objective[(SEASON, "10 km")]))]
for rank, y in enumerate(TOPN, 1):
    if y not in have_years:
        print(f"NOTE: analogue year {y} lies outside the available CHIRPS record "
              f"({min(have_years)}-{max(have_years)}).")
        continue
    f = obs_rec.sel(year=y)
    below, above = f < q1, f > q2
    onehot = xr.concat([below, ~below & ~above, above], "tercile").astype(float)
    panels.append((f"#{rank} — observed {SEASON} {y}",
                   onehot.assign_coords(tercile=[0, 1, 2]).where(np.isfinite(f)),
                   CAT_STYLE))

africas2s.plot_matrix(
    panels, style=CAT_STYLE, ncols=len(panels), smooth=3,
    suptitle=f"The {WIN_LABEL[SEASON]} objective forecast, and the observed "
             f"{SEASON} tercile in each analogue year",
    panel_size=(5.8, 6.8))
plt.savefig(FIG / "10_observed_terciles_analogue_years.png", dpi=150)
plt.show()
# %% [markdown]
# ### 5. Onset of the rains
#
# When the season *starts* matters as much as how much falls. Onset is
# computed from the **daily** C3S forecasts with `africas2s.aggregations`,
# using the library's default onset criterion (the ICPAC/GHACOF definition):
# **20 mm accumulated over 3 days**, rejected if a
# 7-day dry spell follows within the next 21 days. The search window is the
# season's three months plus the following month — **1 December 2025 to
# 31 March 2026**, a window that itself crosses the year boundary — and the
# daily data are fetched one month further (into April) for the dry-spell
# guard; a November initialisation's daily streams reach that far.
#
# Five C3S systems issue daily fields from the same November initialisation;
# each is fetched on its 1° grid, its onset computed member by member, and the
# members pooled into one ensemble (a system whose daily fields are not
# available is dropped with the reason printed). The observed climatology
# applies the identical criterion to daily CHIRPS v3 (the Rhiza mirror,
# 0.25°): the record is sliced to complete December–April blocks, so each
# season year $y$ spans December $y$ to April $y{+}1$ — the library labels
# these wraparound seasons by their starting December, giving the twenty
# complete season years **2005–2024**. Maps are drawn at each product's
# **native grid** — no interpolation — with the packaged ICPAC onset colour
# schemes (`icpac-onset-date`, `icpac-onset-spread`).
#
# **White cells are an honest signal, not a mask**: no qualifying onset in at
# least half the members (forecast) or years (climatology). Over most of the
# Sahara the 20 mm/3 d trigger is simply never met in winter — a qualifying
# onset legitimately does not occur — so expect the coloured cells to hug the
# Maghreb coast, the Atlas and the Mediterranean fringe, and read the white
# interior as "no rainy-season onset", which is the expected result for this domain.

# %%
DAILY_MODELS = [("ECMWF", "c3s/ecmwf"), ("Meteo_France", "c3s/meteofrance-daily"),
                ("CMCC", "c3s/cmcc-sps4-daily"), ("DWD", "c3s/dwd-daily"),
                ("ECCC", "c3s/eccc-daily")]
ONSET_SEASON, ONSET_FETCH = "DJFM", "DJFMA"   # search 1 Dec - 31 Mar; data into Apr
ONSET_START = dt.date(YEAR, 12, 1)

pooled, used, offset = [], [], 0
for label, product in DAILY_MODELS:
    try:
        daily = acmaddl.fetch(product, "precip", init=INIT_MONTH, target=ONSET_FETCH,
                              region=list(NAFRICA_REGION), hindcast=(YEAR, YEAR),
                              grid_res=1.0, verbose=False)["precip"]
        daily = (daily.squeeze("init_time", drop=True)
                      .swap_dims({"lead_time": "time"}).clip(min=0))
        step = agg.onset(daily, season=ONSET_SEASON).step.squeeze("year", drop=True)
        step = step.assign_coords(member=np.arange(offset, offset + step.sizes["member"]))
        offset += step.sizes["member"]
        pooled.append(step)
        used.append(f"{label} ({step.sizes['member']}m)")
    except Exception as e:
        print(f"[onset dropped] {label}: {type(e).__name__}: {str(e)[:110]}")

assert pooled, "no daily onset models available — check CDS credentials / licences"
onset_ens = xr.concat(pooled, dim="member")
print(f"onset ensemble: {onset_ens.sizes['member']} members — " + ", ".join(used))

chirps_daily = acmaddl.fetch("obs/chirps-v3-daily-rhiza", "precip",
                             region=list(NAFRICA_REGION), hindcast=(2005, 2025),
                             months=[12, 1, 2, 3, 4], grid_res=0.25,
                             verbose=False)["precip"]
chirps_daily = chirps_daily.sel(time=slice("2005-12-01", "2025-04-30"))
obs_onset = agg.onset(chirps_daily, season=ONSET_SEASON)
print(f"observed onset climatology: season years "
      f"{int(obs_onset.step.year.min())}-{int(obs_onset.step.year.max())}")

# %%
DATE_SCALE = FieldScale.named("icpac-onset-date")
step_to_date = lambda t: (ONSET_START + dt.timedelta(days=int(t))).strftime("%d %b")

frac = onset_ens.notnull().mean("member")
panels = [
    (onset_ens.mean("member").where(frac >= 0.5), DATE_SCALE,
     f"Onset ensemble mean — DJF {YEAR}/{(YEAR + 1) % 100:02d}"),
    (onset_ens.std("member").where(frac >= 0.5), FieldScale.named("icpac-onset-spread"),
     f"Onset spread — DJF {YEAR}/{(YEAR + 1) % 100:02d}"),
    (obs_onset.step.mean("year").where(obs_onset.occurred.mean("year") >= 0.5), DATE_SCALE,
     "Observed onset climatology (CHIRPS 2005–2024)"),
]
fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), layout="constrained",
                         subplot_kw={"projection": ccrs.PlateCarree()})
for ax, (fld, sc, ttl) in zip(axes, panels):
    im = africas2s.plot_field(fld, ax=ax, style=STYLE_HI, smooth=3, scale=sc, title=ttl)
    cb = fig.colorbar(im, ax=ax, shrink=0.7) if im is not None else None
    if cb is not None and sc is DATE_SCALE:
        cb.set_ticks(sc.levels)
        cb.set_ticklabels([step_to_date(t) for t in sc.levels], fontsize=8)
fig.suptitle(f"Onset of the DJF {YEAR}/{(YEAR + 1) % 100:02d} rains — NA-RCC (20 mm/3 d "
             "trigger, no 7-day dry spell within 21 d, search 1 Dec – 31 Mar)",
             fontsize=14, fontweight="bold")
plt.savefig(FIG / "11_onset_mean_spread_climatology.png", dpi=150)
plt.show()
