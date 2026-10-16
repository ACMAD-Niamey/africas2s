# Regional forecast examples — the five African RCCs

Five worked seasonal rainfall forecasts, one per WMO-designated Regional
Climate Centre on the continent. Each is a complete CCA-based workflow on the
AfricaS2S stack — roster, observations, calibration, objective forecast, skill
and onset — for that RCC's own domain and its own rainy season.

| Example | RCC | Domain | Season | Initialised |
|---|---|---|---|---|
| `ICPAC_OND2026.py` | **ICPAC** (IGAD) | Greater Horn of Africa | OND "short rains" | Sep 2026 |
| `CAPC_AC_ASO2026.py` | **CAPC-AC** (ECCAS/CEEAC) | Central Africa | ASO | Aug 2026 |
| `AGRHYMET_JAS2026.py` | **AGRHYMET / RCC-WAS** (CILSS) | West Africa & the Sahel | JAS monsoon | Jun 2026 |
| `SADC_DJF2025.py` | **SADC CSC** | Southern Africa | DJF | Nov 2025 |
| `NAFRICA_DJF2025.py` | **NA-RCC** (PRESANORD) | Maghreb–Sahara | DJF | Nov 2025 |

They share one structure deliberately, so the parts that differ stand out as
the parts that matter — the domain, the season, and the display conventions:

1. **Setup** — region and season; the model roster with each system's native
   hindcast window; CHIRPS observations; the dry mask and the RCC colour style
2. **Raw forecasts** — each model's anomaly against its own climatology
3. **CCA calibration** — one canonical-correlation fit per model, from a
   global-tropics predictor onto the regional predictand
4. **The objective forecast** — equal-weight probability average
5. **Skill** — cross-validated GROC and RPSS per model
6. **Analogue years** — the past years whose Niño3.4 and IOD evolution most
   resembles this one, and what fell in them
7. **Onset** — from the daily C3S fields, where the season has a meaningful one

## Running them

They are [jupytext](https://jupytext.readthedocs.io) percent-format scripts:
run them directly, or build an executed notebook. Either way, run **from this
directory** — figures are written to `outputs/<EXAMPLE>/` and the RCC marks are
read from `assets/logos/`, both relative paths.

```bash
cd examples/regional-forecasts

python ICPAC_OND2026.py                               # run as a script
python tools/py2nb.py ICPAC_OND2026.py ICPAC_OND2026.ipynb   # or build + execute a notebook
```

### Prerequisites

These are **real-data** examples. Beyond `africas2s` they need:

- **[acmadDL](https://github.com/ACMAD-Niamey/acmadDL)** for the data layer —
  `pip install acmadDL`
- **Copernicus CDS credentials** in `~/.cdsapirc`, with each C3S seasonal
  dataset's licence accepted. The C3S seasonal products live on the Copernicus
  CDS (`https://cds.climate.copernicus.eu/api`); a `.cdsapirc` pointed at the
  ECMWF Data Store instead makes every seasonal request fail with a 404.
- NMME systems, CHIRPS and ERSST need no credentials.

A cold run fetches a lot: the observation records are the slow part (~25
minutes for a new region/season), model hindcasts are minutes each. Everything
caches, so a second run only does what is missing.

The **executed notebooks are committed** alongside the sources, so every
figure and number is readable on GitHub without running anything. Rebuild one
with `tools/py2nb.py` after editing its `.py`.

`py2nb.py` embeds the RCC logo marks as base64 at build time: GitHub's `.ipynb`
renderer does not resolve relative image paths in markdown cells (README files
do), so an embedded image is the only form that renders both on GitHub and in a
local Jupyter.

## A note on the rosters

A model roster is a *request*, not a guarantee. Real-time availability drifts
month to month: a system can have a complete hindcast but no forecast yet, a
provider can serve all-fill data for a whole target, and some systems issue
only quarterly. Each example therefore guards every model with
`acmaddl.usable(...)` and **drops a failing model with a printed reason**
rather than silently. Expect to see a model or two dropped on any given run.

## Display conventions

Every map is drawn through a packaged style
(`africas2s.TercileStyle.named(...)`) rather than an ad-hoc colour scheme, so
these examples double as a reference for the RCC display standards —
probability bands, country clips, dry-season greying and onset ramps.
`africas2s.show_schemes()` prints the full set.
