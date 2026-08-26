"""Demo: the streamlined multi-panel plotting API for replication workflows.

Renders, from synthetic data (no downloads, no credentials):

  1. a member matrix — a grid of per-model tercile forecasts sharing one
     style, one skill mask, and one figure-level legend (`plot_matrix`)
  2. a pre/post-calibration comparison — raw continuous anomalies and
     calibrated tercile maps mixed in a single matrix
  3. the components + objective composite, smooth-contoured in the GHACOF
     color language (`plot_components_objective`)
  4. the same composite in ACMAD's continental palette (`styles/acmad.json`)

Run from the repository root:
  uv run python examples/demo_plotting_panels.py

Figures are written to examples/output/.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import xarray as xr

import africas2s as ds

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Synthetic forecasts: three "models" with correlated but distinct signals
# over an equatorial Africa box, plus masks.
# ---------------------------------------------------------------------------
rng = np.random.RandomState(7)
lat = np.linspace(-12.0, 12.0, 48)
lon = np.linspace(8.0, 42.0, 68)
LON, LAT = np.meshgrid(lon, lat)


def tercile_forecast(shift, noise=0.05):
    """A (tercile, lat, lon) probability field with a smooth wet/dry dipole."""
    signal = np.sin((LON + shift) / 6.0) * np.cos(LAT / 9.0)
    above = np.clip(1 / 3 + 0.33 * signal + noise * rng.randn(*signal.shape), 0.02, 0.92)
    below = np.clip(1 / 3 - 0.33 * signal + noise * rng.randn(*signal.shape), 0.02, 0.92)
    normal = np.clip(1.0 - above - below, 0.02, None)
    p = np.stack([below, normal, above])
    return xr.DataArray(p / p.sum(0), dims=["tercile", "lat", "lon"],
                        coords={"tercile": [0, 1, 2], "lat": lat, "lon": lon})


def anomaly_field(shift):
    """A continuous seasonal-total anomaly (mm) matching the tercile signal."""
    signal = np.sin((LON + shift) / 6.0) * np.cos(LAT / 9.0)
    return xr.DataArray(220.0 * signal + 15.0 * rng.randn(*signal.shape),
                        dims=["lat", "lon"], coords={"lat": lat, "lon": lon})


MODELS = ["CanSIPS-IC4", "NASA-GEOSS2S", "NCEP-CFSv2"]
raw = {m: anomaly_field(4 * i) for i, m in enumerate(MODELS)}
calibrated = {m: tercile_forecast(4 * i) for i, m in enumerate(MODELS)}
objective = sum(calibrated.values()) / len(calibrated)

# Masks: a "dry season" band in the north, and a synthetic skill field.
too_dry = xr.DataArray(LAT > 9.5, dims=["lat", "lon"],
                       coords={"lat": lat, "lon": lon})
skill = xr.DataArray(np.cos(LAT / 14.0) - 0.15 * np.abs(np.sin(LON / 10.0)),
                     dims=["lat", "lon"], coords={"lat": lat, "lon": lon})

# One line per centre: each colour language lives in a JSON style file the
# workflow owns (examples/styles/ here); masks attach as keyword overrides.
STYLES_DIR = Path(__file__).resolve().parent / "styles"
ghacof = ds.TercileStyle.from_json(STYLES_DIR / "ghacof.json", dry_mask=too_dry)
acmad = ds.TercileStyle.from_json(STYLES_DIR / "acmad.json", dry_mask=too_dry)

# ---------------------------------------------------------------------------
# 1. Member matrix: every calibrated forecast, skill-masked, shared legend.
# ---------------------------------------------------------------------------
fig = ds.plot_matrix(
    calibrated, style=ghacof, ncols=3,
    skill_mask=skill < 0.55,
    suptitle="Calibrated members — GHACOF palette, skill-masked",
)
fig.savefig(OUTPUT_DIR / "panels_member_matrix.png", dpi=150)
print("wrote", OUTPUT_DIR / "panels_member_matrix.png")

# ---------------------------------------------------------------------------
# 2. Pre/post calibration in ONE matrix: continuous anomaly panels and
#    tercile panels mix freely; continuous panels share one colorbar.
# ---------------------------------------------------------------------------
panels = []
for m in MODELS:
    panels.append((f"{m} — raw anomaly", raw[m]))
    panels.append((f"{m} — after CCA", calibrated[m]))
fig = ds.plot_matrix(
    panels, style=ghacof, ncols=2,
    cmap=ds.tercile_diverging_cmap(ghacof),
    levels=[-300, -180, -100, -50, -15, 15, 50, 100, 180, 300],
    cbar_label="mm over the season (forecast − climatology)",
    suptitle="Before and after calibration, per model",
)
fig.savefig(OUTPUT_DIR / "panels_pre_post_cca.png", dpi=150)
print("wrote", OUTPUT_DIR / "panels_pre_post_cca.png")

# ---------------------------------------------------------------------------
# 3. Components + objective, smooth-contoured, detailed legend. The second
#    objective panel switches the dry mask off via a per-panel style — the
#    display-time toggle the workflows keep asking for.
# ---------------------------------------------------------------------------
fig = ds.plot_components_objective(
    calibrated, objective, style=ghacof, smooth=True, legend_detailed=True,
    objective_label="OBJECTIVE (1/3 each)",
    suptitle="Components + objective — smooth GHACOF rendering",
)
fig.savefig(OUTPUT_DIR / "panels_components_objective.png", dpi=150)
print("wrote", OUTPUT_DIR / "panels_components_objective.png")

fig = ds.plot_matrix(
    [("dry mask applied", objective),
     ("dry mask OFF", objective, replace(ghacof, dry_mask=None))],
    style=ghacof, ncols=2, smooth=True,
    suptitle="Objective — dry mask as a display switch",
)
fig.savefig(OUTPUT_DIR / "panels_objective_dry_switch.png", dpi=150)
print("wrote", OUTPUT_DIR / "panels_objective_dry_switch.png")

# ---------------------------------------------------------------------------
# 4. Same composite, ACMAD continental palette.
# ---------------------------------------------------------------------------
fig = ds.plot_components_objective(
    calibrated, objective, style=acmad,
    objective_label="OBJECTIVE",
    suptitle="Components + objective — ACMAD palette",
)
fig.savefig(OUTPUT_DIR / "panels_components_objective_acmad.png", dpi=150)
print("wrote", OUTPUT_DIR / "panels_components_objective_acmad.png")

print("done — all figures in", OUTPUT_DIR)
