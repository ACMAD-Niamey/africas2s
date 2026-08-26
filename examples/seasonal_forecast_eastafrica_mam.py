"""End-to-end seasonal forecast reference: East Africa MAM.

A AfricaS2S + Rosetta port of ``pycpt-reference/pycpt_seasonal_forecast.py``,
structured in the same 7 phases and runnable phase-by-phase.

  # full real run (needs CDS creds + network):
  uv run python examples/seasonal_forecast_eastafrica_mam.py
  uv run python examples/seasonal_forecast_eastafrica_mam.py --phase 0 1 2

  # plan only, no network:
  uv run python examples/seasonal_forecast_eastafrica_mam.py --dry-run

  # synthetic smoke (no network), exercises the real pipeline end to end:
  uv run python examples/seasonal_forecast_eastafrica_mam.py --tiny --phase 0 1 2 3 5 6

Model note
----------
PyCPT's reference blends 10 GCMs. SPEAR / SPEARb / CanSIPS-IC4 lived in the
now-sunset IRI Data Library; a CCSR-successor adapter serves them again over
OPeNDAP, so they are now included. Models still served from elsewhere
(ECMWF/CMCC/DWD/Meteo-France via C3S; CCSM4/GEOS via NCEI) round out the
ensemble. Phase 0's banner lists exactly which models the MME used.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import africas2s as ds
from africas2s.plotting import (
    plot_domains, plot_skill_maps, plot_eof_modes, plot_cca_modes,
    plot_tercile_forecast, plot_deterministic_forecast,
    plot_exceedance_probability, plot_flex_pdf,
)

# ---- Configuration (mirrors the PyCPT reference's CONFIGURATION block) -------
REGION = [-12, 6, 28, 42]          # East Africa [lat_s, lat_n, lon_w, lon_e]
INIT = "2025-02"                   # February initialization
TARGET = "MAM"                     # March-April-May
TARGET_MONTHS = [3, 4, 5]
HINDCAST = (1993, 2016)
NAIROBI = (36.8, -1.3)             # (lon, lat) for the flex-forecast PDF point
PREDICTAND = "obs/chirps"          # Sheerwater-backed CHIRPS

# PyCPT reference models, as rosetta products. (track, label, rosetta_product)
# SPEAR + CanSIPS-IC4 are served via the CCSR adapter; hindcast entries are
# used here (CV training over 1991-2020).
PRCP_MODELS = [
    ("ECMWF-SEAS5", "c3s/ecmwf-monthly"),
    ("CCSM4", "nmme/ccsm4"),
    ("GEOS-S2S", "nmme/geoss2s"),
    ("SPEAR", "nmme/spear-hindcast"),
    ("CanSIPS-IC4", "nmme/cansipsic4-hindcast"),
]
SST_MODELS = [
    ("CCSM4", "nmme/ccsm4"),
    ("GEOS-S2S", "nmme/geoss2s"),
    ("SPEARb", "nmme/spearb-hindcast"),
    ("CanSIPS-IC4", "nmme/cansipsic4-hindcast"),
]
EXCLUDED_MODELS = []  # SPEAR/CanSIPS restored via CCSR adapter

# crossvalidation_window=5 matches PyCPT's default (africas2s's own default is 1).
CPT_ARGS = {"crossvalidation_window": 5}


def _banner(header, lines):
    print(f"\n{header}\n" + "-" * len(header))
    for ln in lines:
        print(f"  {ln}")


# ----------------------------------------------------------------------------
# Phase 0: config banner, catalog verification, domain map.
# ----------------------------------------------------------------------------
def phase0(ctx):
    _banner("Seasonal forecast reference: East Africa MAM (AfricaS2S + Rosetta)", [
        f"region={REGION}  init={INIT}  target={TARGET}  hindcast={HINDCAST}",
        f"predictand={PREDICTAND}",
        f"PRCP track: {[m[0] for m in PRCP_MODELS]}",
        f"SST  track: {[m[0] for m in SST_MODELS]}",
        (f"excluded: {EXCLUDED_MODELS}" if EXCLUDED_MODELS
         else "excluded: none (SPEAR/CanSIPS restored via CCSR adapter)"),
    ])
    if ctx["dry_run"] or ctx["tiny"]:
        print("[phase 0] (dry-run/tiny) skipping remote catalog verification.")
    else:
        import rosetta
        for _label, product in PRCP_MODELS + SST_MODELS + [("CHIRPS", PREDICTAND)]:
            try:
                health = rosetta.check_product(product, probe_remote=False)
                ok = getattr(health, "healthy", health.get("healthy") if isinstance(health, dict) else None)
                print(f"[phase 0] catalog {product}: healthy={ok}")
            except Exception as e:
                print(f"[phase 0] catalog {product}: ERROR {e}")
    _save_plot(ctx, "domains", lambda: _plot_domains())


def _plot_domains():
    return plot_domains(predictor_extent=tuple(REGION),
                        predictand_extent=tuple(REGION),
                        title=f"Domain: East Africa {TARGET}")


# ----------------------------------------------------------------------------
# Phase 1: fetch (or synthesize) predictor tracks + predictand.
# ----------------------------------------------------------------------------
def phase1(ctx):
    if ctx["dry_run"]:
        print("[phase 1] DRY RUN - would fetch via Rosetta:")
        for track, models in (("prcp", PRCP_MODELS), ("sst", SST_MODELS)):
            for label, product in models:
                print(f"    {track:4s}  {label:12s} <- {product}  "
                      f"(init={INIT}, target={TARGET}, region={REGION})")
        print(f"    obs        CHIRPS       <- {PREDICTAND}")
        return
    if ctx["tiny"]:
        ctx["tracks"], ctx["obs"] = _synthetic_tracks_and_obs()
        print(f"[phase 1] (tiny) synthetic tracks={list(ctx['tracks'])} "
              f"obs={dict(ctx['obs'].sizes)}")
        return
    ctx["tracks"], ctx["obs"] = _fetch_tracks_and_obs()
    print(f"[phase 1] fetched tracks={list(ctx['tracks'])} obs={dict(ctx['obs'].sizes)}")


def _synthetic_tracks_and_obs():
    """Small synthetic dual-grid data so the whole pipeline runs offline."""
    rng = np.random.default_rng(0)
    years = np.arange(1993, 2016)
    members = np.arange(4)
    clat = np.linspace(-10, 4, 6); clon = np.linspace(30, 40, 6)   # coarse predictor
    flat = np.linspace(-10, 4, 12); flon = np.linspace(30, 40, 12)  # fine predictand (incl. Nairobi)

    signal = np.sin(np.arange(len(years)) * 0.4)

    def _gcm(seed):
        r = np.random.default_rng(seed)
        spatial = np.outer(np.sin(clat * 0.5), np.cos(clon * 0.3))
        data = (signal[:, None, None, None] * spatial[None, None, :, :]
                + r.standard_normal((len(years), len(members), len(clat), len(clon))) * 0.3 + 5.0)
        return xr.DataArray(data, dims=["year", "member", "lat", "lon"],
                            coords={"year": years, "member": members, "lat": clat, "lon": clon})

    spatial_f = np.outer(np.sin(flat * 0.5), np.cos(flon * 0.3))
    obs = xr.DataArray(
        signal[:, None, None] * spatial_f[None, :, :]
        + rng.standard_normal((len(years), len(flat), len(flon))) * 0.2 + 5.0,
        dims=["year", "lat", "lon"],
        coords={"year": years, "lat": flat, "lon": flon},
    )
    tracks = {
        "prcp": {label: (_gcm(i + 1), None) for i, (label, _p) in enumerate(PRCP_MODELS)},
        "sst": {label: (_gcm(i + 10), None) for i, (label, _p) in enumerate(SST_MODELS)},
    }
    return tracks, obs


def _fetch_tracks_and_obs():
    """Fetch the MME predictor tracks + CHIRPS predictand via Rosetta."""
    import rosetta
    y0, y1 = HINDCAST

    def _fetch_gcm(product, variable):
        ds = rosetta.fetch(product, variable, init=INIT, target=TARGET,
                           hindcast=HINDCAST, region=REGION, verbose=False)
        da = ds[variable] if hasattr(ds, "data_vars") and variable in ds.data_vars else ds
        return da

    tracks = {"prcp": {}, "sst": {}}
    for label, product in PRCP_MODELS:
        tracks["prcp"][label] = (_fetch_gcm(product, "precip"), None)
    for label, product in SST_MODELS:
        tracks["sst"][label] = (_fetch_gcm(product, "sst"), None)

    obs_ds = rosetta.fetch(PREDICTAND, "precip", hindcast=HINDCAST, region=REGION, verbose=False)
    obs = obs_ds["precip"] if hasattr(obs_ds, "data_vars") and "precip" in obs_ds.data_vars else obs_ds
    # Aggregate obs to the target season mean per year if a time axis is present.
    if "time" in obs.dims:
        seasonal = obs.sel(time=obs.time.dt.month.isin(TARGET_MONTHS))
        obs = seasonal.groupby("time.year").mean("time")
    return tracks, obs


# ----------------------------------------------------------------------------
# Phase 2: seasonal MME (both tracks).
# ----------------------------------------------------------------------------
def phase2(ctx):
    _require(ctx, "tracks", "obs")
    result = ds.seasonal_mme(ctx["tracks"], ctx["obs"],
                             cpt_args=CPT_ARGS, verbose=False)
    ctx["result"] = result
    print(f"[phase 2] MME complete: forecast={dict(result.forecast.sizes)}, "
          f"scores={list(result.skill_report.scores)[:6]}")


# ----------------------------------------------------------------------------
# Phase 3: per-MME skill maps.
# ----------------------------------------------------------------------------
def phase3(ctx):
    _require(ctx, "result")
    metrics = [m for m in ("rpss", "pearson_r", "spearman", "2afc")
               if m in ctx["result"].skill_report.scores]
    _save_plot(ctx, "skill_maps", lambda: _plot_skill(ctx, metrics))


def _plot_skill(ctx, metrics):
    return plot_skill_maps(ctx["result"].skill_report, metrics, ncols=2)


# ----------------------------------------------------------------------------
# Phase 4: EOF + CCA modes (per model).
# ----------------------------------------------------------------------------
def phase4(ctx):
    _require(ctx, "result")
    methods = ctx["result"].per_model_methods
    if not methods:
        print("[phase 4] no per-model methods to plot; skipping.")
        return
    (_key, m) = next(iter(methods.items()))
    _save_plot(ctx, "eof_modes", lambda: _plot_modes(m, "eof"))
    _save_plot(ctx, "cca_modes", lambda: _plot_modes(m, "cca"))


def _plot_modes(method, kind):
    if kind == "eof":
        return plot_eof_modes(method, kind="predictor", n_modes=2)
    return plot_cca_modes(method, n_modes=2)


# ----------------------------------------------------------------------------
# Phase 5: MME tercile + deterministic forecast + MME skill.
# ----------------------------------------------------------------------------
def phase5(ctx):
    _require(ctx, "result")
    r = ctx["result"]
    _save_plot(ctx, "mme_tercile",
               lambda: _plot_tercile(r.tercile_forecast))
    _save_plot(ctx, "mme_deterministic",
               lambda: _plot_det(r.forecast))
    # MME skill reuses plot_skill_maps (no separate plot_mme_skill helper).
    metrics = [m for m in ("rpss", "pearson_r") if m in r.skill_report.scores]
    if metrics:
        _save_plot(ctx, "mme_skill", lambda: _plot_skill(ctx, metrics))


def _plot_tercile(tercile_forecast):
    tcst = tercile_forecast
    if "year" in tcst.dims:
        tcst = tcst.isel(year=-1)
    return plot_tercile_forecast(tcst, title=f"MME dominant tercile ({TARGET})")


def _plot_det(forecast):
    return plot_deterministic_forecast(forecast, title=f"MME Deterministic ({TARGET})")


# ----------------------------------------------------------------------------
# Phase 6: flex forecast - exceedance probability + Nairobi PDF.
# ----------------------------------------------------------------------------
def phase6(ctx):
    _require(ctx, "result", "obs")
    r = ctx["result"]
    if r.pev is None:
        print("[phase 6] no PEV available; skipping flex forecast.")
        return
    # P(precip > climatological median).
    flex = ds.flex_forecast(r.forecast, r.pev, ctx["obs"], threshold=0.5, is_percentile=True)
    ctx["flex"] = flex
    print(f"[phase 6] flex forecast: mean exceedance(>median)="
          f"{float(flex.exceedance_prob.mean()):.3f}")
    _save_plot(ctx, "exceedance",
               lambda: _plot_exceedance(flex))
    _save_plot(ctx, "nairobi_pdf",
               lambda: _plot_pdf(flex))


def _plot_exceedance(flex):
    return plot_exceedance_probability(flex.exceedance_prob, threshold="median")


def _plot_pdf(flex):
    return plot_flex_pdf(flex.fcst_mu, flex.fcst_scale, flex.climo_mu, flex.climo_scale,
                         location=NAIROBI)


# ---- shared helpers ---------------------------------------------------------
def _require(ctx, *keys):
    missing = [k for k in keys if k not in ctx]
    if missing:
        raise RuntimeError(
            f"phase prerequisites missing: {missing}. Run earlier phases "
            f"(this normally can't happen - the runner auto-includes prerequisites)."
        )


def _save_plot(ctx, name, fig_fn):
    """Render a figure and save it; degrade gracefully if plotting deps absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[plot] matplotlib not installed - skipping '{name}'.")
        return
    try:
        fig = fig_fn()
    except Exception as e:  # plotting is best-effort; never fail the pipeline on a plot
        print(f"[plot] skipped '{name}': {type(e).__name__}: {e}")
        return
    out = ctx["output_dir"] / f"eastafrica_mam_{name}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out}")


_PHASES = {0: phase0, 1: phase1, 2: phase2, 3: phase3, 4: phase4, 5: phase5, 6: phase6}


def _expand_with_prerequisites(selected):
    """Phases >=2 need data (1) and the MME result (2); >=3 also need 2."""
    run = set(selected)
    if any(p >= 2 for p in selected):
        run.update({1, 2})
    if any(p >= 3 for p in selected):
        run.add(2)
    return sorted(run)


def main(argv=None):
    ap = argparse.ArgumentParser(description="East Africa MAM seasonal forecast reference.")
    ap.add_argument("--phase", type=int, nargs="+", default=list(range(7)),
                    choices=range(7), help="Phases to run (default: all).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan; no network, no compute.")
    ap.add_argument("--tiny", action="store_true",
                    help="Run the full pipeline on small synthetic data (no network).")
    ap.add_argument("--output-dir", type=Path,
                    default=Path(__file__).parent / "output" / "eastafrica_mam",
                    help="Where to write figures.")
    args = ap.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ctx = {"dry_run": args.dry_run, "tiny": args.tiny, "output_dir": args.output_dir}

    if args.dry_run:
        phases = sorted({0, 1} | set(args.phase) & {0, 1})
    else:
        phases = _expand_with_prerequisites(args.phase)

    for p in phases:
        _PHASES[p](ctx)

    print("Pipeline complete" + (" (dry-run)" if args.dry_run else
                                  " (tiny)" if args.tiny else ""))


if __name__ == "__main__":
    main()
