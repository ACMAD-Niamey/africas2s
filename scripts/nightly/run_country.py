"""Per-country pipeline driver, invoked once per matrix job.

Usage (from workflow):
    uv run python -m scripts.nightly.run_country \
        --country kenya --today 2026-05-16 --output-root output/

Exit code 0 even if some (country, season) targets fail; per-target status
is captured in output/<country>/<season>/<init>/status.json so publish.py
can record it. The job exits non-zero only if ALL targets for the country
fail (treat that as a "country job died" and let the gather job synthesise
failed rows).
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT.parent / "rosetta" / "src"))

import africas2s  # noqa: E402

from scripts.nightly.config import load_config  # noqa: E402
from scripts.nightly.output_writer import write_output  # noqa: E402
from scripts.nightly.select_targets import Target, select_targets  # noqa: E402


def _fetch_obs(country, target):
    # We fetch the full monthly obs series for the hindcast period; seasonal
    # selection happens downstream in _to_obs_array using the season's
    # target_months. `target` is accepted for signature symmetry with
    # _fetch_gcm and to anchor future per-init obs fetching.
    import rosetta

    bbox = country.bbox
    region = [bbox["south"], bbox["north"], bbox["west"], bbox["east"]]
    return rosetta.fetch(
        country.observations,
        country.predictand_var,
        hindcast=country.hindcast_period,
        region=region,
    )


def _fetch_gcm(country, target, product: str):
    import rosetta

    bbox = country.bbox
    region = [bbox["south"], bbox["north"], bbox["west"], bbox["east"]]
    return rosetta.fetch(
        product,
        country.predictand_var,
        init=f"{target.init_year}-{target.init_month:02d}",
        target=target.season,
        hindcast=country.hindcast_period,
        region=region,
    )


def _to_obs_array(obs_ds, variable, target_months, hindcast_period):
    """Reshape rosetta obs Dataset into (year, lat, lon) seasonal mean DataArray.

    Mirrors examples/demo_seasonal_mme.py::_era5_to_obs.
    """
    da = obs_ds[variable]
    seasonal = da.sel(time=da.time.dt.month.isin(target_months))
    annual = seasonal.groupby("time.year").mean("time")
    years = list(range(hindcast_period[0], hindcast_period[1] + 1))
    return annual.sel(year=[y for y in years if y in annual["year"].values])


def _to_gcm_array(gcm_ds, variable, hindcast_period):
    """Reshape rosetta GCM Dataset into (year, member, lat, lon) seasonal mean.

    Mirrors examples/demo_seasonal_mme.py::_seasonal_to_gcm: averages any
    leadtime-like dim out, then promotes the first available time dim to a
    `year` coord so the array carries (year, member, lat, lon).
    """
    da = gcm_ds[variable]
    keep = {"lat", "lon", "time", "member", "year",
            "forecast_reference_time", "init_time"}
    for dim in list(da.dims):
        if dim not in keep:
            da = da.mean(dim=dim)
    for dim in ("forecast_reference_time", "init_time", "time"):
        if dim in da.dims:
            da = da.assign_coords(year=(dim, da[dim].dt.year.values))
            da = da.swap_dims({dim: "year"}).drop_vars(dim)
            break
    years = list(range(hindcast_period[0], hindcast_period[1] + 1))
    return da.sel(year=[y for y in years if y in da["year"].values])


def _run_one_target(country_name: str, country, target: Target,
                    output_root: Path) -> dict:
    """Run seasonal_mme for one (country, season, init). Return status dict."""
    season = country.seasons[target.season]
    variable = country.predictand_var
    obs_ds = _fetch_obs(country, target)
    obs = _to_obs_array(obs_ds, variable, season.target_months, country.hindcast_period)

    # v1: single-track predictand only. Multi-track (e.g. adding an SST
    # track for PyCPT-style dual-predictor MME) replaces this dict literal.
    predictor_tracks: dict[str, dict[str, tuple]] = {variable: {}}
    for product in country.models:
        gcm_ds = _fetch_gcm(country, target, product)
        gcm = _to_gcm_array(gcm_ds, variable, country.hindcast_period)
        predictor_tracks[variable][product] = (gcm, None)

    result = africas2s.seasonal_mme(
        predictor_tracks,
        obs,
        method=country.method,
        cv=country.cv,
        cpt_args=country.cpt_args,
        verbose=True,
    )
    write_output(
        result=result,
        country=country_name,
        season=target.season,
        init_year=target.init_year,
        init_month=target.init_month,
        root=output_root,
    )
    return {
        "country": country_name, "season": target.season,
        "init_year": target.init_year, "init_month": target.init_month,
        "status": "ok",
    }


def _load_targets_file(path: Path, country_name: str) -> list[Target]:
    """Load explicit (country, season, init_year, init_month) tuples from a
    JSON file. Used by the rebench workflow to pin which historical forecasts
    to re-run, instead of letting select_targets pick today's in-range set.
    """
    raw = json.loads(path.read_text())
    out: list[Target] = []
    for entry in raw:
        c, s, iy, im = entry
        if c != country_name:
            continue
        out.append(Target(c, s, int(iy), int(im)))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--country", required=True)
    p.add_argument("--today", required=True, help="ISO date YYYY-MM-DD")
    p.add_argument("--output-root", required=True)
    p.add_argument("--config", default=str(Path(__file__).parent / "nightly.yml"))
    p.add_argument(
        "--kind", default="operational", choices=["operational", "rebench"],
        help="Forecast kind. operational = today's in-range targets via "
             "select_targets. rebench = re-run an explicit historical set "
             "from --targets-file. Default: operational.",
    )
    p.add_argument(
        "--targets-file", default=None,
        help="Path to a JSON file with explicit "
             "[[country, season, init_year, init_month], ...] tuples. "
             "Required when --kind=rebench; ignored for operational.",
    )
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.country not in cfg.countries:
        known = ", ".join(sorted(cfg.countries))
        print(
            f"[nightly] Unknown country {args.country!r}; known: {known}",
            file=sys.stderr,
        )
        return 2
    country = cfg.countries[args.country]
    today = date.fromisoformat(args.today)

    if args.kind == "rebench":
        if not args.targets_file:
            print(
                "[nightly] --kind=rebench requires --targets-file with the "
                "explicit (country, season, init_year, init_month) list.",
                file=sys.stderr,
            )
            return 2
        targets = _load_targets_file(Path(args.targets_file), args.country)
    else:
        targets = [t for t in select_targets(cfg, today) if t.country == args.country]

    if not targets:
        print(f"[nightly] No targets for {args.country} on {today} (kind={args.kind}); exit 0.")
        return 0

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    statuses: list[dict] = []
    for target in targets:
        try:
            statuses.append(_run_one_target(args.country, country, target, output_root))
            print(f"[nightly] OK {args.country} {target.season} init {target.init_year}-{target.init_month:02d}")
        except Exception as exc:
            traceback.print_exc()
            base = output_root / args.country / target.season / f"{target.init_year}-{target.init_month:02d}"
            base.mkdir(parents=True, exist_ok=True)
            (base / "status.json").write_text(json.dumps({
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }))
            statuses.append({
                "country": args.country, "season": target.season,
                "init_year": target.init_year, "init_month": target.init_month,
                "status": "failed", "reason": str(exc),
            })

    ok = [s for s in statuses if s["status"] == "ok"]
    if not ok:
        # Every target failed — let the runner mark this country job red.
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
