"""Gather-job glue: artifacts/<country>/<season>/<init>/ -> site/.

`publish()` is the unit-tested core. The CLI wrapper at the bottom is the
entry point invoked from the workflow's gather job after artifact download.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any


def _atomic_write_text(path: Path, content: str) -> None:
    """Write text atomically via a tempfile + rename so a crash mid-write
    cannot leave a half-written JSON manifest behind.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


def _spec_json_dumps(obj: Any, **kw: Any) -> str:
    """json.dumps that emits spec-valid JSON. Python's default emits bare
    NaN/Infinity tokens that JSON.parse rejects in the browser, which silently
    blanks the entire dashboard. Replace non-finite floats with null and use
    allow_nan=False so a future stray NaN raises loudly instead of slipping
    through.
    """
    def clean(o: Any) -> Any:
        if isinstance(o, float):
            return o if math.isfinite(o) else None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o
    return json.dumps(clean(obj), allow_nan=False, **kw)


def publish(
    *,
    artifacts_root: Path,
    site_root: Path,
    run_date: str,
    commit_sha: str,
    expected: list[tuple[str, str, int, int]],
    kind: str = "operational",
) -> None:
    """Copy artifacts into site/ and append one metrics.json row per (country,
    season) in `expected`.

    `expected` is the list of (country, season, init_year, init_month) tuples
    the matrix should have produced; missing artifacts become status=failed.

    `kind` distinguishes operational (forecast as-issued at its init date) from
    rebench (re-run of a historical forecast under current africas2s). Rows of
    both kinds coexist in metrics.json; the dashboard filters on `kind`.

    No dedupe: every call appends rows. Re-running publish for the same set
    will produce duplicate rows. The dashboard handles dedupe at read time.
    """
    site_root.mkdir(parents=True, exist_ok=True)

    # Copy/refresh static site assets from scripts/nightly/site/ into site_root.
    # Fail loud if a template is missing — the three filenames are a hard
    # contract; a silent skip would ship a broken dashboard to gh-pages.
    site_src = Path(__file__).resolve().parent / "site"
    for fname in ("index.html", "app.js", "style.css"):
        fsrc = site_src / fname
        if not fsrc.is_file():
            raise FileNotFoundError(f"missing site template: {fsrc}")
        shutil.copy2(fsrc, site_root / fname)

    metrics_path = site_root / "metrics.json"
    existing: list[dict] = (
        json.loads(metrics_path.read_text()) if metrics_path.exists() else []
    )

    new_rows: list[dict] = []
    index_entries: list[dict] = []

    for country, season, iy, im in expected:
        init_str = f"{iy}-{im:02d}"
        src = artifacts_root / country / season / init_str
        if src.is_dir() and (src / "skill_metrics.json").is_file():
            metrics = json.loads((src / "skill_metrics.json").read_text())
            dest = site_root / "forecasts" / country / season / init_str
            dest.mkdir(parents=True, exist_ok=True)
            for fname in ("tercile_map.png", "forecast.nc"):
                fsrc = src / fname
                if fsrc.is_file():
                    shutil.copy2(fsrc, dest / fname)
            new_rows.append({
                "kind": kind,
                "date": run_date,
                "commit": commit_sha,
                "country": country,
                "season": season,
                "init": init_str,
                "status": "ok",
                "metrics": metrics,
                "forecast_dir": f"forecasts/{country}/{season}/{init_str}",
            })
            index_entries.append(
                {"country": country, "season": season, "init": init_str}
            )
        else:
            new_rows.append({
                "kind": kind,
                "date": run_date,
                "commit": commit_sha,
                "country": country,
                "season": season,
                "init": init_str,
                "status": "failed",
                "metrics": None,
                "forecast_dir": None,
                "reason": "artifact_missing",
            })

    _atomic_write_text(metrics_path, _spec_json_dumps(existing + new_rows, indent=2))

    # forecasts/index.json — flat list of all forecast folders that exist on
    # the site after this run. Rebuilt every time so removals (if we ever add
    # retention) would propagate.
    forecasts_root = site_root / "forecasts"
    forecasts_root.mkdir(parents=True, exist_ok=True)
    full_index = list(index_entries)
    # Include pre-existing entries too.
    for c_dir in sorted(forecasts_root.iterdir()):
        if not c_dir.is_dir():
            continue
        for s_dir in sorted(c_dir.iterdir()):
            if not s_dir.is_dir():
                continue
            for i_dir in sorted(s_dir.iterdir()):
                if not i_dir.is_dir():
                    continue
                entry = {
                    "country": c_dir.name,
                    "season": s_dir.name,
                    "init": i_dir.name,
                }
                if entry not in full_index:
                    full_index.append(entry)
    _atomic_write_text(forecasts_root / "index.json", _spec_json_dumps(full_index, indent=2))


def _cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts-root", required=True)
    p.add_argument("--site-root", required=True)
    p.add_argument("--run-date", required=True)
    p.add_argument("--commit-sha", required=True)
    p.add_argument(
        "--expected", required=True,
        help="JSON list of [country, season, init_year, init_month] tuples",
    )
    p.add_argument(
        "--kind", default="operational", choices=["operational", "rebench"],
        help="Row kind. `operational` = forecast as-issued at its init date. "
             "`rebench` = re-run of a historical forecast under current africas2s.",
    )
    args = p.parse_args()
    expected = [tuple(x) for x in json.loads(args.expected)]
    publish(
        artifacts_root=Path(args.artifacts_root),
        site_root=Path(args.site_root),
        run_date=args.run_date,
        commit_sha=args.commit_sha,
        expected=expected,
        kind=args.kind,
    )


if __name__ == "__main__":
    _cli()
