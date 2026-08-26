"""S2S testbed config loader.

The config is a single YAML file (default ``scripts/s2s/s2s.yml``) holding
per-country method allowlists, bounding boxes, and rosetta product IDs,
plus a handful of global fields (lead-day range, climatology baseline
window, store root).

Loading the config validates that every method named per country is
registered in africas2s's method registry — catching typos at start-up
rather than mid-run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml


@dataclass(frozen=True)
class CountryConfig:
    name: str
    bbox: dict
    methods: list[str]
    obs: str
    forecast: str
    variable: str
    obs_live: str | None = None  # Optional: realtime obs source for recent dekads.


@dataclass(frozen=True)
class S2SConfig:
    countries: Mapping[str, CountryConfig]
    lead_days: tuple[int, int]
    climatology_years: tuple[int, int]
    store_root: str
    comparison_window_days: int = 90


def load_config(path: str | Path) -> S2SConfig:
    raw = yaml.safe_load(Path(path).read_text())
    from africas2s.registry import _METHODS  # noqa: WPS433 — internal but stable.
    known_methods = set(_METHODS) | {"raw"}  # "raw" is regrid-only, no method class.

    countries: dict[str, CountryConfig] = {}
    for name, c in raw["countries"].items():
        unknown = [m for m in c["methods"] if m not in known_methods]
        if unknown:
            raise ValueError(
                f"country {name!r}: unknown method(s) {unknown}; "
                f"registered methods are {sorted(known_methods)}"
            )
        countries[name] = CountryConfig(
            name=name,
            bbox=dict(c["bbox"]),
            methods=list(c["methods"]),
            obs=c["obs"],
            forecast=c["forecast"],
            variable=c["variable"],
            obs_live=c.get("obs_live"),
        )

    return S2SConfig(
        countries=countries,
        lead_days=(int(raw["lead_days"]["min"]), int(raw["lead_days"]["max"])),
        climatology_years=tuple(raw["climatology_years"]),
        store_root=os.environ.get("S2S_STORE_ROOT", str(raw["store_root"])),
        comparison_window_days=int(raw.get("comparison_window_days", 90)),
    )
