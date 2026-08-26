"""Unit tests for scripts/nightly/* — pure functions, no network."""
from __future__ import annotations

from pathlib import Path

import pytest

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.nightly.config import Config, load_config


def test_load_config_parses_nightly_yaml():
    cfg = load_config(REPO_ROOT / "scripts" / "nightly" / "nightly.yml")
    assert isinstance(cfg, Config)
    assert set(cfg.countries) == {"kenya", "ethiopia", "nigeria"}
    kenya = cfg.countries["kenya"]
    assert kenya.seasons["MAM"].init_months == [12, 1, 2]
    assert kenya.seasons["MAM"].season_start_month == 3
    assert kenya.method == "cca"
    assert kenya.cv == "loyo"
    assert kenya.hindcast_period == (2000, 2016)


def test_load_config_rejects_missing_required_field(tmp_path):
    bad = tmp_path / "bad.yml"
    bad.write_text("countries:\n  kenya: {}\n")
    with pytest.raises(ValueError) as exc:
        load_config(bad)
    msg = str(exc.value)
    # Aggregated error must list multiple missing keys, not just the first.
    for required in ("models", "hindcast_period", "cv", "method"):
        assert required in msg, f"missing field {required!r} not surfaced in: {msg}"


from datetime import date

from scripts.nightly.select_targets import Target, select_targets


@pytest.fixture
def cfg():
    return load_config(REPO_ROOT / "scripts" / "nightly" / "nightly.yml")


def test_kenya_mam_normal_init_month(cfg):
    # In Feb 2026, MAM init=Feb 2026 should be selected.
    targets = select_targets(cfg, date(2026, 2, 15))
    kenya_mam = [t for t in targets if t.country == "kenya" and t.season == "MAM"]
    assert len(kenya_mam) == 1
    assert kenya_mam[0] == Target("kenya", "MAM", 2026, 2)


def test_kenya_mam_cross_year(cfg):
    # In early Jan 2026, the latest *in-range* init for MAM is Jan 2026.
    # The Dec entry (which would be Dec 2025) should not be picked when a
    # newer Jan 2026 init is available.
    targets = select_targets(cfg, date(2026, 1, 10))
    kenya_mam = [t for t in targets if t.country == "kenya" and t.season == "MAM"]
    assert kenya_mam == [Target("kenya", "MAM", 2026, 1)]


def test_kenya_mam_only_december_available(cfg):
    # In Dec 2025, the only MAM init available so far is Dec 2025 itself.
    targets = select_targets(cfg, date(2025, 12, 20))
    kenya_mam = [t for t in targets if t.country == "kenya" and t.season == "MAM"]
    assert kenya_mam == [Target("kenya", "MAM", 2025, 12)]


def test_kenya_mam_skipped_when_season_underway(cfg):
    # April is past MAM's season_start_month (March). No MAM forecast tonight.
    targets = select_targets(cfg, date(2026, 4, 10))
    assert not any(t.country == "kenya" and t.season == "MAM" for t in targets)


def test_nigeria_jja_skipped_when_no_init_available(cfg):
    # Nigeria JJA init_months = [3,4,5]. In January, no init is available.
    targets = select_targets(cfg, date(2026, 1, 10))
    assert not any(t.country == "nigeria" and t.season == "JJA" for t in targets)


def test_ethiopia_jja_latest_init(cfg):
    # In May, the latest JJA init available is May.
    targets = select_targets(cfg, date(2026, 5, 12))
    et_jja = [t for t in targets if t.country == "ethiopia" and t.season == "JJA"]
    assert et_jja == [Target("ethiopia", "JJA", 2026, 5)]


def test_init_months_cover_every_month(cfg):
    # With all 12 overlapping 3-month seasons, every calendar month should
    # have multiple in-range targets per country — no gap months.
    months_with_any = [
        m for m in range(1, 13)
        if select_targets(cfg, date(2026, m, 15))
    ]
    assert months_with_any == list(range(1, 13)), (
        f"expected every month covered, got {months_with_any}"
    )


import json

import numpy as np
import xarray as xr


def _make_fake_result():
    """Build a minimal stand-in for SeasonalMMEResult without invoking the
    full africas2s pipeline. We only exercise the fields output_writer reads.
    """
    from types import SimpleNamespace

    lat = np.linspace(-5, 5, 4)
    lon = np.linspace(33, 42, 4)
    tercile_forecast = xr.DataArray(
        np.full((3, 4, 4), 1 / 3, dtype=float),
        dims=("tercile", "lat", "lon"),
        coords={"tercile": [0, 1, 2], "lat": lat, "lon": lon},
    )
    tercile_cv = xr.DataArray(
        np.full((5, 3, 4, 4), 1 / 3, dtype=float),
        dims=("year", "tercile", "lat", "lon"),
        coords={"year": list(range(2012, 2017)),
                "tercile": [0, 1, 2], "lat": lat, "lon": lon},
    )
    forecast = xr.DataArray(
        np.zeros((4, 4), dtype=float),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
    )
    skill_report = SimpleNamespace(
        scores={"rpss": 0.36, "roc_area": 0.71, "pearson": 0.48},
        metadata={},
    )
    return SimpleNamespace(
        forecast=forecast,
        tercile_forecast=tercile_forecast,
        tercile_cv=tercile_cv,
        skill_report=skill_report,
        metadata={
            "years_used": list(range(1993, 2017)),
            "forecast_year": 2026,
            "method": "cca",
            "cv": "loyo",
            "n_members": 1,
            "tercile_method": "cpt",
            "tracks": ["prcp"],
            "run_at": "2026-05-16T02:14:00Z",
        },
    )


def test_output_writer_creates_expected_tree(tmp_path):
    from scripts.nightly.output_writer import write_output

    result = _make_fake_result()
    out = write_output(
        result=result,
        country="kenya",
        season="MAM",
        init_year=2026,
        init_month=2,
        root=tmp_path,
    )
    base = tmp_path / "kenya" / "MAM" / "2026-02"
    assert base.is_dir()
    # Catch silently empty/corrupt PNG or netCDF saves.
    assert (base / "tercile_map.png").stat().st_size > 0
    nc_path = base / "forecast.nc"
    assert nc_path.stat().st_size > 0

    with xr.open_dataset(nc_path) as ds:
        assert "tercile_forecast" in ds.data_vars
        assert "forecast" in ds.data_vars
        assert ds.attrs["country"] == "kenya"
        assert ds.attrs["season"] == "MAM"
        assert ds.attrs["init_year"] == 2026
        assert ds.attrs["init_month"] == 2
        assert ds.attrs["method"] == "cca"

    metrics = json.loads((base / "skill_metrics.json").read_text())
    assert metrics["rpss"] == pytest.approx(0.36)
    assert metrics["roc_area"] == pytest.approx(0.71)
    assert metrics["pearson"] == pytest.approx(0.48)
    assert out == base


def _seed_artifact_tree(root: Path, country: str, season: str, init: str,
                        metrics: dict[str, float]):
    base = root / country / season / init
    base.mkdir(parents=True, exist_ok=True)
    (base / "skill_metrics.json").write_text(json.dumps(metrics))
    (base / "tercile_map.png").write_bytes(b"\x89PNG\r\n\x1a\n")  # marker
    (base / "forecast.nc").write_bytes(b"CDF")
    return base


def test_publish_appends_metrics_row(tmp_path):
    from scripts.nightly.publish import publish

    artifacts = tmp_path / "artifacts"
    _seed_artifact_tree(
        artifacts, "kenya", "MAM", "2026-02",
        metrics={"rpss": 0.36, "roc_area": 0.71, "pearson": 0.48},
    )
    site = tmp_path / "site"
    site.mkdir()

    publish(
        artifacts_root=artifacts,
        site_root=site,
        run_date="2026-05-16",
        commit_sha="a540133",
        expected=[("kenya", "MAM", 2026, 2)],
    )

    metrics_path = site / "metrics.json"
    rows = json.loads(metrics_path.read_text())
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "operational"
    assert row["date"] == "2026-05-16"
    assert row["commit"] == "a540133"
    assert row["country"] == "kenya"
    assert row["season"] == "MAM"
    assert row["init"] == "2026-02"
    assert row["status"] == "ok"
    assert row["metrics"]["rpss"] == pytest.approx(0.36)
    assert (site / "forecasts" / "kenya" / "MAM" / "2026-02" / "tercile_map.png").is_file()


def test_publish_marks_missing_country_failed(tmp_path):
    from scripts.nightly.publish import publish

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    site = tmp_path / "site"
    site.mkdir()

    publish(
        artifacts_root=artifacts,
        site_root=site,
        run_date="2026-05-16",
        commit_sha="a540133",
        expected=[
            ("kenya", "MAM", 2026, 2),
            ("ethiopia", "JJA", 2026, 5),
        ],
    )
    rows = json.loads((site / "metrics.json").read_text())
    assert len(rows) == 2
    statuses = {(r["country"], r["season"]): r["status"] for r in rows}
    assert statuses[("kenya", "MAM")] == "failed"
    assert statuses[("ethiopia", "JJA")] == "failed"


def test_publish_writes_index_json(tmp_path):
    from scripts.nightly.publish import publish

    artifacts = tmp_path / "artifacts"
    _seed_artifact_tree(
        artifacts, "kenya", "MAM", "2026-02",
        metrics={"rpss": 0.36},
    )
    site = tmp_path / "site"
    site.mkdir()

    publish(
        artifacts_root=artifacts,
        site_root=site,
        run_date="2026-05-16",
        commit_sha="a540133",
        expected=[("kenya", "MAM", 2026, 2)],
    )

    index = json.loads((site / "forecasts" / "index.json").read_text())
    assert {"country": "kenya", "season": "MAM", "init": "2026-02"} in index


def test_publish_appends_not_overwrites(tmp_path):
    from scripts.nightly.publish import publish

    artifacts = tmp_path / "artifacts"
    _seed_artifact_tree(
        artifacts, "kenya", "MAM", "2026-02",
        metrics={"rpss": 0.36},
    )
    site = tmp_path / "site"
    site.mkdir()
    (site / "metrics.json").write_text(json.dumps([
        {"date": "2026-05-15", "commit": "old", "country": "kenya",
         "season": "MAM", "init": "2026-02", "status": "ok",
         "metrics": {"rpss": 0.30}, "forecast_dir": "forecasts/kenya/MAM/2026-02"},
    ]))

    publish(
        artifacts_root=artifacts,
        site_root=site,
        run_date="2026-05-16",
        commit_sha="new",
        expected=[("kenya", "MAM", 2026, 2)],
    )
    rows = json.loads((site / "metrics.json").read_text())
    assert len(rows) == 2
    assert rows[0]["date"] == "2026-05-15"
    assert rows[1]["date"] == "2026-05-16"


def test_publish_appends_unconditionally(tmp_path):
    """Re-running publish with identical inputs SHOULD duplicate rows.

    The A+B dual-mode design intentionally drops idempotence — every call
    appends, and the dashboard handles dedupe at read time. This pins that
    behavior so a future "helpful" idempotence reintroduction is caught.
    """
    from scripts.nightly.publish import publish

    artifacts = tmp_path / "artifacts"
    _seed_artifact_tree(
        artifacts, "kenya", "MAM", "2026-02",
        metrics={"rpss": 0.36},
    )
    site = tmp_path / "site"
    site.mkdir()

    kwargs = dict(
        artifacts_root=artifacts,
        site_root=site,
        run_date="2026-05-16",
        commit_sha="abc",
        expected=[("kenya", "MAM", 2026, 2)],
    )
    publish(**kwargs)
    publish(**kwargs)

    rows = json.loads((site / "metrics.json").read_text())
    assert len(rows) == 2
    assert all(r["commit"] == "abc" for r in rows)


def test_publish_emits_spec_valid_json_when_metrics_contain_nan(tmp_path):
    """metrics.json must be parseable by browsers' JSON.parse, which rejects
    bare NaN/Infinity tokens that Python's json.dumps emits by default. When a
    skill metric computes to NaN (e.g. zero variance on a tile), publish must
    substitute null so the dashboard keeps rendering. Regression test for the
    silent blank-dashboard bug on the 2026-05-18 nightly run.
    """
    from scripts.nightly.publish import publish

    artifacts = tmp_path / "artifacts"
    _seed_artifact_tree(
        artifacts, "kenya", "MAM", "2026-02",
        metrics={"rpss": float("nan"), "roc_area": 0.71, "pearson": float("inf")},
    )
    site = tmp_path / "site"
    site.mkdir()

    publish(
        artifacts_root=artifacts,
        site_root=site,
        run_date="2026-05-18",
        commit_sha="deadbee",
        expected=[("kenya", "MAM", 2026, 2)],
    )

    raw = (site / "metrics.json").read_text()
    assert "NaN" not in raw
    assert "Infinity" not in raw
    # Strict-mode json.loads rejects bare NaN/Infinity the same way browsers do.
    import json as _json
    rows = _json.loads(raw, parse_constant=_strict_parse_constant)
    assert rows[0]["metrics"]["rpss"] is None
    assert rows[0]["metrics"]["pearson"] is None
    assert rows[0]["metrics"]["roc_area"] == pytest.approx(0.71)


def _strict_parse_constant(c):  # used by the spec-JSON test above
    raise ValueError(f"non-spec JSON token: {c}")


def test_publish_writes_rebench_kind(tmp_path):
    """publish with kind='rebench' tags rows accordingly so the dashboard can
    distinguish operational issuances from retrospective re-runs."""
    from scripts.nightly.publish import publish

    artifacts = tmp_path / "artifacts"
    _seed_artifact_tree(
        artifacts, "kenya", "MAM", "2026-02",
        metrics={"rpss": 0.41},
    )
    site = tmp_path / "site"
    site.mkdir()

    publish(
        artifacts_root=artifacts,
        site_root=site,
        run_date="2026-08-01",
        commit_sha="bench1",
        expected=[("kenya", "MAM", 2026, 2)],
        kind="rebench",
    )

    rows = json.loads((site / "metrics.json").read_text())
    assert len(rows) == 1
    assert rows[0]["kind"] == "rebench"
    assert rows[0]["commit"] == "bench1"


def test_publish_copies_site_templates(tmp_path):
    from scripts.nightly.publish import publish

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    site = tmp_path / "site"
    site.mkdir()
    publish(
        artifacts_root=artifacts, site_root=site,
        run_date="2026-05-16", commit_sha="abc", expected=[],
    )
    for fname in ("index.html", "app.js", "style.css"):
        assert (site / fname).is_file(), fname
