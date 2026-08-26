import numpy as np
import pytest
import xarray as xr


# ===================================================================
# 22. Reports, comparison, and SVSLRF rendering
# ===================================================================
# Sub-section: metric diagram capture
# -------------------------------------------------------------------

def test_roc_metric_compute_diagram(perfect_tercile_forecast, synthetic_obs):
    """compute_diagram returns per-tercile (fpr, tpr, area). Area matches
    the corresponding compute() scalar within float tolerance."""
    from africas2s.metrics.roc import ROCMetric

    metric = ROCMetric()
    scalars = metric.compute(perfect_tercile_forecast, synthetic_obs)
    diagram = metric.compute_diagram(perfect_tercile_forecast, synthetic_obs)

    assert set(diagram.keys()) == {"bn", "nn", "an"}
    for cat in ("bn", "nn", "an"):
        entry = diagram[cat]
        assert set(entry.keys()) == {"fpr", "tpr", "area"}
        assert isinstance(entry["fpr"], np.ndarray)
        assert isinstance(entry["tpr"], np.ndarray)
        assert entry["fpr"].shape == entry["tpr"].shape
        # area matches the compute() scalar
        scalar_key = f"roc_{cat}"
        assert abs(entry["area"] - scalars[scalar_key]) < 1e-9
        # monotone non-decreasing
        assert np.all(np.diff(entry["fpr"]) >= -1e-12)
        assert np.all(np.diff(entry["tpr"]) >= -1e-12)


def test_reliability_metric_compute_diagram(perfect_tercile_forecast, synthetic_obs):
    """compute_diagram returns a list of 3 per-tercile bin payloads."""
    from africas2s.metrics.reliability import ReliabilityMetric

    metric = ReliabilityMetric()
    diagram = metric.compute_diagram(perfect_tercile_forecast, synthetic_obs)

    assert isinstance(diagram, list)
    assert len(diagram) == 3
    labels = [d["tercile"] for d in diagram]
    assert labels == ["bn", "nn", "an"]
    for entry in diagram:
        assert "bins" in entry
        for b in entry["bins"]:
            assert set(b.keys()) == {"mean_prob", "obs_freq", "n"}
            assert 0.0 <= b["mean_prob"] <= 1.0
            assert 0.0 <= b["obs_freq"] <= 1.0
            assert b["n"] >= 1


def test_skill_report_has_metadata_and_diagrams_fields():
    """SkillReport must expose metadata and diagrams fields, default empty dicts."""
    from africas2s.skill import SkillReport
    report = SkillReport()
    assert report.metadata == {}
    assert report.diagrams == {}


def test_skill_populates_diagrams(climatology_forecast, synthetic_obs):
    """skill() must auto-populate diagrams for metrics with compute_diagram()."""
    from africas2s.skill import skill
    report = skill(climatology_forecast, synthetic_obs, metrics=["roc", "reliability"])
    assert "roc" in report.diagrams
    assert "reliability" in report.diagrams
    assert set(report.diagrams["roc"].keys()) == {"bn", "nn", "an"}
    assert len(report.diagrams["reliability"]) == 3


def test_skill_report_to_table():
    """to_table returns a flat metric/value DataFrame from scores."""
    import pandas as pd
    from africas2s.skill import SkillReport

    report = SkillReport(scores={"rpss": 0.42, "rmse": 1.7})
    df = report.to_table()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["metric", "value"]
    assert set(df["metric"]) == {"rpss", "rmse"}
    assert float(df.loc[df["metric"] == "rpss", "value"].iloc[0]) == 0.42


def test_skill_report_to_dict_roundtrip(climatology_forecast, synthetic_obs):
    """to_dict() produces a JSON-serializable nested-list payload covering
    scores, spatial, diagrams, and metadata."""
    import json
    from africas2s.skill import skill

    report = skill(
        climatology_forecast, synthetic_obs,
        metrics=["rpss", "roc", "reliability"], spatial=True,
    )
    report.metadata = {"region": "East Africa", "method": "BCSD"}

    d = report.to_dict()
    # json.dumps must succeed with no custom encoders
    s = json.dumps(d)
    assert isinstance(s, str)
    assert "rpss" in d["scores"]
    assert "rpss" in d["spatial"]
    sp = d["spatial"]["rpss"]
    assert set(sp.keys()) == {"dims", "coords", "values"}
    assert isinstance(sp["values"], list)
    assert "roc" in d["diagrams"]
    # ROC curves come through as plain lists
    assert isinstance(d["diagrams"]["roc"]["bn"]["fpr"], list)
    assert d["metadata"]["region"] == "East Africa"


def test_skill_report_to_geotiff(tmp_path, climatology_forecast, synthetic_obs):
    """to_geotiff writes a real GeoTIFF with EPSG:4326 CRS."""
    pytest.importorskip("rioxarray")
    import rioxarray  # noqa: F401  (registers the .rio accessor)
    import xarray as xr
    from africas2s.skill import skill

    report = skill(climatology_forecast, synthetic_obs, metrics=["rpss"], spatial=True)
    path = tmp_path / "rpss.tif"
    report.to_geotiff(path, metric="rpss")

    assert path.exists() and path.stat().st_size > 0
    reopened = xr.open_dataarray(path, engine="rasterio")
    assert reopened.rio.crs.to_epsg() == 4326


def test_skill_report_to_geotiff_missing_metric_raises(climatology_forecast, synthetic_obs, tmp_path):
    """Missing metric raises KeyError naming available metrics."""
    pytest.importorskip("rioxarray")
    from africas2s.skill import skill

    report = skill(climatology_forecast, synthetic_obs, metrics=["rpss"], spatial=True)
    with pytest.raises(KeyError, match="rpss"):
        report.to_geotiff(tmp_path / "nope.tif", metric="not_a_metric")


def test_skill_report_to_geotiff_scalar_only_raises(tmp_path):
    """Scalar-only report raises ValueError with helpful message."""
    pytest.importorskip("rioxarray")
    from africas2s.skill import SkillReport

    report = SkillReport(scores={"rpss": 0.5})  # no spatial maps
    with pytest.raises(ValueError, match="no spatial map"):
        report.to_geotiff(tmp_path / "x.tif", metric="rpss")


def test_reporting_subpackage_imports():
    """Reporting subpackage must import cleanly even without optional deps loaded."""
    import africas2s.reporting  # noqa: F401
    from africas2s.reporting._pages import _METRIC_STYLE  # noqa: F401
    # Sentinel entries that downstream primitives rely on
    assert "rpss" in _METRIC_STYLE
    assert _METRIC_STYLE["rpss"]["cmap"] == "RdBu"
    assert _METRIC_STYLE["rpss"]["vmin"] == -1
    assert _METRIC_STYLE["rpss"]["vmax"] == 1


def test_title_page_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    pypdf = pytest.importorskip("pypdf")
    from matplotlib.backends.backend_pdf import PdfPages
    from africas2s.reporting._pages import title_page

    path = tmp_path / "title.pdf"
    with PdfPages(path) as pdf:
        title_page(pdf, title="Test", subtitle="subtitle line",
                   metadata={"region": "East Africa", "target": "MAM"})
    reader = pypdf.PdfReader(str(path))
    assert len(reader.pages) == 1
    text = reader.pages[0].extract_text() or ""
    assert "Test" in text
    assert "East Africa" in text


def test_title_page_empty_metadata_renders(tmp_path):
    pytest.importorskip("matplotlib")
    from matplotlib.backends.backend_pdf import PdfPages
    from africas2s.reporting._pages import title_page

    path = tmp_path / "title.pdf"
    with PdfPages(path) as pdf:
        title_page(pdf, title="Test", subtitle=None, metadata={})
    assert path.stat().st_size > 0


def test_scalar_table_page_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    pypdf = pytest.importorskip("pypdf")
    from matplotlib.backends.backend_pdf import PdfPages
    from africas2s.reporting._pages import scalar_table_page

    path = tmp_path / "scalars.pdf"
    with PdfPages(path) as pdf:
        scalar_table_page(pdf, {"rpss": 0.18, "rmse": 2.4}, title="Mandatory triplet")
    reader = pypdf.PdfReader(str(path))
    assert len(reader.pages) == 1
    text = reader.pages[0].extract_text() or ""
    assert "rpss" in text
    assert "Mandatory triplet" in text


def test_map_grid_page_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    pypdf = pytest.importorskip("pypdf")
    from matplotlib.backends.backend_pdf import PdfPages
    from africas2s.reporting._pages import map_grid_page

    lat = np.linspace(-5, 5, 6)
    lon = np.linspace(30, 45, 8)
    maps = {
        "rpss": xr.DataArray(
            np.random.RandomState(0).uniform(-1, 1, (6, 8)),
            dims=["lat", "lon"], coords={"lat": lat, "lon": lon},
        ),
        "rmse": xr.DataArray(
            np.random.RandomState(1).uniform(0, 2, (6, 8)),
            dims=["lat", "lon"], coords={"lat": lat, "lon": lon},
        ),
    }

    path = tmp_path / "maps.pdf"
    with PdfPages(path) as pdf:
        map_grid_page(pdf, maps, ncols=2)
    reader = pypdf.PdfReader(str(path))
    assert len(reader.pages) == 1


def test_roc_page_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    pypdf = pytest.importorskip("pypdf")
    from matplotlib.backends.backend_pdf import PdfPages
    from africas2s.reporting._pages import roc_page

    roc_diagram = {
        "bn": {"fpr": np.array([0, 0.3, 1]), "tpr": np.array([0, 0.7, 1]), "area": 0.72},
        "nn": {"fpr": np.array([0, 0.5, 1]), "tpr": np.array([0, 0.5, 1]), "area": 0.50},
        "an": {"fpr": np.array([0, 0.2, 1]), "tpr": np.array([0, 0.8, 1]), "area": 0.78},
    }
    path = tmp_path / "roc.pdf"
    with PdfPages(path) as pdf:
        roc_page(pdf, roc_diagram)
    reader = pypdf.PdfReader(str(path))
    assert len(reader.pages) == 1


def test_reliability_page_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    pypdf = pytest.importorskip("pypdf")
    from matplotlib.backends.backend_pdf import PdfPages
    from africas2s.reporting._pages import reliability_page

    diagram = [
        {"tercile": "bn", "bins": [
            {"mean_prob": 0.2, "obs_freq": 0.22, "n": 10},
            {"mean_prob": 0.7, "obs_freq": 0.62, "n": 7},
        ]},
        {"tercile": "nn", "bins": [
            {"mean_prob": 0.3, "obs_freq": 0.33, "n": 12},
        ]},
        {"tercile": "an", "bins": [
            {"mean_prob": 0.5, "obs_freq": 0.5, "n": 9},
            {"mean_prob": 0.9, "obs_freq": 0.85, "n": 4},
        ]},
    ]
    path = tmp_path / "rel.pdf"
    with PdfPages(path) as pdf:
        reliability_page(pdf, diagram)
    reader = pypdf.PdfReader(str(path))
    assert len(reader.pages) == 1


def test_heatmap_page_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    pypdf = pytest.importorskip("pypdf")
    import pandas as pd
    from matplotlib.backends.backend_pdf import PdfPages
    from africas2s.reporting._pages import heatmap_page

    df = pd.DataFrame(
        {"rpss": [0.2, 0.1], "rmse": [1.5, 1.8]},
        index=pd.Index(["A", "B"], name="method"),
    )
    path = tmp_path / "hm.pdf"
    with PdfPages(path) as pdf:
        heatmap_page(pdf, df, title="Test heatmap")
    reader = pypdf.PdfReader(str(path))
    assert len(reader.pages) == 1


def test_comparison_map_grid_page_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    pypdf = pytest.importorskip("pypdf")
    from matplotlib.backends.backend_pdf import PdfPages
    from africas2s.reporting._pages import comparison_map_grid_page

    lat = np.linspace(-5, 5, 6)
    lon = np.linspace(30, 45, 8)
    def _da(seed):
        return xr.DataArray(
            np.random.RandomState(seed).uniform(-1, 1, (6, 8)),
            dims=["lat", "lon"], coords={"lat": lat, "lon": lon},
        )
    maps = {"A": _da(0), "B": _da(1)}

    path = tmp_path / "cmp_maps.pdf"
    with PdfPages(path) as pdf:
        comparison_map_grid_page(pdf, metric="rpss", maps=maps)
    reader = pypdf.PdfReader(str(path))
    assert len(reader.pages) == 1


# -------------------------------------------------------------------
# Sub-section: SVSLRF composition
# -------------------------------------------------------------------

def test_svslrf_render_minimal(tmp_path, climatology_forecast, synthetic_obs):
    """A scalar-only report renders to PDF (no spatial, no diagrams)."""
    pytest.importorskip("matplotlib")
    pypdf = pytest.importorskip("pypdf")
    from africas2s.skill import skill
    from africas2s.reporting.svslrf import render

    report = skill(climatology_forecast, synthetic_obs, metrics=["rpss"])
    path = tmp_path / "svslrf.pdf"
    render(report, path)

    assert path.stat().st_size > 0
    reader = pypdf.PdfReader(str(path))
    # cover + mandatory triplet are combined on one page; no diagrams,
    # no spatial, no secondary in this minimal report.
    assert len(reader.pages) >= 1


def test_svslrf_render_full(tmp_path, climatology_forecast, synthetic_obs):
    """A spatial report with roc + reliability + rpss renders all pages."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    pypdf = pytest.importorskip("pypdf")
    from africas2s.skill import skill
    from africas2s.reporting.svslrf import render

    report = skill(
        climatology_forecast, synthetic_obs,
        metrics=["rpss", "roc", "reliability"], spatial=True,
    )
    report.metadata = {"region": "East Africa", "method": "BCSD"}
    path = tmp_path / "svslrf_full.pdf"
    render(report, path)

    reader = pypdf.PdfReader(str(path))
    # cover+mandatory + diagrams (roc+reliability combined) + maps = 3 pages.
    # secondary is empty here (everything is in mandatory).
    assert len(reader.pages) >= 3

    # Title page must contain the region metadata
    text0 = reader.pages[0].extract_text() or ""
    assert "East Africa" in text0


def test_skill_report_to_pdf_smoke(tmp_path, climatology_forecast, synthetic_obs):
    pytest.importorskip("matplotlib")
    pypdf = pytest.importorskip("pypdf")
    from africas2s.skill import skill

    report = skill(climatology_forecast, synthetic_obs, metrics=["rpss"])
    path = tmp_path / "out.pdf"
    report.to_pdf(path)
    assert path.stat().st_size > 0
    reader = pypdf.PdfReader(str(path))
    assert len(reader.pages) >= 1


def test_skill_report_to_pdf_unknown_style_raises(tmp_path):
    from africas2s.skill import SkillReport
    report = SkillReport(scores={"rpss": 0.3})
    with pytest.raises(ValueError, match="unknown style"):
        report.to_pdf(tmp_path / "x.pdf", style="not_a_style")


def test_svslrf_includes_member_contributions_page_when_present(tmp_path):
    """When report.diagrams['member_contributions'] is present, the PDF
    gains an extra page vs the same report without it."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    pypdf = pytest.importorskip("pypdf")
    import numpy as np
    import xarray as xr
    from africas2s.skill import SkillReport
    from africas2s.reporting.svslrf import render

    # Minimal report — scores + a synthetic member_contributions diagram.
    coords = {"lat": np.linspace(-5, 5, 4), "lon": np.linspace(30, 40, 4)}
    member_contribs = {
        "A": {
            "correlation_with_mme_mean": 0.8,
            "skill_delta": xr.DataArray(
                np.full((4, 4), -0.1), dims=("lat", "lon"), coords=coords,
            ),
        },
        "B": {
            "correlation_with_mme_mean": 0.4,
            "skill_delta": xr.DataArray(
                np.full((4, 4), 0.1), dims=("lat", "lon"), coords=coords,
            ),
        },
    }

    base = SkillReport(scores={"rpss": 0.3})
    base.metadata = {"region": "Test"}
    base_path = tmp_path / "base.pdf"
    render(base, base_path)
    base_pages = len(pypdf.PdfReader(str(base_path)).pages)

    with_mc = SkillReport(scores={"rpss": 0.3})
    with_mc.metadata = {"region": "Test"}
    with_mc.diagrams = {"member_contributions": member_contribs}
    mc_path = tmp_path / "with_mc.pdf"
    render(with_mc, mc_path)
    mc_pages = len(pypdf.PdfReader(str(mc_path)).pages)

    assert mc_pages == base_pages + 1


def test_svslrf_omits_member_contributions_when_absent(tmp_path):
    """When member_contributions is not in diagrams, no extra page is added.
    This is the negative case of the test above."""
    pytest.importorskip("matplotlib")
    pypdf = pytest.importorskip("pypdf")
    from africas2s.skill import SkillReport
    from africas2s.reporting.svslrf import render

    report = SkillReport(scores={"rpss": 0.3})
    report.metadata = {"region": "Test"}
    path = tmp_path / "no_mc.pdf"
    render(report, path)
    # cover+triplet only; no diagrams, no spatial, no secondary.
    pages = len(pypdf.PdfReader(str(path)).pages)
    assert pages == 1


def test_skill_compare_basic(climatology_forecast, perfect_tercile_forecast, synthetic_obs):
    from africas2s.compare import skill_compare, ComparisonReport
    from africas2s.skill import SkillReport

    cmp = skill_compare(
        {"A": climatology_forecast, "B": perfect_tercile_forecast},
        synthetic_obs,
        metrics=["rpss"],
    )
    assert isinstance(cmp, ComparisonReport)
    assert cmp.methods == ["A", "B"]
    assert isinstance(cmp.reports["A"], SkillReport)
    assert "rpss" in cmp.reports["A"].scores
    assert "rpss" in cmp.reports["B"].scores


def test_skill_compare_empty_dict_raises(synthetic_obs):
    from africas2s.compare import skill_compare
    with pytest.raises(ValueError, match="at least one forecast"):
        skill_compare({}, synthetic_obs, metrics=["rpss"])


def test_skill_compare_grid_mismatch_raises(climatology_forecast, synthetic_obs):
    """A forecast on a shifted-lat grid raises ValueError naming the bad key."""
    from africas2s.compare import skill_compare

    shifted = climatology_forecast.assign_coords(
        lat=climatology_forecast["lat"].values + 10.0
    )
    with pytest.raises(ValueError, match="'B'"):
        skill_compare(
            {"A": climatology_forecast, "B": shifted},
            synthetic_obs,
            metrics=["rpss"],
        )


def test_skill_compare_to_table(climatology_forecast, perfect_tercile_forecast, synthetic_obs):
    from africas2s.compare import skill_compare

    cmp = skill_compare(
        {"A": climatology_forecast, "B": perfect_tercile_forecast},
        synthetic_obs, metrics=["rpss"],
    )
    df = cmp.to_table()
    assert list(df.index) == ["A", "B"]
    assert "rpss" in df.columns
    assert df.loc["A", "rpss"] != df.loc["B", "rpss"]


def test_skill_compare_to_heatmap_smoke(climatology_forecast, perfect_tercile_forecast, synthetic_obs, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from africas2s.compare import skill_compare

    cmp = skill_compare(
        {"A": climatology_forecast, "B": perfect_tercile_forecast},
        synthetic_obs, metrics=["rpss"],
    )
    fig = cmp.to_heatmap()
    assert fig is not None
    out = tmp_path / "heatmap.png"
    fig2 = cmp.to_heatmap(path=out)
    assert fig2 is not None
    assert out.exists() and out.stat().st_size > 0
    plt.close(fig)
    plt.close(fig2)


def test_skill_compare_to_pdf_smoke(climatology_forecast, perfect_tercile_forecast, synthetic_obs, tmp_path):
    pytest.importorskip("matplotlib")
    pypdf = pytest.importorskip("pypdf")
    from africas2s.compare import skill_compare

    cmp = skill_compare(
        {"A": climatology_forecast, "B": perfect_tercile_forecast},
        synthetic_obs, metrics=["rpss"],
    )
    path = tmp_path / "cmp.pdf"
    cmp.to_pdf(path)
    assert path.stat().st_size > 0
    reader = pypdf.PdfReader(str(path))
    n_default = len(reader.pages)
    assert n_default >= 2


def test_skill_compare_to_pdf_spatial_maps_smoke(climatology_forecast, perfect_tercile_forecast, synthetic_obs, tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    pypdf = pytest.importorskip("pypdf")
    from africas2s.compare import skill_compare

    cmp = skill_compare(
        {"A": climatology_forecast, "B": perfect_tercile_forecast},
        synthetic_obs, metrics=["rpss"], spatial=True,
    )
    no_maps = tmp_path / "cmp_no_maps.pdf"
    with_maps = tmp_path / "cmp_with_maps.pdf"
    cmp.to_pdf(no_maps)
    cmp.to_pdf(with_maps, spatial_maps=True)

    n_no = len(pypdf.PdfReader(str(no_maps)).pages)
    n_yes = len(pypdf.PdfReader(str(with_maps)).pages)
    assert n_yes > n_no


# ---------------------------------------------------------------------------
# §22 – Top-level re-exports
# ---------------------------------------------------------------------------

def test_top_level_reexports():
    """skill_compare and ComparisonReport are importable from africas2s."""
    import africas2s
    assert hasattr(africas2s, "skill_compare")
    assert hasattr(africas2s, "ComparisonReport")
    # Sanity: the re-exports are the same objects as the canonical ones
    from africas2s.compare import skill_compare, ComparisonReport
    assert africas2s.skill_compare is skill_compare
    assert africas2s.ComparisonReport is ComparisonReport
