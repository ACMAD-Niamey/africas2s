"""End-to-end reporting cluster test.

Exercises SkillReport export methods, skill_compare, and SVSLRF PDF
rendering on a real-enough synthetic pipeline. Skipped automatically if
the [plotting] extra isn't installed.
"""

import json

import numpy as np
import pytest
import xarray as xr


pytestmark = pytest.mark.integration


def _make_terc_forecast(seed, lat, lon, year):
    """Return a synthetic tercile probability forecast with a deliberate
    bias so different seeds give different skill numbers."""
    rng = np.random.RandomState(seed)
    n_year = len(year)
    n_lat = len(lat)
    n_lon = len(lon)
    raw = rng.dirichlet([1.0, 1.0, 1.0], size=(n_year, n_lat, n_lon))
    # raw shape: (year, lat, lon, tercile); transpose to (tercile, year, lat, lon)
    raw = raw.transpose(3, 0, 1, 2)
    return xr.DataArray(
        raw,
        dims=["tercile", "year", "lat", "lon"],
        coords={"tercile": [0, 1, 2], "year": year, "lat": lat, "lon": lon},
    )


@pytest.fixture
def reporting_inputs():
    rng = np.random.RandomState(0)
    year = np.arange(2000, 2020)
    lat = np.linspace(-5, 5, 6)
    lon = np.linspace(30, 45, 8)
    obs = xr.DataArray(
        rng.standard_normal((len(year), len(lat), len(lon))),
        dims=["year", "lat", "lon"],
        coords={"year": year, "lat": lat, "lon": lon},
    )
    fcst_a = _make_terc_forecast(1, lat, lon, year)
    fcst_b = _make_terc_forecast(2, lat, lon, year)
    return fcst_a, fcst_b, obs


def test_reporting_cluster_end_to_end(tmp_path, reporting_inputs):
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    pytest.importorskip("rioxarray")
    pypdf = pytest.importorskip("pypdf")

    from africas2s.skill import skill
    from africas2s.compare import skill_compare

    fcst_a, fcst_b, obs = reporting_inputs

    # Single-method path
    report = skill(fcst_a, obs, metrics="svslrf", spatial=True)
    report.metadata = {
        "region": "Synthetic region",
        "target": "JAS test target",
        "init": "January 2000",
        "predictand": "synthetic anomaly",
        "method": "synthetic A",
    }
    pdf_path = tmp_path / "single.pdf"
    report.to_pdf(pdf_path)
    geotiff_path = tmp_path / "rpss.tif"
    report.to_geotiff(geotiff_path, metric="rpss")

    # Round-trip the dict
    dumped = json.dumps(report.to_dict())
    assert isinstance(dumped, str) and len(dumped) > 0

    # Comparison path
    cmp = skill_compare(
        {"A": fcst_a, "B": fcst_b}, obs,
        metrics="svslrf", spatial=True,
    )
    compare_path = tmp_path / "compare.pdf"
    cmp.to_pdf(compare_path, spatial_maps=True)

    # All outputs exist and are nonzero
    for p in (pdf_path, geotiff_path, compare_path):
        assert p.exists(), f"missing {p}"
        assert p.stat().st_size > 0, f"empty {p}"

    # PDFs open and have >= 1 page
    for p in (pdf_path, compare_path):
        reader = pypdf.PdfReader(str(p))
        assert len(reader.pages) >= 1
