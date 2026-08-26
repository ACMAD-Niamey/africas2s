"""Unit tests for africas2s.Index (teleconnection SST indices)."""
import numpy as np
import pytest
import xarray as xr

from africas2s import Index


def _sst(years, lat, lon, values):
    return xr.DataArray(
        values, dims=["year", "lat", "lon"],
        coords={"year": years, "lat": lat, "lon": lon},
    )


@pytest.fixture
def global_sst():
    """Global-ish SST grid (0-360 lon) with a controllable per-year signal."""
    years = np.arange(2000, 2020)
    lat = np.arange(-40, 41, 5.0)
    lon = np.arange(0, 360, 5.0)
    rng = np.random.default_rng(0)
    base = rng.normal(27.0, 0.5, (len(years), len(lat), len(lon)))
    return _sst(years, lat, lon, base)


def test_named_unknown_raises():
    with pytest.raises(KeyError):
        Index.named("not_an_index")


def test_wvg_returns_year_series(global_sst):
    idx = Index.named("wvg").reduce(global_sst)
    assert idx.dims == ("year",)
    assert idx.sizes["year"] == global_sst.sizes["year"]
    assert idx.name == "wvg"


def test_wvg_is_standardized_anomaly(global_sst):
    """Self-referenced WVG is a combination of z-scores → ~zero mean."""
    idx = Index.named("wvg").reduce(global_sst)
    assert abs(float(idx.mean())) < 1e-6


def test_wvg_matches_manual_formula(global_sst):
    """Default WVG == z(nino34) - (z(wnp)+z(wep)+z(wsp))/3 (3-box) with plain box means."""
    boxes = {
        "nino34": dict(south=-5, north=5, west=190, east=240),
        "wnp": dict(south=20, north=35, west=160, east=210),
        "wep": dict(south=-15, north=20, west=120, east=160),
        "wsp": dict(south=-30, north=-15, west=155, east=210),
    }
    z = {}
    for name, b in boxes.items():
        sub = global_sst.where(
            (global_sst.lat >= b["south"]) & (global_sst.lat <= b["north"])
            & (global_sst.lon >= b["west"]) & (global_sst.lon <= b["east"])
        )
        series = sub.mean(["lat", "lon"], skipna=True)
        z[name] = (series - series.mean()) / series.std()
    expected = z["nino34"] - (z["wnp"] + z["wep"] + z["wsp"]) / 3.0
    got = Index.named("wvg").reduce(global_sst)
    np.testing.assert_allclose(got.values, expected.values, rtol=1e-10)


def test_custom_wvg_matches_named_definition(global_sst):
    custom = Index.custom(
        name="wvg_candidate",
        regions={
            "nino34": [-5, 5, 190, 240],
            "wnp": [20, 35, 160, 210],
            "wep": [-15, 20, 120, 160],
            "wsp": [-30, -15, 155, 210],
        },
        combine=lambda z: z["nino34"] - (z["wnp"] + z["wep"] + z["wsp"]) / 3,
    )
    np.testing.assert_allclose(
        custom.reduce(global_sst).values,
        Index.named("wvg").reduce(global_sst).values,
        rtol=1e-10,
    )


def test_custom_requires_regions():
    with pytest.raises(ValueError, match="at least one region"):
        Index.custom(name="empty", regions={}, combine=lambda z: z["x"])


def test_wvg2_is_the_two_box_variant(global_sst):
    """wvg2 drops the WSP box; it differs from the 3-box default."""
    i3 = Index.named("wvg").reduce(global_sst)
    i2 = Index.named("wvg2").reduce(global_sst)
    assert not np.allclose(i3.values, i2.values)


def test_climatology_reuses_reference_stats(global_sst):
    """reduce(fcst, climatology=hcst) standardizes the forecast box means using
    the hindcast mean/std (so the two indices share a scale)."""
    hcst = global_sst
    # A forecast year with a deliberately warm Nino3.4 box.
    fcst = global_sst.isel(year=[0]).copy()
    nino_box = ((fcst.lat >= -5) & (fcst.lat <= 5)
                & (fcst.lon >= 190) & (fcst.lon <= 240))
    fcst = fcst + xr.where(nino_box, 3.0, 0.0)

    idx_f = Index.named("wvg").reduce(fcst, climatology=hcst)
    assert idx_f.sizes["year"] == 1
    # Warming the Nino3.4 box pushes WVG = z(nino34) - ... strongly positive.
    assert float(idx_f.values.reshape(-1)[0]) > 1.0


def test_lon_convention_invariance(global_sst):
    """-180..180 input gives the same WVG as the 0-360 input."""
    idx_360 = Index.named("wvg").reduce(global_sst)
    shifted = global_sst.assign_coords(
        lon=(((global_sst.lon + 180) % 360) - 180)
    ).sortby("lon")
    idx_180 = Index.named("wvg").reduce(shifted)
    np.testing.assert_allclose(
        idx_360.values, idx_180.values, rtol=1e-9, atol=1e-9
    )


def test_member_dim_is_averaged():
    """A member dimension is reduced out before forming the index."""
    years = np.arange(2000, 2010)
    lat = np.arange(-40, 41, 5.0)
    lon = np.arange(0, 360, 5.0)
    rng = np.random.default_rng(1)
    single = _sst(years, lat, lon, rng.normal(27, 0.5, (len(years), len(lat), len(lon))))
    multi = single.expand_dims(member=[0, 1, 2]).transpose("year", "member", "lat", "lon")
    # Identical members → index equals the single-field index.
    np.testing.assert_allclose(
        Index.named("wvg").reduce(multi).values,
        Index.named("wvg").reduce(single).values,
        rtol=1e-9,
    )


def test_nino34_single_box_is_unit_variance(global_sst):
    idx = Index.named("nino34").reduce(global_sst)
    assert idx.dims == ("year",)
    np.testing.assert_allclose(float(idx.std(ddof=0)), 1.0, rtol=1e-6)
