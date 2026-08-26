"""Index generalization: transform modes, area weighting, baselines, new names.

The tests in `test_indices.py` pin the pre-existing WVG behaviour and must keep
passing unchanged — that is the back-compatibility contract. These cover the
axes that were previously hardcoded.
"""
import numpy as np
import pytest
import xarray as xr

from africas2s import Index
from africas2s.indices import REGIONS


def _field(years, lat, lon, values):
    return xr.DataArray(
        values, dims=["year", "lat", "lon"],
        coords={"year": years, "lat": lat, "lon": lon},
    )


@pytest.fixture
def global_sst():
    years = np.arange(1981, 2027)
    lat = np.arange(-40, 41, 2.0)
    lon = np.arange(0, 360, 2.0)
    rng = np.random.default_rng(0)
    return _field(years, lat, lon, rng.normal(27.0, 0.5, (len(years), len(lat), len(lon))))


def _box_mean(field, box, weighted=False):
    """Reference box mean, computed independently of Index internals."""
    lon360 = field.assign_coords(lon=field.lon % 360).sortby("lon")
    lon_w, lon_e = box["west"] % 360, box["east"] % 360
    if lon_w < lon_e:
        lon_mask = (lon360.lon >= lon_w) & (lon360.lon <= lon_e)
    elif lon_w > lon_e:
        lon_mask = (lon360.lon >= lon_w) | (lon360.lon <= lon_e)
    else:
        lon_mask = xr.ones_like(lon360.lon, dtype=bool)
    sub = lon360.where(
        (lon360.lat >= box["south"]) & (lon360.lat <= box["north"]) & lon_mask
    )
    if not weighted:
        return sub.mean(["lat", "lon"], skipna=True)
    w = np.cos(np.deg2rad(sub.lat)).clip(min=0.0)
    return sub.weighted(w).mean(["lat", "lon"], skipna=True)


# --- transform modes -------------------------------------------------------


def test_default_transform_is_still_standardize(global_sst):
    assert Index.named("wvg").transform == "standardize"
    assert Index.named("wvg").weights is None


def test_anomaly_transform_keeps_physical_units(global_sst):
    """An anomaly index is centred on zero but carries the field's spread, so it
    is not unit-variance the way a standardized index is."""
    anom = Index.custom(
        name="n34_anom", regions={"nino34": "nino34"},
        combine=lambda a: a["nino34"], transform="anomaly",
    ).reduce(global_sst)
    expected = _box_mean(global_sst, REGIONS["nino34"])
    np.testing.assert_allclose(anom.values, (expected - expected.mean()).values, rtol=1e-10)
    assert abs(float(anom.mean())) < 1e-10
    assert not np.isclose(float(anom.std(ddof=0)), 1.0)


def test_raw_transform_passes_the_box_mean_through_untouched(global_sst):
    raw = Index.custom(
        name="n34_raw", regions={"nino34": "nino34"},
        combine=lambda r: r["nino34"], transform="raw",
    ).reduce(global_sst)
    np.testing.assert_allclose(raw.values, _box_mean(global_sst, REGIONS["nino34"]).values)
    # Absolute SST, so it sits near the field's 27 °C, not near zero.
    assert 26.0 < float(raw.mean()) < 28.0


def test_raw_index_needs_no_climatology_and_no_time_axis(global_sst):
    """An absolute threshold on a single forecast map has no reference period to
    standardize against; a raw index must still work."""
    single_map = global_sst.isel(year=0, drop=True)
    value = Index.named("wio").reduce(single_map)
    assert value.dims == ()
    assert np.isfinite(float(value))


def test_transform_can_be_set_per_region(global_sst):
    idx = Index.custom(
        name="mixed",
        regions={"nino34": "nino34", "nino4": "nino4"},
        combine=lambda t: t["nino34"] - t["nino4"],
        transform={"nino34": "anomaly", "nino4": "raw"},
    )
    got = idx.reduce(global_sst)
    n34 = _box_mean(global_sst, REGIONS["nino34"])
    n4 = _box_mean(global_sst, REGIONS["nino4"])
    expected = (n34 - n34.mean()) - n4
    np.testing.assert_allclose(got.values, expected.values, rtol=1e-10)


def test_unknown_transform_is_rejected():
    with pytest.raises(ValueError, match="transform must be one of"):
        Index.custom(name="x", regions={"nino34": "nino34"},
                     combine=lambda z: z["nino34"], transform="zscore")


def test_per_region_transform_naming_an_absent_region_is_rejected():
    with pytest.raises(ValueError, match="does not use"):
        Index.custom(name="x", regions={"nino34": "nino34"},
                     combine=lambda z: z["nino34"], transform={"nino4": "raw"})


# --- weighting -------------------------------------------------------------


def test_cos_lat_weighting_changes_a_tall_box_and_matches_a_manual_computation(global_sst):
    """The RONI tropical band spans 40° of latitude; unweighted and weighted
    means genuinely differ there, which is why RONI declares cos-lat."""
    unweighted = Index.custom(
        name="t_u", regions={"tropics": "tropics"},
        combine=lambda r: r["tropics"], transform="raw",
    ).reduce(global_sst)
    weighted = Index.custom(
        name="t_w", regions={"tropics": "tropics"},
        combine=lambda r: r["tropics"], transform="raw", weights="cos_lat",
    ).reduce(global_sst)

    np.testing.assert_allclose(
        weighted.values, _box_mean(global_sst, REGIONS["tropics"], weighted=True).values
    )
    assert not np.allclose(unweighted.values, weighted.values)


def test_weighting_is_a_no_op_on_a_latitudinally_symmetric_uniform_field():
    """A constant field has the same mean under any weighting — a sanity bound
    on the weighting not introducing bias."""
    lat = np.arange(-20, 21, 2.0)
    lon = np.arange(0, 360, 2.0)
    const = _field([2000], lat, lon, np.full((1, len(lat), len(lon)), 27.0))
    for weights in (None, "cos_lat"):
        idx = Index.custom(name="c", regions={"tropics": "tropics"},
                           combine=lambda r: r["tropics"], transform="raw",
                           weights=weights).reduce(const)
        assert float(idx.values[0]) == pytest.approx(27.0)


def test_full_longitude_sweep_selects_the_whole_band_not_one_meridian():
    """`tropics` spans west=0 east=360. Both bounds are 0 after the mod, and a
    naive `lon >= 0 & lon <= 0` mask would keep a single column of cells."""
    lat = np.arange(-30, 31, 2.0)
    lon = np.arange(0, 360, 2.0)
    values = np.tile(lon, (1, len(lat), 1)).astype(float)  # varies only with lon
    field = _field([2000], lat, lon, values)
    idx = Index.custom(name="t", regions={"tropics": "tropics"},
                       combine=lambda r: r["tropics"], transform="raw").reduce(field)
    # Mean of 0, 2, ..., 358 is 179. A single meridian at lon=0 would give 0.
    assert float(idx.values[0]) == pytest.approx(179.0)


def test_unknown_weights_are_rejected(global_sst):
    idx = Index.custom(name="x", regions={"nino34": "nino34"},
                       combine=lambda z: z["nino34"], weights="area")
    with pytest.raises(ValueError, match="weights must be None"):
        idx.reduce(global_sst)


# --- baselines -------------------------------------------------------------


def test_baseline_restricts_the_reference_period(global_sst):
    """Standardizing against 1991-2020 must equal standardizing against a
    hand-sliced climatology."""
    baselined = Index.named("nino34", baseline=(1991, 2020)).reduce(global_sst)
    manual = Index.named("nino34").reduce(
        global_sst, climatology=global_sst.sel(year=slice(1991, 2020))
    )
    np.testing.assert_allclose(baselined.values, manual.values, rtol=1e-12)


def test_baseline_differs_from_the_full_record(global_sst):
    full = Index.named("nino34").reduce(global_sst)
    baselined = Index.named("nino34", baseline=(1991, 2020)).reduce(global_sst)
    assert not np.allclose(full.values, baselined.values)


def test_reduce_baseline_argument_overrides_the_index_baseline(global_sst):
    idx = Index.named("nino34", baseline=(1991, 2020))
    overridden = idx.reduce(global_sst, baseline=(1981, 2010))
    expected = Index.named("nino34", baseline=(1981, 2010)).reduce(global_sst)
    np.testing.assert_allclose(overridden.values, expected.values, rtol=1e-12)


def test_baseline_works_on_a_time_dim_not_just_a_year_dim(global_sst):
    stamped = global_sst.rename(year="time").assign_coords(
        time=np.array([np.datetime64(f"{y}-01-01") for y in range(1981, 2027)])
    )
    got = Index.named("nino34", baseline=(1991, 2020)).reduce(stamped)
    manual = Index.named("nino34").reduce(
        stamped, climatology=stamped.sel(time=slice("1991", "2020"))
    )
    assert got.sizes["time"] == stamped.sizes["time"]
    np.testing.assert_allclose(got.values, manual.values, rtol=1e-12)


def test_baseline_selecting_nothing_raises(global_sst):
    with pytest.raises(ValueError, match="selects no points"):
        Index.named("nino34", baseline=(1900, 1910)).reduce(global_sst)


def test_baseline_without_a_time_like_dim_raises(global_sst):
    single_map = global_sst.isel(year=0, drop=True)
    with pytest.raises(ValueError, match="no 'year', 'time' or 'init_time' dim"):
        Index.named("nino34", baseline=(1991, 2020)).reduce(single_map)


# --- named ocean indices ---------------------------------------------------


def test_roni_is_nino34_anomaly_minus_the_weighted_tropical_mean_anomaly(global_sst):
    got = Index.named("roni").reduce(global_sst)
    n34 = _box_mean(global_sst, REGIONS["nino34"], weighted=True)
    trop = _box_mean(global_sst, REGIONS["tropics"], weighted=True)
    expected = (n34 - n34.mean()) - (trop - trop.mean())
    np.testing.assert_allclose(got.values, expected.values, rtol=1e-10)


def test_roni_removes_a_uniform_basin_wide_warming_trend():
    """The point of RONI: a warming applied to the whole tropics must not
    register as a Niño signal, whereas a raw Niño3.4 anomaly would."""
    years = np.arange(1981, 2027)
    lat = np.arange(-30, 31, 2.0)
    lon = np.arange(0, 360, 2.0)
    trend = np.linspace(0, 2.0, len(years))[:, None, None]
    field = _field(years, lat, lon, 27.0 + np.broadcast_to(trend, (len(years), len(lat), len(lon))).copy())

    roni = Index.named("roni").reduce(field)
    oni = Index.named("oni").reduce(field)

    np.testing.assert_allclose(roni.values, 0.0, atol=1e-9)
    assert float(oni.max()) > 0.9  # the raw anomaly tracks the trend


def test_dmi_is_the_difference_of_the_two_iod_pole_anomalies(global_sst):
    got = Index.named("dmi").reduce(global_sst)
    wtio = _box_mean(global_sst, REGIONS["wtio"], weighted=True)
    setio = _box_mean(global_sst, REGIONS["setio"], weighted=True)
    expected = (wtio - wtio.mean()) - (setio - setio.mean())
    np.testing.assert_allclose(got.values, expected.values, rtol=1e-10)


def test_iod_is_an_alias_for_dmi(global_sst):
    np.testing.assert_allclose(
        Index.named("iod").reduce(global_sst).values,
        Index.named("dmi").reduce(global_sst).values,
    )


def test_a_warm_west_pole_makes_dmi_positive(global_sst):
    warmed = global_sst + xr.where(
        (global_sst.lat >= -10) & (global_sst.lat <= 10)
        & (global_sst.lon >= 50) & (global_sst.lon <= 70),
        3.0, 0.0,
    )
    # Reference the anomalies to the unwarmed climatology so the warming shows.
    dmi = Index.named("dmi").reduce(warmed, climatology=global_sst)
    assert float(dmi.mean()) > 2.0


def test_wio_reports_absolute_temperature_for_threshold_comparisons(global_sst):
    """The >29 °C criterion needs degrees, not a z-score."""
    wio = Index.named("wio").reduce(global_sst)
    expected = _box_mean(global_sst, REGIONS["wtio"], weighted=True)
    np.testing.assert_allclose(wio.values, expected.values, rtol=1e-10)
    assert 26.0 < float(wio.mean()) < 28.0


def test_wpac_is_a_standardized_box_mean_and_works_on_precipitation():
    """The module is not SST-specific: the same reduction over a precip field is
    the Walker-circulation indicator."""
    years = np.arange(1981, 2026)
    lat = np.arange(-20, 21, 1.0)
    lon = np.arange(90, 160, 1.0)
    rng = np.random.default_rng(3)
    precip = _field(years, lat, lon, rng.gamma(2.0, 2.0, (len(years), len(lat), len(lon))))
    idx = Index.named("wpac").reduce(precip)
    assert idx.dims == ("year",)
    np.testing.assert_allclose(float(idx.std(ddof=0)), 1.0, rtol=1e-6)


def test_list_named_advertises_every_index():
    listed = Index.list_named()
    for name in ("wvg", "wvg2", "nino34", "roni", "dmi", "wio", "wpac", "oni"):
        assert name in listed and listed[name]


def test_named_accepts_transform_and_weight_overrides(global_sst):
    """A caller who wants a standardized RONI (some definitions divide by its
    own std) should not have to rebuild the box definitions."""
    standardized = Index.named("roni", transform="standardize").reduce(global_sst)
    assert abs(float(standardized.mean())) < 1e-9


# --- deprecated alias ------------------------------------------------------


def test_reduce_accepts_the_deprecated_sst_keyword(global_sst):
    with pytest.warns(DeprecationWarning, match="now `field`"):
        got = Index.named("nino34").reduce(sst=global_sst)
    np.testing.assert_allclose(got.values, Index.named("nino34").reduce(global_sst).values)


def test_reduce_rejects_both_field_and_sst(global_sst):
    with pytest.raises(TypeError, match="not both"):
        Index.named("nino34").reduce(global_sst, sst=global_sst)


def test_reduce_requires_a_field():
    with pytest.raises(TypeError, match="requires a `field`"):
        Index.named("nino34").reduce()
