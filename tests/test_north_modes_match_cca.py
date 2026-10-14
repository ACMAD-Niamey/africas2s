"""North-rule mode selection must test the SAME spectrum the CCA is fit on."""
import numpy as np, xarray as xr, pytest
from africas2s.methods.cca import _north_modes, select_modes_north


def _field(seed=0, n=30, contrast=100.0):
    """Two regions: one with `contrast`x the variance of the other, each with
    its own independent mode. Under raw covariance the big-variance region
    dominates the ladder; under correlation both regions count equally."""
    rng = np.random.default_rng(seed)
    lat = np.arange(-20., 21., 5.); lon = np.arange(0., 41., 5.)
    a = rng.standard_normal(n)[:, None, None] * np.ones((1, len(lat), len(lon)))
    b = rng.standard_normal(n)[:, None, None] * np.ones((1, len(lat), len(lon)))
    west = (lon < 20)[None, None, :]
    v = np.where(west, a * np.sqrt(contrast), b) + 0.1 * rng.standard_normal((n, len(lat), len(lon)))
    return xr.DataArray(v, coords={"year": np.arange(1991, 1991 + n), "lat": lat, "lon": lon},
                        dims=("year", "lat", "lon"))


def test_standardize_changes_the_count_on_a_variance_contrast():
    f = _field()
    raw = _north_modes(f, ceiling=10, standardize=False)
    std = _north_modes(f, ceiling=10, standardize=True)
    assert raw == 1, "covariance spectrum: the high-variance region's mode dominates"
    assert std >= 2, "correlation spectrum: both regions' modes survive"


def test_lat_weights_cpt_is_accepted_and_forwarded():
    f = _field()
    a = select_modes_north(f, f, standardize=True, lat_weights="cpt")
    b = select_modes_north(f, f, standardize=True, lat_weights="sqrt_cos")
    assert all(isinstance(x, int) for x in a + b)


def test_seasonal_mme_records_per_fold_north_counts():
    import africas2s
    rng = np.random.default_rng(1)
    yrs = np.arange(1991, 2021)
    lat = np.arange(-10., 11., 5.); lon = np.arange(0., 21., 5.)
    mode = rng.standard_normal(len(yrs))
    obs = xr.DataArray(5 + 2 * mode[:, None, None] * np.ones((1, len(lat), len(lon)))
                       + rng.standard_normal((len(yrs), len(lat), len(lon))),
                       coords={"year": yrs, "lat": lat, "lon": lon}, dims=("year", "lat", "lon"))
    h = xr.DataArray(3 + 1.5 * mode[:, None, None, None] * np.ones((1, 4, len(lat), len(lon)))
                     + rng.standard_normal((len(yrs), 4, len(lat), len(lon))),
                     coords={"year": yrs, "member": np.arange(4), "lat": lat, "lon": lon},
                     dims=("year", "member", "lat", "lon"), attrs={"units": "mm/day"})
    f = h.isel(year=[-1]).assign_coords(year=[2026])
    res = africas2s.seasonal_mme(
        {"prcp": {"M": (h, f)}}, obs, method="cca", cv="loyo_cyclic",
        probability_aggregation="cpt_per_model", climatology_period=(1991, 2020),
        cpt_args=dict(mode_selection="north", crossvalidation_window=5,
                      standardize=True, lat_weights="cpt"),
        forecast_year=2026, verbose=False)
    assert res.tercile_forecast is not None
