"""Series-level quantile mapping and forecast-error confidence bounds."""
import numpy as np
import pytest
import xarray as xr
from scipy.stats import norm

from africas2s.methods._qm_kernel import empirical_map, plotting_positions
from africas2s.series import ErrorBounds, error_bounds, quantile_map


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def _years(values, start=1991):
    years = np.arange(start, start + len(values))
    return xr.DataArray(np.asarray(values, dtype=float), dims="year",
                        coords={"year": years})


# --- the shared kernel -----------------------------------------------------


def test_kernel_maps_a_reference_point_onto_its_counterpart():
    source = np.array([0.0, 1.0, 2.0, 3.0])
    target = np.array([10.0, 20.0, 30.0, 40.0])
    # The k-th sorted source value sits at the k-th plotting position, so it
    # must come back as the k-th sorted target value.
    for k in range(4):
        assert empirical_map(source[k], source, target) == pytest.approx(target[k])


def test_kernel_handles_unequal_sample_sizes():
    """A 5-year model record mapped onto a 9-year observed record."""
    source = np.arange(5.0)
    target = np.arange(9.0) * 2.0
    mapped = empirical_map(source, source, target)
    assert mapped.shape == source.shape
    assert np.all(np.diff(mapped) > 0)  # monotone
    assert target.min() <= mapped.min() and mapped.max() <= target.max()


def test_kernel_equal_sizes_reproduces_the_paired_sort_form():
    source, target = np.sort(np.arange(10.0)), np.sort(np.arange(10.0) ** 2)
    pp = plotting_positions(10)
    x = np.array([2.3, 7.7])
    expected = np.interp(np.interp(x, source, pp), pp, target)
    np.testing.assert_allclose(empirical_map(x, source, target), expected)


def test_kernel_clamps_out_of_support_values_by_default():
    source = np.array([0.0, 1.0, 2.0])
    target = np.array([5.0, 6.0, 7.0])
    assert empirical_map(-99.0, source, target) == pytest.approx(5.0)
    assert empirical_map(99.0, source, target) == pytest.approx(7.0)


def test_kernel_linear_extrapolation_continues_the_end_slope():
    """A record-breaking forecast must not be silently truncated to the
    strongest event in the training record."""
    source = np.array([0.0, 1.0, 2.0])
    target = np.array([0.0, 10.0, 20.0])  # slope 10
    assert empirical_map(3.0, source, target, extrapolate="linear") == pytest.approx(30.0)
    assert empirical_map(-1.0, source, target, extrapolate="linear") == pytest.approx(-10.0)


def test_kernel_linear_extrapolation_falls_back_to_clamping_on_tied_endpoints():
    source = np.array([1.0, 1.0, 2.0])  # zero-width lower interval
    target = np.array([5.0, 6.0, 7.0])
    got = empirical_map(-99.0, source, target, extrapolate="linear")
    assert got == pytest.approx(5.0)


def test_kernel_propagates_non_finite_inputs():
    source, target = np.arange(3.0), np.arange(3.0)
    assert np.isnan(empirical_map(np.nan, source, target))
    assert np.all(np.isnan(empirical_map(np.array([1.0]), np.array([0.0, np.nan]), target)))


def test_kernel_rejects_an_unknown_extrapolate():
    with pytest.raises(ValueError, match="extrapolate must be"):
        empirical_map(1.0, np.arange(3.0), np.arange(3.0), extrapolate="spline")


# --- quantile_map ----------------------------------------------------------


def test_quantile_map_moves_a_biased_forecast_onto_the_observed_distribution(rng):
    observed = rng.normal(0.0, 1.0, 500)
    modelled = observed * 2.0 + 5.0  # inflated spread, large warm bias
    corrected = quantile_map(modelled, modelled, observed)
    assert abs(corrected.mean() - observed.mean()) < 0.05
    assert abs(corrected.std() - observed.std()) < 0.05


def test_quantile_map_is_monotone(rng):
    source, target = rng.normal(size=60), rng.gamma(2.0, 1.5, size=60)
    x = np.linspace(source.min(), source.max(), 25)
    mapped = quantile_map(x, source, target)
    assert np.all(np.diff(mapped) >= -1e-12)


def test_quantile_map_preserves_dataarray_structure():
    source, target = np.arange(10.0), np.arange(10.0) + 100.0
    x = _years(np.arange(3.0))
    out = quantile_map(x, source, target)
    assert isinstance(out, xr.DataArray)
    assert out.dims == ("year",)
    np.testing.assert_array_equal(out.year.values, x.year.values)


def test_quantile_map_accepts_dataarray_references():
    out = quantile_map(2.0, _years(np.arange(10.0)), _years(np.arange(10.0) + 100.0))
    assert out == pytest.approx(102.0)


def test_quantile_map_drops_non_finite_reference_values():
    source = np.array([0.0, 1.0, np.nan, 2.0])
    target = np.array([10.0, 11.0, 12.0])
    assert np.isfinite(quantile_map(1.0, source, target))


def test_quantile_map_parametric_matches_the_gaussian_rescaling():
    source = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    target = source * 3.0 + 7.0
    got = quantile_map(2.0, source, target, variant="parametric")
    expected = target.mean() + (2.0 - source.mean()) / source.std(ddof=1) * target.std(ddof=1)
    assert got == pytest.approx(expected)


def test_quantile_map_parametric_extrapolates_beyond_the_training_range():
    source = np.arange(5.0)
    target = np.arange(5.0)
    assert quantile_map(100.0, source, target, variant="parametric") > 90.0
    assert quantile_map(100.0, source, target, variant="empirical") == pytest.approx(4.0)


def test_quantile_map_rejects_an_unknown_variant():
    with pytest.raises(ValueError, match="variant must be"):
        quantile_map(1.0, np.arange(3.0), np.arange(3.0), variant="gamma")


def test_quantile_map_rejects_an_all_nan_reference():
    with pytest.raises(ValueError, match="no finite values"):
        quantile_map(1.0, np.array([np.nan, np.nan]), np.arange(3.0))


# --- error_bounds ----------------------------------------------------------


def test_error_bounds_brackets_the_forecast_for_an_unbiased_model(rng):
    obs = _years(rng.normal(0.0, 1.0, 400))
    pred = obs + _years(rng.normal(0.0, 0.5, 400))
    bounds = error_bounds(pred, obs, 3.0, level=0.8)
    assert bounds.lower < 3.0 < bounds.upper
    assert float(bounds.bias) == pytest.approx(0.0, abs=0.06)


def test_error_bounds_removes_a_systematic_model_bias():
    """A model that always runs 2 units warm should have its forecast shifted
    down by 2, not merely bracketed."""
    obs = _years(np.arange(30.0))
    pred = obs + 2.0
    bounds = error_bounds(pred, obs, 10.0, level=0.8)
    assert float(bounds.bias) == pytest.approx(2.0)
    midpoint = (bounds.lower + bounds.upper) / 2.0
    assert float(midpoint) == pytest.approx(8.0)


def test_error_bounds_interval_is_ordered_and_widens_with_level(rng):
    obs = _years(rng.normal(0.0, 1.0, 200))
    pred = obs + _years(rng.normal(0.0, 1.0, 200))
    narrow = error_bounds(pred, obs, 0.0, level=0.5)
    wide = error_bounds(pred, obs, 0.0, level=0.95)
    assert narrow.lower < narrow.upper
    assert wide.lower < narrow.lower and narrow.upper < wide.upper


def test_error_bounds_gaussian_matches_the_normal_quantiles():
    obs = _years(np.zeros(50))
    errors = np.linspace(-1, 1, 50)
    pred = _years(errors)
    bounds = error_bounds(pred, obs, 0.0, level=0.8, method="gaussian")
    spread = errors.std(ddof=1)
    assert float(bounds.upper) == pytest.approx(norm.ppf(0.9) * spread, rel=1e-6)


def test_error_bounds_empirical_and_gaussian_agree_on_a_normal_error_sample(rng):
    obs = _years(np.zeros(4000))
    pred = _years(rng.normal(0.0, 1.0, 4000))
    emp = error_bounds(pred, obs, 0.0, level=0.8)
    gau = error_bounds(pred, obs, 0.0, level=0.8, method="gaussian")
    assert float(emp.upper) == pytest.approx(float(gau.upper), abs=0.08)


def test_error_bounds_unpacks_as_a_pair():
    obs = _years(np.arange(20.0))
    lower, upper = error_bounds(obs + _years(np.linspace(-1, 1, 20)), obs, 5.0)
    assert lower < upper


def test_a_model_with_no_error_spread_gets_a_zero_width_interval():
    """Degenerate but correct: if every hindcast year missed by exactly 1, the
    only defensible interval around the bias-corrected forecast is a point."""
    obs = _years(np.arange(20.0))
    lower, upper = error_bounds(obs + 1.0, obs, 5.0)
    assert float(lower) == pytest.approx(4.0) == float(upper)


def test_error_bounds_is_an_errorbounds_instance_carrying_the_error_sample():
    obs = _years(np.arange(20.0))
    got = error_bounds(obs + 1.0, obs, 5.0)
    assert isinstance(got, ErrorBounds)
    assert got.errors.sizes["year"] == 20
    np.testing.assert_allclose(got.errors.values, 1.0)


def test_error_bounds_broadcasts_over_surviving_dims():
    """A per-cell (or per-model) interval falls out of the same call."""
    years = np.arange(1991, 2011)
    obs = xr.DataArray(np.zeros((20, 2)), dims=("year", "lat"),
                       coords={"year": years, "lat": [0.0, 1.0]})
    err = np.stack([np.linspace(-1, 1, 20), np.linspace(-4, 4, 20)], axis=1)
    pred = obs + xr.DataArray(err, dims=("year", "lat"), coords=obs.coords)
    forecast = xr.DataArray([0.0, 0.0], dims="lat", coords={"lat": [0.0, 1.0]})
    bounds = error_bounds(pred, obs, forecast, level=0.8)
    # The noisier cell must get the wider interval.
    width = bounds.upper - bounds.lower
    assert float(width.sel(lat=1.0)) > float(width.sel(lat=0.0))


def test_error_bounds_rejects_a_forecast_carrying_the_sample_dim():
    obs = _years(np.arange(20.0))
    with pytest.raises(ValueError, match="must not carry the sample dim"):
        error_bounds(obs + 1.0, obs, obs)


def test_error_bounds_rejects_a_bad_level_or_method():
    obs = _years(np.arange(20.0))
    with pytest.raises(ValueError, match="level must lie strictly"):
        error_bounds(obs + 1.0, obs, 5.0, level=1.0)
    with pytest.raises(ValueError, match="method must be"):
        error_bounds(obs + 1.0, obs, 5.0, method="bootstrap")


def test_error_bounds_rejects_hindcasts_without_the_sample_dim():
    obs = _years(np.arange(20.0))
    with pytest.raises(ValueError, match="must carry the sample dim"):
        error_bounds(obs.rename(year="time"), obs, 5.0)
