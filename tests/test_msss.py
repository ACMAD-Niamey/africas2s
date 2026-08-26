import numpy as np
import pytest
import xarray as xr

from africas2s.registry import get_metric


def _da(vals):
    ny, nla, nlo = vals.shape
    return xr.DataArray(vals, dims=("year", "lat", "lon"),
                        coords={"year": np.arange(ny), "lat": np.arange(nla),
                                "lon": np.arange(nlo)})


def test_msss_registered():
    assert get_metric("msss") is get_metric("mean_square_skill_score")


def test_msss_perfect_forecast_is_one():
    rng = np.random.default_rng(0)
    o = _da(rng.standard_normal((20, 3, 4)))
    m = get_metric("msss")()
    assert m.compute(o, o) == pytest.approx(1.0)


def test_msss_climatology_forecast_is_zero():
    rng = np.random.default_rng(1)
    o = _da(rng.standard_normal((20, 3, 4)))
    clim = o.mean("year")                       # broadcast constant-in-year forecast
    f = clim.broadcast_like(o)
    m = get_metric("msss")()
    assert m.compute(f, o) == pytest.approx(0.0, abs=1e-12)


def test_msss_matches_closed_form():
    rng = np.random.default_rng(2)
    o = _da(rng.standard_normal((30, 2, 2)))
    f = o + 0.5 * rng.standard_normal((30, 2, 2))
    m = get_metric("msss")()
    mse = float((((f - o) ** 2).mean("year")).mean())
    varo = float(((o - o.mean("year")) ** 2).mean("year").mean())
    assert m.compute(f, o) == pytest.approx(1.0 - mse / varo, abs=1e-9)


def test_msss_spatial_returns_field():
    rng = np.random.default_rng(3)
    o = _da(rng.standard_normal((20, 3, 4)))
    f = o + 0.1 * rng.standard_normal((20, 3, 4))
    out = get_metric("msss")().compute(f, o, spatial=True)
    assert out.dims == ("lat", "lon") and out.shape == (3, 4)


def test_msss_rejects_tercile():
    o = _da(np.ones((5, 2, 2)))
    f = o.expand_dims(tercile=[0, 1, 2])
    with pytest.raises(ValueError, match="tercile"):
        get_metric("msss")().compute(f, o)
