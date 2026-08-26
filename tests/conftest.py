import numpy as np
import xarray as xr
import pytest

np.random.seed(42)


def pytest_collection_modifyitems(config, items):
    """Keep bare pytest runs on local/unit tests unless marker-gated tests are requested."""
    if config.option.markexpr:
        return
    gated = {"agreement", "gpu", "integration"}
    skip = pytest.mark.skip(reason="requires explicit marker selection")
    for item in items:
        if gated.intersection(item.keywords):
            item.add_marker(skip)


# Small spatial grids
_COARSE_LAT = np.linspace(-4, 4, 5)
_COARSE_LON = np.linspace(30, 38, 5)
_FINE_LAT = np.linspace(-4, 4, 20)
_FINE_LON = np.linspace(30, 38, 20)
_YEARS = np.arange(2000, 2010)
_MEMBERS = np.arange(3)


def _planted_signal(years, lat, lon):
    """Create data with a spatial pattern that varies with year."""
    signal = np.sin(np.arange(len(years)) * 0.5)[:, None, None]
    spatial = np.outer(np.sin(lat * 0.5), np.cos(lon * 0.3))[None, :, :]
    return signal * spatial


@pytest.fixture
def synthetic_gcm_hindcast():
    signal = _planted_signal(_YEARS, _COARSE_LAT, _COARSE_LON)
    noise = np.random.randn(len(_YEARS), len(_MEMBERS), len(_COARSE_LAT), len(_COARSE_LON)) * 0.3
    data = signal[:, None, :, :] + noise + 5.0  # add offset to keep positive
    return xr.DataArray(
        data,
        dims=["year", "member", "lat", "lon"],
        coords={"year": _YEARS, "member": _MEMBERS, "lat": _COARSE_LAT, "lon": _COARSE_LON},
    )


@pytest.fixture
def synthetic_gcm_forecast():
    data = np.random.randn(len(_MEMBERS), len(_COARSE_LAT), len(_COARSE_LON)) * 0.5 + 5.0
    return xr.DataArray(
        data,
        dims=["member", "lat", "lon"],
        coords={"member": _MEMBERS, "lat": _COARSE_LAT, "lon": _COARSE_LON},
    )


@pytest.fixture
def synthetic_obs():
    signal = _planted_signal(_YEARS, _FINE_LAT, _FINE_LON)
    noise = np.random.randn(len(_YEARS), len(_FINE_LAT), len(_FINE_LON)) * 0.2
    data = signal + noise + 5.0
    return xr.DataArray(
        data,
        dims=["year", "lat", "lon"],
        coords={"year": _YEARS, "lat": _FINE_LAT, "lon": _FINE_LON},
    )


@pytest.fixture
def perfect_tercile_forecast(synthetic_obs):
    """Tercile forecast that perfectly matches obs categories (CPT-compatible)."""
    from africas2s.metrics.rpss import _cpt_boundaries
    t33, t67 = _cpt_boundaries(synthetic_obs.values)
    # CPT categorization: strict < for lower boundary
    cat = xr.where(
        xr.DataArray(t33, dims=["lat", "lon"], coords={k: synthetic_obs.coords[k] for k in ["lat", "lon"]}) > synthetic_obs,
        0,
        xr.where(
            xr.DataArray(t67, dims=["lat", "lon"], coords={k: synthetic_obs.coords[k] for k in ["lat", "lon"]}) > synthetic_obs,
            1, 2),
    )
    onehot = xr.concat([(cat == i).astype(float) for i in range(3)], dim="tercile")
    onehot["tercile"] = [0, 1, 2]
    return onehot  # (year, tercile, lat, lon)


@pytest.fixture
def climatology_forecast(synthetic_obs):
    """Uniform 1/3 tercile probabilities everywhere."""
    shape = (len(_YEARS), 3, len(_FINE_LAT), len(_FINE_LON))
    data = np.full(shape, 1.0 / 3.0)
    return xr.DataArray(
        data,
        dims=["year", "tercile", "lat", "lon"],
        coords={"year": _YEARS, "tercile": [0, 1, 2], "lat": _FINE_LAT, "lon": _FINE_LON},
    )


@pytest.fixture
def synthetic_gcm_hindcast2():
    """Second GCM hindcast (different noise) for multi-model tests."""
    signal = _planted_signal(_YEARS, _COARSE_LAT, _COARSE_LON)
    noise = np.random.randn(len(_YEARS), len(_MEMBERS), len(_COARSE_LAT), len(_COARSE_LON)) * 0.4
    data = signal[:, None, :, :] + noise + 5.0
    return xr.DataArray(
        data,
        dims=["year", "member", "lat", "lon"],
        coords={"year": _YEARS, "member": _MEMBERS, "lat": _COARSE_LAT, "lon": _COARSE_LON},
    )
