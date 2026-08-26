"""Unit tests for the CorrDiff downscaling method.

Tests are organised into three tiers:

1. **No-dep tests** -- run without torch or earth2studio.
2. **Mock tests** -- require torch but mock earth2studio (no GPU needed).
3. **GPU tests** -- in test_corrdiff_integration.py (require real GPU).
"""

import sys
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_OBS_LAT = np.linspace(10, 20, 40)
_OBS_LON = np.linspace(100, 110, 40)
_YEARS = np.arange(2000, 2010)
_MEMBERS = np.arange(3)

# Fake model grid (subset of a real 0.25-deg global grid)
_MODEL_IN_LAT = np.linspace(90, -90, 64)
_MODEL_IN_LON = np.linspace(0, 357.1875, 128)
_MODEL_OUT_LAT = np.linspace(90, -90, 721)
_MODEL_OUT_LON = np.linspace(0, 359.75, 1440)


def _make_obs():
    data = np.random.randn(len(_YEARS), len(_OBS_LAT), len(_OBS_LON)) + 290.0
    return xr.DataArray(
        data, dims=["year", "lat", "lon"],
        coords={"year": _YEARS, "lat": _OBS_LAT, "lon": _OBS_LON},
    )


def _make_hindcast():
    data = np.random.randn(len(_YEARS), len(_MEMBERS), 5, 5)
    return xr.DataArray(
        data, dims=["year", "member", "lat", "lon"],
        coords={"year": _YEARS, "member": _MEMBERS,
                "lat": np.linspace(10, 20, 5), "lon": np.linspace(100, 110, 5)},
    )


def _make_forecast():
    data = np.random.randn(len(_MEMBERS), 5, 5) + 290.0
    return xr.DataArray(
        data, dims=["member", "lat", "lon"],
        coords={"member": _MEMBERS,
                "lat": np.linspace(10, 20, 5), "lon": np.linspace(100, 110, 5)},
    )


# ---------------------------------------------------------------------------
# Fake earth2studio model
# ---------------------------------------------------------------------------

_FAKE_INPUT_VARS = [
    "va10", "vas", "prc", "ua10", "ta850", "rls", "tasmin", "wap850",
    "hursmax", "ua850", "ua50", "va850", "hus10", "rlut", "va1000", "pr",
    "zg1000", "sfcWindmax", "hurs", "ta50", "rsus", "sfcWind", "wap10",
    "ta500", "ua100", "hus1000", "zg500", "hus250", "ua500", "ua1000",
    "hursmin", "ta700", "va250", "hus700", "hus100", "ua700", "wap100",
    "zg100", "ta10", "va500", "tas", "ua250", "wap1000", "zg700", "va100",
    "rlds", "tasmax", "va700", "clt", "rsds", "zg10", "ta100", "wap500",
    "ta250", "rss", "hfls", "rlus", "va50", "wap250", "ta1000", "hfss",
    "zg850", "uas", "wap700", "snc", "zg50", "wap50", "zg250", "psl",
    "hus50", "hus850", "hus500", "siconc", "ts",
]

_FAKE_OUTPUT_VARS = [
    "u10m", "v10m", "u100m", "v100m", "t2m", "sp", "msl", "tcwv",
] + [f"{v}{p}" for v in ["u", "v", "z", "t", "q"]
     for p in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
] + ["sst", "d2m"]


class _FakeModel:
    """Mimics a loaded CorrDiffCMIP6 model for testing."""

    def __init__(self):
        import torch as _torch
        self.input_variables = _FAKE_INPUT_VARS
        self.output_variables = _FAKE_OUTPUT_VARS
        self.lat_input_grid = _torch.tensor(_MODEL_IN_LAT)
        self.lon_input_grid = _torch.tensor(_MODEL_IN_LON)
        self.lat_output_grid = _torch.tensor(_MODEL_OUT_LAT)
        self.lon_output_grid = _torch.tensor(_MODEL_OUT_LON)
        self.number_of_samples = 1

    def to(self, device):
        return self

    def __call__(self, x, coords):
        import torch
        n_samples = self.number_of_samples
        n_out_vars = len(self.output_variables)
        n_lat = len(self.lat_output_grid)
        n_lon = len(self.lon_output_grid)
        # output: (batch, sample, time, lead_time, variable, lat, lon)
        batch = x.shape[0]
        out = torch.randn(batch, n_samples, 1, 1, n_out_vars, n_lat, n_lon)
        out_coords = OrderedDict({
            "batch": coords["batch"],
            "sample": np.arange(n_samples),
            "time": coords["time"],
            "lead_time": np.array([np.timedelta64(0, "h")]),
            "variable": np.array(self.output_variables),
            "lat": self.lat_output_grid.numpy(),
            "lon": self.lon_output_grid.numpy(),
        })
        return out, out_coords


@pytest.fixture
def _mock_earth2studio():
    """Patch earth2studio so CorrDiffMethod can load without real deps."""
    # _FakeModel builds torch tensors, so torch must be importable. Skip
    # cleanly (rather than error at fixture setup) when it isn't installed.
    pytest.importorskip("torch")
    fake_model = _FakeModel()

    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = fake_model

    modules = {
        "earth2studio": MagicMock(),
        "earth2studio.models": MagicMock(),
        "earth2studio.models.dx": MagicMock(CorrDiffCMIP6=fake_cls),
    }
    with patch.dict(sys.modules, modules):
        yield fake_model


# ---------------------------------------------------------------------------
# Tier 1: No-dep tests
# ---------------------------------------------------------------------------

def test_corrdiff_registered():
    """corrdiff should always be in the registry (even without torch)."""
    from africas2s.registry import get_method
    cls = get_method("corrdiff")
    assert cls.__name__ == "CorrDiffMethod"


def test_is_pretrained_flag():
    from africas2s.methods.corrdiff import CorrDiffMethod
    assert CorrDiffMethod.is_pretrained is True


def test_corrdiff_save_load_raise_not_implemented():
    """save/load are deferred for the pretrained model (#27/#28).

    Uses object.__new__ to bypass __init__ (which requires torch); the
    override raises before touching any model state, so this needs no deps
    and runs even on machines without torch.
    """
    from africas2s.methods.corrdiff import CorrDiffMethod
    m = object.__new__(CorrDiffMethod)
    with pytest.raises(NotImplementedError, match="#27/#28"):
        m.save("ignored.pkl")
    with pytest.raises(NotImplementedError, match="#27/#28"):
        m.load("ignored.pkl")


def test_import_error_without_torch(monkeypatch):
    """Instantiation should give a clear error when torch is missing."""
    monkeypatch.setattr(
        "africas2s.methods.corrdiff.require_optional",
        lambda name, **kw: (_ for _ in ()).throw(
            ImportError(f"{name} is required. pip install torch")
        ),
    )
    from africas2s.methods.corrdiff import CorrDiffMethod
    with pytest.raises(ImportError, match="torch"):
        CorrDiffMethod()


def test_parse_variable_name():
    from africas2s.methods.corrdiff import parse_variable_name
    assert parse_variable_name("ua850") == ("ua", 850)
    assert parse_variable_name("ta10") == ("ta", 10)
    assert parse_variable_name("hus1000") == ("hus", 1000)
    assert parse_variable_name("tas") == ("tas", None)
    assert parse_variable_name("psl") == ("psl", None)
    assert parse_variable_name("sfcWind") == ("sfcWind", None)
    assert parse_variable_name("tasmax") == ("tasmax", None)
    assert parse_variable_name("va50") == ("va", 50)
    assert parse_variable_name("wap250") == ("wap", 250)


# ---------------------------------------------------------------------------
# Tier 2: Mock tests (require torch, mock earth2studio)
# ---------------------------------------------------------------------------

def test_fit_stores_obs_metadata(_mock_earth2studio):
    torch = pytest.importorskip("torch")
    from africas2s.methods.corrdiff import CorrDiffMethod

    m = CorrDiffMethod(device="cpu")
    m.fit(_make_hindcast(), _make_obs())

    np.testing.assert_array_equal(m.obs_lat_, _OBS_LAT)
    np.testing.assert_array_equal(m.obs_lon_, _OBS_LON)
    assert m.obs_clim_ is not None
    assert m.obs_clim_.dims == ("lat", "lon")


def test_predict_output_shape(_mock_earth2studio):
    """predict() should return (member, lat, lon) at the obs grid."""
    torch = pytest.importorskip("torch")
    from africas2s.methods.corrdiff import CorrDiffMethod

    m = CorrDiffMethod(device="cpu", n_samples=5, target_variable="t2m")
    m.fit(_make_hindcast(), _make_obs())

    # Build a minimal corrdiff_input tuple
    tensor = torch.randn(1, 1, 3, len(_FAKE_INPUT_VARS),
                         len(_MODEL_IN_LAT), len(_MODEL_IN_LON))
    coords = OrderedDict({
        "batch": np.array([0]),
        "time": np.array([np.datetime64("2005-03-15")]),
        "lead_time": np.array([
            np.timedelta64(-24, "h"),
            np.timedelta64(0, "h"),
            np.timedelta64(24, "h"),
        ]),
        "variable": np.array(_FAKE_INPUT_VARS),
        "lat": _MODEL_IN_LAT,
        "lon": _MODEL_IN_LON,
    })

    result = m.predict(_make_forecast(), corrdiff_input=(tensor, coords))

    assert isinstance(result, xr.DataArray)
    assert set(result.dims) == {"member", "lat", "lon"}
    assert result.sizes["member"] == 5
    assert result.sizes["lat"] == len(_OBS_LAT)
    assert result.sizes["lon"] == len(_OBS_LON)


def test_predict_requires_corrdiff_input(_mock_earth2studio):
    torch = pytest.importorskip("torch")
    from africas2s.methods.corrdiff import CorrDiffMethod

    m = CorrDiffMethod(device="cpu")
    m.fit(_make_hindcast(), _make_obs())

    with pytest.raises(ValueError, match="corrdiff_input"):
        m.predict(_make_forecast())


def test_predict_n_samples_override(_mock_earth2studio):
    torch = pytest.importorskip("torch")
    from africas2s.methods.corrdiff import CorrDiffMethod

    m = CorrDiffMethod(device="cpu", n_samples=3)
    m.fit(_make_hindcast(), _make_obs())

    tensor = torch.randn(1, 1, 3, len(_FAKE_INPUT_VARS),
                         len(_MODEL_IN_LAT), len(_MODEL_IN_LON))
    coords = OrderedDict({
        "batch": np.array([0]),
        "time": np.array([np.datetime64("2005-03-15")]),
        "lead_time": np.array([
            np.timedelta64(-24, "h"),
            np.timedelta64(0, "h"),
            np.timedelta64(24, "h"),
        ]),
        "variable": np.array(_FAKE_INPUT_VARS),
        "lat": _MODEL_IN_LAT,
        "lon": _MODEL_IN_LON,
    })

    result = m.predict(_make_forecast(), corrdiff_input=(tensor, coords),
                       n_samples=7)
    assert result.sizes["member"] == 7


def test_model_property_loads_lazily(_mock_earth2studio):
    torch = pytest.importorskip("torch")
    from africas2s.methods.corrdiff import CorrDiffMethod

    m = CorrDiffMethod(device="cpu")
    assert m._model is None
    _ = m.model
    assert m._model is not None


# ---------------------------------------------------------------------------
# prepare_corrdiff_input tests
# ---------------------------------------------------------------------------

def test_prepare_input_from_split_dataset(_mock_earth2studio):
    """Test prepare_corrdiff_input with pre-split variables."""
    torch = pytest.importorskip("torch")
    from africas2s.methods.corrdiff import prepare_corrdiff_input, CorrDiffMethod
    import pandas as pd

    m = CorrDiffMethod(device="cpu")
    model = m.model

    # Build a minimal dataset with all input variables on a coarse grid
    times = pd.date_range("2005-03-14", "2005-03-16", freq="D")
    lat = np.linspace(90, -90, 64)
    lon = np.linspace(0, 357.1875, 128)

    data_vars = {}
    for var_name in model.input_variables:
        data_vars[var_name] = xr.DataArray(
            np.random.randn(3, 64, 128).astype(np.float32),
            dims=["time", "lat", "lon"],
            coords={"time": times, "lat": lat, "lon": lon},
        )
    ds = xr.Dataset(data_vars)

    tensor, coords = prepare_corrdiff_input(ds, "2005-03-15", model)

    n_vars = len(model.input_variables)
    assert tensor.shape == (1, 1, 3, n_vars, 64, 128)
    assert "variable" in coords
    assert len(coords["variable"]) == n_vars
    assert coords["lead_time"][1] == np.timedelta64(0, "h")


def test_prepare_input_from_plev_dataset(_mock_earth2studio):
    """Test prepare_corrdiff_input with base variables + plev dim."""
    torch = pytest.importorskip("torch")
    from africas2s.methods.corrdiff import prepare_corrdiff_input, CorrDiffMethod
    import pandas as pd

    m = CorrDiffMethod(device="cpu")
    model = m.model

    times = pd.date_range("2005-03-14", "2005-03-16", freq="D")
    lat = np.linspace(90, -90, 64)
    lon = np.linspace(0, 357.1875, 128)
    plevs = np.array([10, 50, 100, 200, 250, 300, 500, 700, 850, 1000]) * 100  # Pa

    data_vars = {}
    # Add pressure-level variables with a plev dimension
    for base_var in ["ua", "va", "ta", "zg", "hus", "wap"]:
        data_vars[base_var] = xr.DataArray(
            np.random.randn(3, len(plevs), 64, 128).astype(np.float32),
            dims=["time", "plev", "lat", "lon"],
            coords={"time": times, "plev": plevs, "lat": lat, "lon": lon},
        )
    # Add all surface variables directly
    from africas2s.methods.corrdiff import parse_variable_name
    for var_name in model.input_variables:
        _, plev = parse_variable_name(var_name)
        if plev is None and var_name not in data_vars:
            data_vars[var_name] = xr.DataArray(
                np.random.randn(3, 64, 128).astype(np.float32),
                dims=["time", "lat", "lon"],
                coords={"time": times, "lat": lat, "lon": lon},
            )
    ds = xr.Dataset(data_vars)

    tensor, coords = prepare_corrdiff_input(ds, "2005-03-15", model)

    n_vars = len(model.input_variables)
    assert tensor.shape == (1, 1, 3, n_vars, 64, 128)


def test_prepare_input_missing_variable(_mock_earth2studio):
    """Should raise KeyError for missing variables."""
    torch = pytest.importorskip("torch")
    from africas2s.methods.corrdiff import prepare_corrdiff_input, CorrDiffMethod

    m = CorrDiffMethod(device="cpu")
    model = m.model

    ds = xr.Dataset()  # empty
    with pytest.raises(KeyError, match="not found in dataset"):
        prepare_corrdiff_input(ds, "2005-03-15", model)
