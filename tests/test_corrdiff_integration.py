"""Integration tests for CorrDiff that require a GPU and earth2studio.

Run with::

    pytest tests/test_corrdiff_integration.py -m gpu
"""

import numpy as np
import pytest
import xarray as xr

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA GPU"),
]

try:
    from earth2studio.models.dx import CorrDiffCMIP6  # noqa: F401
    _HAS_EARTH2STUDIO = True
except ImportError:
    _HAS_EARTH2STUDIO = False


@pytest.mark.skipif(not _HAS_EARTH2STUDIO, reason="earth2studio not installed")
class TestCorrDiffGPU:
    """Tests that load the real CorrDiff model and run inference."""

    @pytest.fixture(scope="class")
    def method(self):
        from africas2s.methods.corrdiff import CorrDiffMethod
        m = CorrDiffMethod(device="cuda", n_samples=2, target_variable="t2m")
        obs = xr.DataArray(
            np.random.randn(5, 20, 20) + 290.0,
            dims=["year", "lat", "lon"],
            coords={
                "year": np.arange(2000, 2005),
                "lat": np.linspace(20, 25, 20),
                "lon": np.linspace(118, 123, 20),
            },
        )
        hindcast = xr.DataArray(
            np.random.randn(5, 3, 5, 5) + 290.0,
            dims=["year", "member", "lat", "lon"],
            coords={
                "year": np.arange(2000, 2005),
                "member": np.arange(3),
                "lat": np.linspace(20, 25, 5),
                "lon": np.linspace(118, 123, 5),
            },
        )
        m.fit(hindcast, obs)
        return m

    def test_model_loads_and_has_variables(self, method):
        model = method.model
        assert len(model.input_variables) > 0
        assert len(model.output_variables) > 0
        assert hasattr(model, "lat_input_grid")
        assert hasattr(model, "lon_input_grid")

    def test_prepare_and_predict(self, method):
        import pandas as pd
        from africas2s.methods.corrdiff import prepare_corrdiff_input, _to_numpy

        model = method.model
        times = pd.date_range("2005-03-14", "2005-03-16", freq="D")
        lat = _to_numpy(model.lat_input_grid)
        lon = _to_numpy(model.lon_input_grid)

        data_vars = {}
        for var_name in model.input_variables:
            data_vars[var_name] = xr.DataArray(
                np.random.randn(3, len(lat), len(lon)).astype(np.float32),
                dims=["time", "lat", "lon"],
                coords={"time": times, "lat": lat, "lon": lon},
            )
        ds = xr.Dataset(data_vars)

        tensor, coords = prepare_corrdiff_input(ds, "2005-03-15", model)
        forecast = xr.DataArray(
            np.random.randn(3, 5, 5) + 290.0,
            dims=["member", "lat", "lon"],
            coords={
                "member": np.arange(3),
                "lat": np.linspace(20, 25, 5),
                "lon": np.linspace(118, 123, 5),
            },
        )
        result = method.predict(forecast, corrdiff_input=(tensor, coords))

        assert isinstance(result, xr.DataArray)
        assert set(result.dims) == {"member", "lat", "lon"}
        assert result.sizes["member"] == 2
        np.testing.assert_array_almost_equal(result.lat.values, method.obs_lat_)
        np.testing.assert_array_almost_equal(result.lon.values, method.obs_lon_)
