"""CorrDiff generative diffusion downscaling method.

Wraps NVIDIA's pre-trained CorrDiff model (nvidia/corrdiff-cmip6-era5) as a
africas2s downscaling method.  CorrDiff downscales coarse CMIP6 data (~300 km)
to ERA5-resolution (0.25 deg, ~25 km) using a two-step residual diffusion
architecture: a deterministic UNet predicts the mean, then a stochastic
diffusion model adds realistic fine-scale structure.

Requirements
------------
Install the GPU dependencies manually (they require NVIDIA package indices)::

    pip install torch              # https://pytorch.org
    pip install earth2studio       # https://github.com/NVIDIA/earth2studio
    pip install nvidia-physicsnemo # https://github.com/NVIDIA/physicsnemo

Usage
-----
CorrDiff requires the full multi-variable CMIP6 input, not just the single
variable that africas2s normally passes through ``forecast``.  Prepare the
input using :func:`prepare_corrdiff_input` and pass it via the ``corrdiff_input``
keyword argument::

    from africas2s.methods.corrdiff import prepare_corrdiff_input

    # cmip6_ds is an xr.Dataset with the 74 required variables on a daily
    # time axis.  See prepare_corrdiff_input() for the expected format.
    tensor, coords = prepare_corrdiff_input(cmip6_ds, target_time, model)

    result = africas2s.downscale(
        predictor_hindcast=gcm_hindcast,
        obs=obs,
        method="corrdiff",
        target_variable="t2m",
        corrdiff_input=(tensor, coords),
    )
"""

from collections import OrderedDict
import re

import numpy as np
import xarray as xr

from .base import MethodBase
from ..registry import register_method
from .._optional import require_optional


def _to_numpy(x):
    """Convert a torch.Tensor or numpy array to numpy."""
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Variable name helpers
# ---------------------------------------------------------------------------

# CMIP6 base variable names that have pressure-level variants.
_PLEV_BASE_VARS = {"ua", "va", "ta", "zg", "hus", "wap"}

# Regex: variable name optionally ending in a pressure level (integer).
_PLEV_RE = re.compile(r"^([a-zA-Z]+?)(\d+)$")


def parse_variable_name(name):
    """Decompose a CorrDiff variable name into (base_var, pressure_level).

    Pressure-level variables like ``"ua850"`` return ``("ua", 850)``.
    Surface variables like ``"tas"`` return ``("tas", None)``.

    Parameters
    ----------
    name : str
        A variable name from the CorrDiff input or output list.

    Returns
    -------
    tuple[str, int | None]
    """
    m = _PLEV_RE.match(name)
    if m:
        base, level_str = m.group(1), m.group(2)
        if base in _PLEV_BASE_VARS:
            return base, int(level_str)
    return name, None


def prepare_corrdiff_input(dataset, target_time, model):
    """Convert an xarray CMIP6 Dataset into CorrDiff model input.

    This function assembles the 3-day window of input variables that
    ``CorrDiffCMIP6`` expects (74 variables as of earth2studio 0.13),
    interpolates to the model's input grid, and returns a
    ``(tensor, coords)`` tuple ready to pass to ``predict()``.

    Parameters
    ----------
    dataset : xr.Dataset
        CMIP6 data containing the variables listed in
        ``model.input_variables``.  Must have a ``time`` dimension with
        at least three consecutive daily time steps centred on
        *target_time*.

        Variables can be provided in two ways:

        * **Already split** -- each variable is a separate DataArray named
          exactly as earth2studio expects (e.g. ``"ua850"``, ``"tas"``).
          DataArrays should have dims ``(time, lat, lon)``.

        * **With a pressure-level dim** -- 3-D atmospheric variables are
          provided with their CMIP6 base name (e.g. ``"ua"``), with an
          extra ``plev`` dimension in **Pa**.  The function will select the
          required levels automatically.

    target_time : datetime-like
        The centre of the 3-day window.  ``target_time - 1 day`` and
        ``target_time + 1 day`` must also be present in *dataset*.

    model : CorrDiffCMIP6
        A loaded ``CorrDiffCMIP6`` model instance (from
        ``CorrDiffCMIP6.from_pretrained()``).  Used to retrieve the expected
        input variable list and grid coordinates.

    Returns
    -------
    tensor : torch.Tensor
        Shape ``(1, 1, 3, n_vars, lat, lon)`` -- batch=1, time=1,
        lead_time=3, variables, and the model's input grid.
    coords : OrderedDict
        A ``CoordSystem`` compatible with ``model.__call__()``.
    """
    import torch
    import pandas as pd

    input_vars = list(model.input_variables)
    target_time = pd.Timestamp(target_time)
    offsets = [pd.Timedelta(days=-1), pd.Timedelta(0), pd.Timedelta(days=1)]
    times = [target_time + dt for dt in offsets]

    model_lats = _to_numpy(model.lat_input_grid)
    model_lons = _to_numpy(model.lon_input_grid)
    n_lat, n_lon = len(model_lats), len(model_lons)

    # Assemble the 3-D array: (lead_time=3, variable, lat, lon)
    data = np.empty((3, len(input_vars), n_lat, n_lon), dtype=np.float32)

    for v_idx, var_name in enumerate(input_vars):
        base_var, plev = parse_variable_name(var_name)

        # Try exact name first (pre-split dataset), then base name + plev
        if var_name in dataset:
            da = dataset[var_name]
        elif base_var in dataset and plev is not None:
            # Select pressure level.  CMIP6 plev is in Pa; convert hPa->Pa.
            plev_pa = plev * 100
            da = dataset[base_var]
            if "plev" in da.dims:
                da = da.sel(plev=plev_pa, method="nearest")
            elif "level" in da.dims:
                da = da.sel(level=plev, method="nearest")
        else:
            raise KeyError(
                f"Variable {var_name!r} (base={base_var!r}, plev={plev}) "
                f"not found in dataset.  Available: {list(dataset.data_vars)}"
            )

        for t_idx, t in enumerate(times):
            field = da.sel(time=t, method="nearest")
            # Interpolate to model grid if shapes don't match
            if field.shape != (n_lat, n_lon):
                field = field.interp(
                    lat=model_lats, lon=model_lons, method="linear",
                )
            data[t_idx, v_idx] = field.values

    # Shape: (batch=1, time=1, lead_time=3, variable, lat, lon)
    tensor = torch.as_tensor(data[np.newaxis, np.newaxis], dtype=torch.float32)

    coords = OrderedDict({
        "batch": np.array([0]),
        "time": np.array([target_time.to_datetime64()]),
        "lead_time": np.array([
            np.timedelta64(-24, "h"),
            np.timedelta64(0, "h"),
            np.timedelta64(24, "h"),
        ]),
        "variable": np.array(input_vars),
        "lat": model_lats,
        "lon": model_lons,
    })

    return tensor, coords


# ---------------------------------------------------------------------------
# Method class
# ---------------------------------------------------------------------------

@register_method("corrdiff")
class CorrDiffMethod(MethodBase):
    """Generative diffusion downscaling using a pre-trained CorrDiff model.

    Unlike BCSD/CCA, CorrDiff is pre-trained on global data and is not
    retrained on the user's hindcast.  The ``fit()`` method stores observation
    grid metadata for spatial subsetting; it does not update the neural network
    weights.

    Ensembles are generated by the diffusion model's stochastic sampling --
    ``n_samples`` controls how many independent realisations are drawn.

    Parameters
    ----------
    device : str
        PyTorch device string (``"cuda"``, ``"cuda:0"``, ``"cpu"``).
    n_samples : int
        Number of ensemble members to generate per inference call.
    target_variable : str
        Which of CorrDiff's output variables to extract and return.
        Use ``model.output_variables`` after loading to see the full list.
    """

    is_pretrained = True

    def __init__(self, device="cuda", n_samples=10, target_variable="t2m"):
        require_optional("torch", install_hint="pip install torch (see https://pytorch.org)")
        self.device = device
        self.n_samples = n_samples
        self.target_variable = target_variable
        self._model = None

    def save(self, path):
        raise NotImplementedError(
            "CorrDiff save/load is not supported yet: the pretrained torch model "
            "isn't picklable. Tracked for the DL-training work (#27/#28)."
        )

    load = save  # same deferral message for both directions

    def _load_model(self):
        """Lazy-load the CorrDiff model from HuggingFace."""
        if self._model is not None:
            return
        from earth2studio.models.dx import CorrDiffCMIP6

        self._model = CorrDiffCMIP6.from_pretrained()
        self._model = self._model.to(self.device)
        self._model.number_of_samples = self.n_samples

        # Cache output variable index for the target variable
        out_vars = list(self._model.output_variables)
        if self.target_variable not in out_vars:
            raise ValueError(
                f"target_variable {self.target_variable!r} not in model "
                f"output variables: {out_vars}"
            )
        self._target_var_idx = out_vars.index(self.target_variable)

    @property
    def model(self):
        """The underlying earth2studio model (loaded on first access)."""
        self._load_model()
        return self._model

    def fit(self, hindcast, obs, **kwargs):
        """Store observation grid metadata and load the model.

        The neural network weights are not modified.  This method records the
        observation coordinates so that ``predict()`` can subset CorrDiff's
        global output to the user's region.
        """
        self._load_model()
        self.obs_lat_ = obs.lat.values.copy()
        self.obs_lon_ = obs.lon.values.copy()
        self.obs_clim_ = obs.mean("year")

    def predict(self, forecast, **kwargs):
        """Run CorrDiff inference and return the target variable.

        Parameters
        ----------
        forecast : xr.DataArray
            Standard africas2s forecast with dims ``(member, lat, lon)``.
            Used only for metadata.  The actual model input comes from
            ``corrdiff_input``.
        corrdiff_input : tuple[torch.Tensor, dict]
            A ``(tensor, coords)`` pair as returned by
            :func:`prepare_corrdiff_input`.  The tensor has shape
            ``(1, 1, 3, n_vars, lat, lon)`` and coords is an
            ``OrderedDict`` with keys
            ``batch, time, lead_time, variable, lat, lon``.
        n_samples : int, optional
            Override the instance-level ``n_samples``.

        Returns
        -------
        xr.DataArray
            Downscaled field with dims ``(member, lat, lon)`` at the
            observation grid resolution.
        """
        import torch

        corrdiff_input = kwargs.get("corrdiff_input")
        if corrdiff_input is None:
            raise ValueError(
                "CorrDiff requires prepared multi-variable input.  "
                "Use prepare_corrdiff_input() to build the input, then pass "
                "corrdiff_input=(tensor, coords) as a keyword argument."
            )

        input_tensor, input_coords = corrdiff_input
        n_samples = kwargs.get("n_samples", self.n_samples)

        # Move input to device
        if not isinstance(input_tensor, torch.Tensor):
            input_tensor = torch.as_tensor(input_tensor, dtype=torch.float32)
        input_tensor = input_tensor.to(self.device)

        # Configure ensemble size and run inference
        self._model.number_of_samples = n_samples
        with torch.no_grad():
            output, output_coords = self._model(input_tensor, input_coords)

        # output shape: (batch, sample, time, lead_time, variable, lat, lon)
        # We want: the target variable, first batch, first time, first lead_time
        var_idx = self._target_var_idx
        # (sample, lat, lon)
        field = output[0, :, 0, 0, var_idx].cpu().numpy()

        # Wrap in xarray on the model's output grid
        out_lats = _to_numpy(output_coords["lat"])
        out_lons = _to_numpy(output_coords["lon"])
        global_da = xr.DataArray(
            field,
            dims=["member", "lat", "lon"],
            coords={
                "member": np.arange(n_samples),
                "lat": out_lats,
                "lon": out_lons,
            },
        )

        # Subset to the user's regional observation grid
        return self._subset_to_obs(global_da)

    def _subset_to_obs(self, global_da):
        """Subset and interpolate the global CorrDiff output to the obs grid."""
        lat_min, lat_max = self.obs_lat_.min(), self.obs_lat_.max()
        lon_min, lon_max = self.obs_lon_.min(), self.obs_lon_.max()

        # Determine if lats are descending (90 -> -90) or ascending
        lats = global_da.lat.values
        if len(lats) > 1 and lats[0] > lats[-1]:
            lat_slice = slice(lat_max + 0.5, lat_min - 0.5)
        else:
            lat_slice = slice(lat_min - 0.5, lat_max + 0.5)

        sub = global_da.sel(
            lat=lat_slice,
            lon=slice(lon_min - 0.5, lon_max + 0.5),
        )

        return sub.interp(
            lat=self.obs_lat_, lon=self.obs_lon_, method="linear",
        )
