import warnings
import xarray as xr
from .registry import get_method
from .tercile import to_tercile
from .methods.base import ProbabilisticMethodBase

_METHOD_PARAMS = ("n_modes", "x_eof_modes", "y_eof_modes", "cca_modes",
                  "device", "n_samples", "target_variable", "variant")


def _log(verbose, msg):
    if verbose:
        print(f"[africas2s] {msg}")


def downscale(predictor_hindcast=None, obs=None, method="bcsd",
              output_type="continuous", weights_path=None, **kwargs):
    # Deprecation shim: `gcm=` was the old name for `predictor_hindcast`.
    legacy_gcm = kwargs.pop("gcm", None)
    if legacy_gcm is not None:
        if predictor_hindcast is not None:
            raise TypeError(
                "downscale() received both `predictor_hindcast` and the deprecated "
                "`gcm` keyword. Use `predictor_hindcast` only."
            )
        warnings.warn(
            "`gcm=` is deprecated; use `predictor_hindcast=` instead.",
            DeprecationWarning, stacklevel=2,
        )
        predictor_hindcast = legacy_gcm

    if predictor_hindcast is None:
        raise TypeError("downscale() requires `predictor_hindcast`.")
    if output_type == "tercile" and obs is None:
        raise TypeError("downscale(output_type='tercile') requires `obs` for the "
                        "tercile boundaries.")

    verbose = kwargs.pop("verbose", True)
    forecast = kwargs.pop("forecast", None)
    method_cls = get_method(method)
    ctor_kwargs = {k: v for k, v in kwargs.items() if k in _METHOD_PARAMS}

    if weights_path is not None:
        # Inference-only: load a pre-trained checkpoint, skip fit (§10.2 #27).
        _log(verbose, f"downscale: loading weights from {weights_path} (method={method})")
        m = method_cls(**ctor_kwargs)
        m.load(weights_path)
        if forecast is None:
            forecast = (predictor_hindcast.isel(year=-1, drop=True)
                        if "year" in predictor_hindcast.dims else predictor_hindcast)
    else:
        if obs is None:
            raise TypeError("downscale() requires `obs` for training (or pass "
                            "`weights_path=` to run inference on a saved model).")
        if getattr(method_cls, "requires_training", False):
            raise RuntimeError(
                f"Method '{method}' requires separate training. Run "
                f"africas2s.training.train('{method}', hindcast, obs, save_to=PATH) "
                f"first, then downscale(..., weights_path=PATH)."
            )
        _log(verbose, f"downscale start: method={method}")
        fit_predictor, fit_obs = predictor_hindcast, obs
        if "year" in predictor_hindcast.dims and forecast is None:
            if len(predictor_hindcast.year) < 2:
                raise ValueError("Need at least 2 years when forecast is not provided.")
            fit_predictor = predictor_hindcast.isel(year=slice(None, -1))
            fit_obs = obs.sel(year=fit_predictor.year)
            forecast = predictor_hindcast.isel(year=-1, drop=True)
        m = method_cls(**ctor_kwargs)
        m.fit(fit_predictor, fit_obs, **kwargs)
        forecast = predictor_hindcast if forecast is None else forecast

    # Probabilistic methods produce a native ensemble; the tercile path counts
    # members directly (no Gaussian fit-to-deterministic). Deterministic methods
    # keep the existing predict() + to_tercile() path.
    if output_type == "tercile" and isinstance(m, ProbabilisticMethodBase):
        ensemble = m.predict_distribution(forecast, **kwargs)
        result = to_tercile(ensemble, obs, method="counting")
    else:
        result = m.predict(forecast, **kwargs)
        if output_type == "tercile":
            result = to_tercile(result, obs)

    _log(verbose, "downscale complete")
    return result
