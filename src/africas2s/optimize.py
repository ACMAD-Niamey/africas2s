from dataclasses import dataclass
import numpy as np
import xarray as xr
from .registry import get_method
from .cv import get_cv
from .skill import skill
from .tercile import to_tercile

_METHOD_PARAMS = ("n_modes", "x_eof_modes", "y_eof_modes", "cca_modes",
                  "device", "n_samples", "target_variable")


def _iter(items, desc, enabled=True):
    if not enabled:
        return items
    try:
        from tqdm.auto import tqdm
        return tqdm(items, desc=desc)
    except Exception:
        return items


def _log(verbose, msg):
    if verbose:
        print(f"[africas2s] {msg}")


@dataclass
class OptimizeResult:
    method: str
    score: float
    forecast: xr.DataArray

def optimize(gcm, obs, methods=None, cv="loyo", primary_metric="rpss", **kwargs):
    progress = kwargs.pop("progress", True)
    verbose = kwargs.pop("verbose", True)
    if methods is None:
        methods = ["bcsd", "cca"]

    best_score = -np.inf
    best_method = None
    best_forecast = None

    years = list(gcm.year.values)
    # Accept either a registered name or a pre-configured callable, so callers
    # can pass e.g. `partial(expanding, min_train=4)` for a custom CV scheme.
    cv_fn = cv if callable(cv) else get_cv(cv)
    cv_label = cv if isinstance(cv, str) else getattr(cv_fn, "__name__", "callable")
    _log(verbose, f"optimize start: methods={len(methods)}, cv={cv_label}, years={len(years)}")

    for method_name in _iter(methods, "AfricaS2S methods", enabled=progress):
        _log(verbose, f"scoring method={method_name}")
        cv_forecasts = []
        cv_obs_list = []

        method_cls = get_method(method_name)
        is_pretrained = getattr(method_cls, "is_pretrained", False)

        # Pre-trained methods (e.g. CorrDiff) are not retrained per fold.
        # Create once and fit on the full dataset to store metadata.
        if is_pretrained:
            m = method_cls(**{k: v for k, v in kwargs.items() if k in _METHOD_PARAMS})
            m.fit(gcm, obs, **kwargs)

        for train_years, test in _iter(list(cv_fn(years)), f"{method_name} folds", enabled=progress):
            # CV schemes yield either a scalar test year (loyo/expanding) or
            # a list of consecutive test years (lko/blocked). Normalize.
            test_years = test if isinstance(test, list) else [test]
            obs_train = obs.sel(year=train_years)

            if not is_pretrained:
                m = method_cls(**{k: v for k, v in kwargs.items() if k in _METHOD_PARAMS})
                m.fit(gcm.sel(year=train_years), obs_train, **kwargs)

            for test_year in test_years:
                pred = m.predict(gcm.sel(year=test_year), **kwargs)
                tercile_pred = to_tercile(pred, obs_train).expand_dims(year=[test_year])
                cv_forecasts.append(tercile_pred)
                cv_obs_list.append(obs.sel(year=[test_year]))

        # Schemes that overlap (e.g. lko sliding) produce multiple predictions
        # per year; average them into one prediction per year before scoring.
        # Schemes that don't overlap pass through unchanged.
        cv_fcst = xr.concat(cv_forecasts, dim="year").groupby("year").mean("year")
        cv_obs = xr.concat(cv_obs_list, dim="year").groupby("year").first()

        report = skill(cv_fcst, cv_obs, metrics=[primary_metric])
        score = report.scores.get(
            primary_metric, report.scores.get(list(report.scores.keys())[0])
        )

        if score > best_score:
            best_score = score
            best_method = method_name
            method_cls = get_method(method_name)
            m = method_cls(**{k: v for k, v in kwargs.items() if k in _METHOD_PARAMS})
            m.fit(gcm, obs, **kwargs)
            final_fcst = kwargs.get("forecast", gcm.mean("year") if "year" in gcm.dims else gcm)
            best_forecast = m.predict(final_fcst, **kwargs)
            _log(verbose, f"new best: {best_method} ({best_score:.4f})")

    _log(verbose, f"optimize complete: best={best_method}, score={best_score:.4f}")
    return OptimizeResult(method=best_method, score=best_score, forecast=best_forecast)
