"""Cross-validated prediction error variance per grid cell.

See docs/superpowers/specs/2026-05-15-prediction-error-variance-design.md
for the full contract and rationale.
"""

import xarray as xr


def prediction_error_variance(
    cv_predictions: xr.DataArray,
    obs: xr.DataArray,
) -> xr.DataArray:
    """Cross-validated prediction error variance per grid cell.

    Returns the year-mean of squared residuals between cross-validated
    predictions and observations. Plain MSE-style estimator — no
    degrees-of-freedom correction, no leverage inflation.

    Parameters
    ----------
    cv_predictions : xr.DataArray
        Cross-validated predictions; must include a ``year`` dim.
    obs : xr.DataArray
        Observations covering the same set of years as ``cv_predictions``
        (order need not match — alignment is by coordinate value).

    Returns
    -------
    xr.DataArray
        Per-cell variance with the ``year`` dim removed. NaN propagates
        per cell (e.g. ocean cells in a land-only obs grid stay NaN).

    Raises
    ------
    ValueError
        If the two inputs don't cover the same set of years.

    Notes
    -----
    For probabilistic-by-construction methods (CorrDiff samples, EMOS
    predictive σ), the residual-based PEV returned here estimates
    uncertainty of the *mean* prediction. Those methods carry richer
    native predictive distributions that should be preferred where
    available; this function is the right tool for deterministic
    methods (CCA, BCSD, regression-style downscaling).

    The CPT-compatible tercile path (``tercile._tercile_cpt``) uses an
    unbiased variant ``sum / (n - n_modes - 1)`` plus per-year leverage
    inflation. That estimator is method-specific (CCA's EOF mode count
    enters the denominator) and stays where it is — not a duplicate of
    this function but a deliberately different statistical choice.
    """
    pred_years = set(cv_predictions.year.values.tolist())
    obs_years = set(obs.year.values.tolist())
    if pred_years != obs_years:
        only_in_pred = sorted(pred_years - obs_years)
        only_in_obs = sorted(obs_years - pred_years)
        raise ValueError(
            "prediction_error_variance: cv_predictions and obs must cover the "
            "same set of years. "
            f"Only in cv_predictions: {only_in_pred}; only in obs: {only_in_obs}"
        )
    obs_aligned = obs.sel(year=cv_predictions.year)
    residuals = cv_predictions - obs_aligned
    return (residuals ** 2).mean("year")
