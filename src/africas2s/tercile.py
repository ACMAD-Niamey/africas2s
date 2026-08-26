"""Tercile-probability conversion for forecasts.

There are two callable surfaces:

- `to_tercile(forecast, obs_climatology, ...)` is the **production-time** path:
  convert a single-year ensemble forecast into tercile probabilities using
  thresholds computed from the supplied historical climatology. The caller
  is responsible for the `obs_climatology` argument; pass only the years
  you would have known when issuing the forecast.

- `to_tercile_cv(cv_predictions, obs, ...)` is the **scoring** path: convert
  a year-stacked array of cross-validated predictions into tercile probs
  with proper held-out discipline (LOO boundaries, LOO PEV, etc).

**Leakage hazard.** Using `to_tercile(year_pred, obs)` to score CV hindcasts
is a common bug: the tercile boundaries computed from `obs` will include
the held-out year, inflating apparent skill. For any CV-scoring context,
prefer `to_tercile_cv()`. The default boundary convention there is
`cpt_boundaries=True`, which matches the CPT/PyCPT reference; pass
`cpt_boundaries=False` only when reproducing legacy non-CPT runs.
"""

import numpy as np
import xarray as xr
from scipy.stats import norm, t as t_dist


def _cpt_tercile_probs(loc, t33, t67, dofr, pesd):
    """CPT Student-t below/normal/above probabilities for one forecast field.

    ``loc``/``t33``/``t67``/``pesd`` are same-shape numpy arrays; ``dofr`` is the
    residual degrees of freedom. NaN in any input propagates to NaN across all
    three categories. This is the single CPT tercile kernel shared by the CV
    path (`_tercile_cpt`) and the production per-model path (`cpt_tercile_forecast`).
    """
    p_bn = t_dist.cdf(t33, df=dofr, loc=loc, scale=pesd)
    p_an = t_dist.sf(t67, df=dofr, loc=loc, scale=pesd)
    p_nn = 1.0 - p_bn - p_an
    nan_mask = np.isnan(loc) | np.isnan(t33) | np.isnan(t67) | np.isnan(pesd)
    if nan_mask.any():
        p_bn[nan_mask] = np.nan
        p_nn[nan_mask] = np.nan
        p_an[nan_mask] = np.nan
    return p_bn, p_nn, p_an


def cpt_tercile_forecast(forecast, t33, t67, s2, dofr, leverage=0.0):
    """CPT-style Student-t tercile probabilities for a single forecast map.

    ``forecast``, ``t33``, ``t67`` and ``s2`` are DataArrays on the same grid;
    the boundary/PEV arrays are aligned to ``forecast``'s dim order before the
    (numpy) Student-t evaluation, so a transposed input can't mismap
    probabilities onto the wrong cells. ``s2`` is the unbiased PEV; the scale is
    ``sqrt(s2 * (1 + leverage))`` (leverage-inflated, Wilks 2006 / CPT). Returns
    ``(tercile, *spatial)`` probabilities.
    """
    spatial = list(forecast.dims)
    t33 = t33.transpose(*spatial)
    t67 = t67.transpose(*spatial)
    s2 = s2.transpose(*spatial)
    pesd = np.sqrt(np.maximum(s2.values * (1.0 + leverage), 1e-24))
    p_bn, p_nn, p_an = _cpt_tercile_probs(
        forecast.values, t33.values, t67.values, dofr, pesd)
    out = xr.concat(
        [xr.DataArray(p_bn, dims=spatial, coords=forecast.coords),
         xr.DataArray(p_nn, dims=spatial, coords=forecast.coords),
         xr.DataArray(p_an, dims=spatial, coords=forecast.coords)],
        dim=xr.IndexVariable("tercile", [0, 1, 2]),
    )
    return out


def to_tercile(forecast, obs_climatology, method="counting"):
    """Convert a single-year ensemble forecast into tercile probabilities.

    For CV scoring contexts, use `to_tercile_cv()` instead — passing full
    obs to this function leaks the held-out year into the boundaries.

    forecast: (member, lat, lon) - continuous ensemble forecast
    obs_climatology: (year, lat, lon) - historical obs for computing thresholds
    method: "counting" (member counting) or "gaussian" (parametric fit)
    Returns: (tercile, lat, lon) - probabilities
    """
    t33 = obs_climatology.quantile(1 / 3, dim="year").drop_vars("quantile")
    t67 = obs_climatology.quantile(2 / 3, dim="year").drop_vars("quantile")

    if method == "gaussian":
        mu = forecast.mean("member")
        sigma = forecast.std("member")
        # Avoid division by zero for constant ensembles
        sigma = xr.where(sigma < 1e-12, 1e-12, sigma)
        p_bn = xr.DataArray(
            norm.cdf(t33.values, loc=mu.values, scale=sigma.values),
            dims=mu.dims, coords=mu.coords,
        )
        p_an = xr.DataArray(
            1.0 - norm.cdf(t67.values, loc=mu.values, scale=sigma.values),
            dims=mu.dims, coords=mu.coords,
        )
        p_nn = 1.0 - p_bn - p_an
    else:
        n_members = len(forecast.member)
        p_bn = (forecast < t33).sum("member") / n_members
        p_an = (forecast > t67).sum("member") / n_members
        p_nn = 1.0 - p_bn - p_an

    result = xr.concat([p_bn, p_nn, p_an], dim="tercile")
    result["tercile"] = [0, 1, 2]
    return result


def to_tercile_cv(cv_predictions, obs, method="bootstrap", leverages=None, n_modes=3,
                  cpt_boundaries=True):
    """
    Compute tercile probabilities from cross-validated predictions.

    Parameters
    ----------
    cv_predictions : xr.DataArray, dims (year, lat, lon)
        Cross-validated continuous predictions (e.g. ensemble mean from LOYO).
    obs : xr.DataArray, dims (year, lat, lon)
        Observations matching the same years.
    method : str
        "cpt" — CPT-compatible: Student's t with leverage-inflated PEV.
            Matches CPT/PyCPT's approach: PEV is inflated by the hat-matrix
            diagonal (1 + h_i) to account for parameter estimation uncertainty.
            Requires `leverages` array from CCA fit.
        "bootstrap" — Residual bootstrap with full LOO.
        "gaussian_loo" — Gaussian with LOO PEV and LOO tercile boundaries.
        "t" — Student's t-distribution with PEV (no leverage correction).
        "gaussian_pev" — Gaussian with PEV.
    leverages : array-like, optional
        Per-year hat-matrix leverage values from CCA fit. Required for "cpt".

    Returns
    -------
    xr.DataArray, dims (year, tercile, lat, lon)
        Tercile probabilities for each year.
    """
    n = len(obs.year)
    residuals = cv_predictions - obs

    if method == "cpt":
        return _tercile_cpt(cv_predictions, obs, residuals, n, leverages,
                            n_modes=n_modes, cpt_boundaries=cpt_boundaries)
    elif method == "bootstrap":
        return _tercile_bootstrap(cv_predictions, obs, residuals)
    elif method == "gaussian_loo":
        return _tercile_parametric_loo(cv_predictions, obs, residuals, dist="gaussian")
    else:
        return _tercile_parametric_global(cv_predictions, obs, residuals, n, method)


def _tercile_cpt(cv_predictions, obs, residuals, n, leverages, n_modes=3,
                 cpt_boundaries=False):
    """CPT-compatible tercile probabilities: Student's t with leverage-inflated PEV.

    From CPT Fortran source (regression.F95 / distribs.F95):
      s2 = sum(residuals^2) / dofr        (unbiased PEV estimate)
      pesd = sqrt(s2 * (1 + h_i))         (leverage-inflated std dev)
      dofr = n - m - 1                    (m = number of X EOF modes)
      P(below) = t_cdf(threshold, dofr, loc=forecast, scale=pesd)
    """
    dofr = n - n_modes - 1
    if cpt_boundaries:
        from africas2s.metrics.rpss import _cpt_boundaries
        t33_arr, t67_arr = _cpt_boundaries(obs.values)
        spatial_dims = [d for d in obs.dims if d != "year"]
        spatial_coords = {k: v for k, v in obs.coords.items()
                          if k != "year" and set(obs[k].dims).issubset(set(spatial_dims))}
        t33 = xr.DataArray(t33_arr, dims=spatial_dims, coords=spatial_coords)
        t67 = xr.DataArray(t67_arr, dims=spatial_dims, coords=spatial_coords)
    else:
        t33 = obs.quantile(1 / 3, dim="year").drop_vars("quantile")
        t67 = obs.quantile(2 / 3, dim="year").drop_vars("quantile")

    # Unbiased PEV: sum of squared residuals / (n - m - 1)
    s2 = (residuals ** 2).sum("year").values / dofr

    results = []
    for i, yr in enumerate(obs.year.values):
        pred = cv_predictions.sel(year=yr)
        loc = pred.values
        h_i = leverages[i] if leverages is not None else 0.0
        pesd = np.sqrt(np.maximum(s2 * (1 + h_i), 1e-24))

        p_bn_vals, p_nn_vals, p_an_vals = _cpt_tercile_probs(
            loc, t33.values, t67.values, dofr, pesd)

        spatial_dims = [d for d in pred.dims if d != "year"]
        spatial_coords = {k: v for k, v in pred.coords.items() if k != "year"}

        tercile = xr.concat(
            [xr.DataArray(p_bn_vals, dims=spatial_dims, coords=spatial_coords),
             xr.DataArray(p_nn_vals, dims=spatial_dims, coords=spatial_coords),
             xr.DataArray(p_an_vals, dims=spatial_dims, coords=spatial_coords)],
            dim="tercile",
        )
        tercile["tercile"] = [0, 1, 2]
        results.append(tercile)

    out = xr.concat(results, dim="year")
    out["year"] = obs.year.values
    return out


def _tercile_bootstrap(cv_predictions, obs, residuals):
    """Residual-bootstrap tercile probabilities with full LOO."""
    results = []
    for yr in obs.year.values:
        mask = obs.year != yr
        obs_loo = obs.sel(year=mask)
        res_loo = residuals.sel(year=mask)

        t33 = obs_loo.quantile(1 / 3, dim="year").drop_vars("quantile")
        t67 = obs_loo.quantile(2 / 3, dim="year").drop_vars("quantile")

        pred = cv_predictions.sel(year=yr, drop=True)
        # Synthetic ensemble: plausible observations given prediction
        ensemble = pred - res_loo  # (year_loo, lat, lon)

        n_members = len(ensemble.year)
        p_bn = (ensemble < t33).sum("year").astype(float) / n_members
        p_an = (ensemble > t67).sum("year").astype(float) / n_members
        p_nn = 1.0 - p_bn - p_an

        # Propagate NaN where pred or obs boundaries are NaN
        nan_mask = np.isnan(pred.values) | np.isnan(t33.values)
        p_bn = p_bn.where(~xr.DataArray(nan_mask, dims=p_bn.dims, coords=p_bn.coords))
        p_an = p_an.where(~xr.DataArray(nan_mask, dims=p_an.dims, coords=p_an.coords))
        p_nn = p_nn.where(~xr.DataArray(nan_mask, dims=p_nn.dims, coords=p_nn.coords))

        tercile = xr.concat([p_bn, p_nn, p_an], dim="tercile")
        tercile["tercile"] = [0, 1, 2]
        results.append(tercile)

    out = xr.concat(results, dim="year")
    out["year"] = obs.year.values
    return out


def _tercile_parametric_loo(cv_predictions, obs, residuals, dist="gaussian"):
    """Parametric tercile probabilities with LOO PEV and boundaries."""
    results = []
    for yr in obs.year.values:
        mask = obs.year != yr
        obs_loo = obs.sel(year=mask)
        res_loo = residuals.sel(year=mask)

        t33 = obs_loo.quantile(1 / 3, dim="year").drop_vars("quantile")
        t67 = obs_loo.quantile(2 / 3, dim="year").drop_vars("quantile")

        pev_loo = (res_loo ** 2).mean("year")
        scale = np.sqrt(np.maximum(pev_loo.values, 1e-24))

        pred = cv_predictions.sel(year=yr)
        loc = pred.values

        if dist == "t":
            n_loo = int((~mask).sum())  # should be n-1
            p_bn_vals = t_dist.cdf(t33.values, df=n_loo, loc=loc, scale=scale)
            p_an_vals = t_dist.sf(t67.values, df=n_loo, loc=loc, scale=scale)
        else:
            p_bn_vals = norm.cdf(t33.values, loc=loc, scale=scale)
            p_an_vals = 1.0 - norm.cdf(t67.values, loc=loc, scale=scale)

        p_nn_vals = 1.0 - p_bn_vals - p_an_vals

        # Propagate NaN from obs or predictions (e.g. ocean/coastal cells)
        nan_mask = np.isnan(t33.values) | np.isnan(loc) | np.isnan(scale)
        if nan_mask.any():
            p_bn_vals[nan_mask] = np.nan
            p_nn_vals[nan_mask] = np.nan
            p_an_vals[nan_mask] = np.nan

        spatial_dims = [d for d in pred.dims if d != "year"]
        spatial_coords = {k: v for k, v in pred.coords.items() if k != "year"}

        tercile = xr.concat(
            [xr.DataArray(p_bn_vals, dims=spatial_dims, coords=spatial_coords),
             xr.DataArray(p_nn_vals, dims=spatial_dims, coords=spatial_coords),
             xr.DataArray(p_an_vals, dims=spatial_dims, coords=spatial_coords)],
            dim="tercile",
        )
        tercile["tercile"] = [0, 1, 2]
        results.append(tercile)

    out = xr.concat(results, dim="year")
    out["year"] = obs.year.values
    return out


def _tercile_parametric_global(cv_predictions, obs, residuals, n, method):
    """Legacy parametric method with global PEV and boundaries."""
    t33 = obs.quantile(1 / 3, dim="year").drop_vars("quantile")
    t67 = obs.quantile(2 / 3, dim="year").drop_vars("quantile")

    pev = (residuals ** 2).mean("year")
    scale = np.sqrt(np.maximum((n - 2) / n * pev.values, 1e-24))

    results = []
    for yr in obs.year.values:
        pred = cv_predictions.sel(year=yr)
        loc = pred.values

        if method == "t":
            p_bn_vals = t_dist.cdf(t33.values, df=n, loc=loc, scale=scale)
            p_an_vals = t_dist.sf(t67.values, df=n, loc=loc, scale=scale)
        else:  # gaussian_pev
            p_bn_vals = norm.cdf(t33.values, loc=loc, scale=scale)
            p_an_vals = 1.0 - norm.cdf(t67.values, loc=loc, scale=scale)

        p_nn_vals = 1.0 - p_bn_vals - p_an_vals

        # Propagate NaN from obs or predictions
        nan_mask = np.isnan(t33.values) | np.isnan(loc) | np.isnan(scale)
        if nan_mask.any():
            p_bn_vals[nan_mask] = np.nan
            p_nn_vals[nan_mask] = np.nan
            p_an_vals[nan_mask] = np.nan

        spatial_dims = [d for d in pred.dims if d != "year"]
        spatial_coords = {k: v for k, v in pred.coords.items() if k != "year"}

        p_bn = xr.DataArray(p_bn_vals, dims=spatial_dims, coords=spatial_coords)
        p_an = xr.DataArray(p_an_vals, dims=spatial_dims, coords=spatial_coords)
        p_nn = xr.DataArray(p_nn_vals, dims=spatial_dims, coords=spatial_coords)

        tercile = xr.concat([p_bn, p_nn, p_an], dim="tercile")
        tercile["tercile"] = [0, 1, 2]
        results.append(tercile)

    out = xr.concat(results, dim="year")
    out["year"] = obs.year.values
    return out
