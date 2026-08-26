"""§5 Flex forecast: exceedance probabilities at arbitrary thresholds.

PyCPT's Phase-6 capability. Fits a Gaussian predictive distribution per grid
cell — location = the deterministic MME forecast, scale = sqrt(prediction-error
variance) — and integrates the upper tail to give P(Y > threshold).

Percentile thresholds reference the *climatological* distribution (estimated
from ``obs``), not the forecast distribution: "probability of exceeding the
median" means the climatological median.

V1 is Gaussian-only. Gamma is deferred: many tropical-precip cells are
zero-inflated and a plain Gamma fits poorly there (see issue #20). The
deterministic-forecast-as-location / sqrt(PEV)-as-scale form is the regression
posterior under a Gaussian-likelihood assumption — documented, not hidden.
"""
from dataclasses import dataclass, field

import numpy as np
import xarray as xr
from scipy.stats import norm

# Floor on the predictive variance so a perfect-hindcast cell (PEV == 0) yields
# a degenerate step (exceedance 0/1) rather than a divide-by-zero in norm.cdf.
_TINY = 1e-12


@dataclass
class FlexForecastResult:
    """Per-grid-cell flex-forecast output.

    All fields are ``(lat, lon)`` DataArrays except ``metadata``. ``to_dict()``
    follows the SkillReport convention (dims/coords/values payloads) so the
    result round-trips to JSON.
    """

    exceedance_prob: xr.DataArray       # P(Y > threshold), in [0, 1]
    fcst_mu: xr.DataArray               # forecast distribution location
    fcst_scale: xr.DataArray            # forecast distribution scale (sqrt PEV)
    climo_mu: xr.DataArray              # climatological mean (from obs)
    climo_scale: xr.DataArray           # climatological std (from obs)
    transformed_threshold: xr.DataArray  # threshold expressed in data space
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        def _payload(da):
            if not isinstance(da, xr.DataArray):
                return da
            return {
                "dims": list(da.dims),
                "coords": {k: da.coords[k].values.tolist() for k in da.coords},
                "values": da.values.tolist(),
            }

        return {
            "exceedance_prob": _payload(self.exceedance_prob),
            "fcst_mu": _payload(self.fcst_mu),
            "fcst_scale": _payload(self.fcst_scale),
            "climo_mu": _payload(self.climo_mu),
            "climo_scale": _payload(self.climo_scale),
            "transformed_threshold": _payload(self.transformed_threshold),
            "metadata": dict(self.metadata),
        }


def flex_forecast(det_fcst, pev, obs, threshold, is_percentile=True,
                  distribution="gaussian"):
    """Exceedance probability P(Y > threshold) per grid cell.

    Parameters
    ----------
    det_fcst : xr.DataArray
        Deterministic MME forecast, dims ``(lat, lon)``. Forecast distribution
        location.
    pev : xr.DataArray
        Prediction-error variance, dims ``(lat, lon)`` (e.g.
        ``SeasonalMMEResult.pev``). Forecast distribution scale is ``sqrt(pev)``.
    obs : xr.DataArray
        Observations, dims ``(year, lat, lon)``, for the climatological
        reference distribution and percentile thresholds.
    threshold : float
        Absolute value (``is_percentile=False``) or a quantile in ``[0, 1]``
        (``is_percentile=True``) of the *climatological* distribution.
    is_percentile : bool
        If True, ``threshold`` is a climatological quantile; the returned
        ``transformed_threshold`` is that quantile evaluated per cell from obs.
    distribution : {"gaussian"}
        V1 supports ``"gaussian"`` only.

    Returns
    -------
    FlexForecastResult
    """
    if distribution != "gaussian":
        raise NotImplementedError(
            f"flex_forecast V1 supports distribution='gaussian' only; got "
            f"{distribution!r}. Gamma is deferred (zero-inflation; see §5)."
        )

    fcst_mu = det_fcst
    fcst_scale = np.sqrt(np.maximum(pev, _TINY))
    climo_mu = obs.mean("year")
    climo_scale = obs.std("year")

    if is_percentile:
        transformed_threshold = obs.quantile(threshold, dim="year")
        if "quantile" in transformed_threshold.coords:
            transformed_threshold = transformed_threshold.drop_vars("quantile")
    else:
        transformed_threshold = xr.full_like(climo_mu, float(threshold))

    # P(Y > t) = 1 - F(t); norm.cdf broadcasts over the (lat, lon) grid and
    # propagates NaN (e.g. ocean cells in a land-only obs grid).
    exceedance_prob = 1.0 - xr.apply_ufunc(
        norm.cdf, transformed_threshold, fcst_mu, fcst_scale,
    )

    return FlexForecastResult(
        exceedance_prob=exceedance_prob,
        fcst_mu=fcst_mu,
        fcst_scale=fcst_scale,
        climo_mu=climo_mu,
        climo_scale=climo_scale,
        transformed_threshold=transformed_threshold,
        metadata={
            "distribution": distribution,
            "threshold": float(threshold),
            "is_percentile": bool(is_percentile),
        },
    )
