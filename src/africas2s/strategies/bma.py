"""Bayesian Model Averaging ensemble strategy.

Reference: Raftery, Gneiting, Balabdaoui & Polakowski (2005), MWR — "Using
Bayesian Model Averaging to Calibrate Forecast Ensembles." Treats the
combined forecast as a mixture of Gaussians (one per ensemble member),
where each member contributes a Gaussian centred on its own prediction.
The mixture weights and per-member variances are learned from a hindcast
record using EM. The combined forecast is the weight-averaged member mean.

This implementation uses **spatially-uniform** weights — one weight per
member across the whole domain. Per-grid-cell BMA (one EM fit per cell) is
the more accurate option for large domains but is far slower; the uniform
variant is the standard practitioner shortcut and is what most papers use
when comparing strategies.
"""

import numpy as np

from .base import StrategyBase
from ..registry import register_strategy
from .uniform import _as_array


_LOG_2PI = float(np.log(2.0 * np.pi))
_SIGMA2_FLOOR = 1e-12


def _logsumexp(a, axis):
    """Numerically stable logsumexp along an axis (avoids the scipy import)."""
    a_max = np.max(a, axis=axis, keepdims=True)
    out = np.log(np.sum(np.exp(a - a_max), axis=axis))
    return out + np.squeeze(a_max, axis=axis)


@register_strategy("bma")
class BMAStrategy(StrategyBase):
    """BMA via EM. Fit weights/variances on hindcasts, apply to forecasts."""

    def __init__(self, max_iter=200, tol=1e-7):
        self.max_iter = max_iter
        self.tol = tol

    def _fit_em(self, hindcasts, obs):
        """Fit BMA weights and variances by EM.

        Parameters
        ----------
        hindcasts : list of xr.DataArray
            One per member, each shaped like obs (year, lat, lon).
        obs : xr.DataArray
            (year, lat, lon).

        Returns
        -------
        weights : ndarray of shape (n_members,)
        sigma2  : ndarray of shape (n_members,)
        """
        K = len(hindcasts)
        H = np.stack([np.asarray(h.values).ravel() for h in hindcasts])  # (K, N)
        O = np.asarray(obs.values).ravel()
        # Drop samples where any member or obs is NaN.
        mask = np.isfinite(O) & np.all(np.isfinite(H), axis=0)
        H = H[:, mask]
        O = O[mask]
        if H.shape[1] == 0:
            raise ValueError("BMA fit needs at least one finite (member, obs) sample.")

        residuals = H - O[None, :]
        sigma2 = residuals.var(axis=1, ddof=0).clip(min=_SIGMA2_FLOOR)
        w = np.full(K, 1.0 / K)

        for _ in range(self.max_iter):
            # E-step: posterior responsibility z[k, n].
            log_pdf = -0.5 * (_LOG_2PI + np.log(sigma2)[:, None]
                              + residuals ** 2 / sigma2[:, None])
            log_unnorm = np.log(np.maximum(w, 1e-300))[:, None] + log_pdf
            log_z = log_unnorm - _logsumexp(log_unnorm, axis=0)[None, :]
            z = np.exp(log_z)

            # M-step
            new_w = z.mean(axis=1)
            denom = z.sum(axis=1)
            denom = np.where(denom > 0, denom, 1.0)
            new_sigma2 = ((z * residuals ** 2).sum(axis=1) / denom).clip(min=_SIGMA2_FLOOR)

            if np.max(np.abs(new_w - w)) < self.tol:
                w, sigma2 = new_w, new_sigma2
                break
            w, sigma2 = new_w, new_sigma2

        # Numerical hygiene: re-normalize.
        w = w / w.sum()
        return w, sigma2

    def fit(self, forecasts, obs, years=None, *, primary_metric="rpss", **kwargs):
        """BMA weights via EM. Restricts to ``years`` if provided."""
        if years is not None:
            obs_y = obs.sel(year=list(years))
            hindcasts_y = [_as_array(f).sel(year=list(years)) for f in forecasts]
        else:
            obs_y = obs
            hindcasts_y = [_as_array(f) for f in forecasts]
        w, _ = self._fit_em(hindcasts_y, obs_y)
        return w

    def self_shrinks(self):
        return True

    def combine(self, forecasts, obs=None, *, hindcasts=None, weights=None, **kwargs):
        arrays = [_as_array(f) for f in forecasts]
        if weights is None:
            if hindcasts is None or obs is None:
                raise ValueError(
                    "bma requires `hindcasts=` and `obs=` to fit weights, "
                    "or pass `weights=` directly."
                )
            if len(hindcasts) != len(forecasts):
                raise ValueError(
                    f"hindcasts has {len(hindcasts)} entries, "
                    f"but {len(forecasts)} forecasts."
                )
            weights, _ = self._fit_em(hindcasts, obs)
        else:
            weights = np.asarray(weights, dtype=float)
            if weights.sum() <= 0:
                raise ValueError("weights must have positive sum.")
            weights = weights / weights.sum()

        result = arrays[0] * float(weights[0])
        for arr, wi in zip(arrays[1:], weights[1:]):
            result = result + arr * float(wi)
        return result
