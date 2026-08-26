"""Two-Alternative Forced Choice (2AFC) skill score.

Equivalent to the area under the ROC curve for continuous forecasts. For
every pair (i, j) of years, the score counts the forecast as concordant
when its rank order matches the obs rank order. Range [0, 1] with 0.5 = no
skill, 1.0 = perfect.
"""

import numpy as np
from .base import MetricBase
from ..registry import register_metric


def _afc_per_cell(forecast, obs):
    """1D 2AFC computation. Returns float or NaN."""
    f = np.asarray(forecast)
    o = np.asarray(obs)
    diffs_o = o[:, None] - o[None, :]
    diffs_f = f[:, None] - f[None, :]
    sign_o = np.sign(diffs_o)
    sign_f = np.sign(diffs_f)
    # Pairs where obs differ
    nonzero_o = sign_o != 0
    concordant = ((sign_o == sign_f) & nonzero_o & (sign_f != 0)).sum()
    discordant = ((sign_o == -sign_f) & nonzero_o & (sign_f != 0)).sum()
    ties = (nonzero_o & (sign_f == 0)).sum()
    total = concordant + discordant + ties
    if total == 0:
        return float("nan")
    # Half-credit-for-ties convention: a fully-tied forecast (constant) scores 0.5,
    # matching the issue's stated "constant forecast (no skill) ≈ 0.5" criterion.
    return float((concordant + 0.5 * ties) / total)


@register_metric("2afc")
class TwoAFCMetric(MetricBase):
    """2AFC = P(forecast ranks two random non-tied obs years correctly).

    Range [0, 1]; 0.5 = no skill, 1.0 = perfect ranking.
    """

    def compute(self, forecast, obs, spatial=False, **kwargs):
        if "member" in forecast.dims:
            forecast = forecast.mean("member")
        if "tercile" in forecast.dims:
            raise ValueError(
                "2afc requires a continuous-valued forecast; got a forecast "
                "with a 'tercile' dim (tercile probabilities). Pass the "
                "deterministic ensemble forecast instead."
            )

        import xarray as xr
        rho = xr.apply_ufunc(
            _afc_per_cell, forecast, obs,
            input_core_dims=[["year"], ["year"]],
            vectorize=True,
            output_dtypes=[float],
        )
        if spatial:
            return rho
        return float(rho.mean())
