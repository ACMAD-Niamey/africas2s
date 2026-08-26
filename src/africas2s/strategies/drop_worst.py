"""Drop-worst ensemble strategy: discard the N lowest-skill members, average the rest.

Reference: Weigel, Liniger & Appenzeller (2008), QJRMS.
"""

import numpy as np

from .base import StrategyBase
from ..registry import register_strategy
from .uniform import _as_array


def _resolve_scores(forecasts, scores_kwarg):
    """Pull a score per forecast, preferring an explicit kwarg over OptimizeResult."""
    if scores_kwarg is not None:
        if len(scores_kwarg) != len(forecasts):
            raise ValueError(
                f"scores has length {len(scores_kwarg)}, but {len(forecasts)} forecasts."
            )
        return list(scores_kwarg)
    scores = []
    for f in forecasts:
        if not hasattr(f, "score"):
            raise ValueError(
                "drop_worst needs a per-forecast score: pass `scores=[...]` "
                "or supply OptimizeResult objects with a `.score` attribute."
            )
        scores.append(f.score)
    return scores


def _per_member_skill(forecasts, obs, years, primary_metric):
    """Score each member individually on the given years using the metric registry.

    Used by drop_worst.fit and skill_weighted.fit to derive weights from data.
    """
    from ..skill import skill

    if years is not None:
        obs_y = obs.sel(year=list(years))
        forecasts_y = [_as_array(f).sel(year=list(years)) for f in forecasts]
    else:
        obs_y = obs
        forecasts_y = [_as_array(f) for f in forecasts]
    out = []
    for f in forecasts_y:
        report = skill(f, obs_y, metrics=[primary_metric])
        if primary_metric not in report.scores:
            raise ValueError(
                f"primary_metric={primary_metric!r} expands to "
                f"{sorted(report.scores)}; pick one of those leaf metrics instead."
            )
        out.append(float(report.scores[primary_metric]))
    return out


@register_strategy("drop_worst")
class DropWorstStrategy(StrategyBase):
    def fit(self, forecasts, obs, years=None, *, primary_metric="rpss", n_drop=1, scores=None, **kwargs):
        """Indicator weights for the top-(N - n_drop) members by per-member skill.

        Returns a (N,) array with `1 / (N - n_drop)` for kept members and `0`
        for dropped. If ``scores=`` is passed, use those instead of computing.
        """
        n = len(forecasts)
        if n_drop >= n:
            raise ValueError(
                f"n_drop={n_drop} would leave nothing to combine (have {n} forecasts)."
            )
        if scores is None:
            try:
                scores = _resolve_scores(forecasts, None)
            except ValueError:
                scores = _per_member_skill(forecasts, obs, years, primary_metric)
        order = sorted(range(n), key=lambda i: scores[i])
        kept = set(order[n_drop:])
        weights = np.zeros(n)
        for i in kept:
            weights[i] = 1.0 / len(kept)
        return weights

    def combine(self, forecasts, obs=None, *, weights=None, n_drop=1, scores=None, **kwargs):
        arrays = [_as_array(f) for f in forecasts]
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if len(weights) != len(arrays):
                raise ValueError(
                    f"weights has length {len(weights)}, but {len(arrays)} forecasts."
                )
            result = arrays[0] * float(weights[0])
            for arr, wi in zip(arrays[1:], weights[1:]):
                result = result + arr * float(wi)
            return result
        if n_drop >= len(forecasts):
            raise ValueError(
                f"n_drop={n_drop} would leave nothing to combine "
                f"(have {len(forecasts)} forecasts)."
            )
        scores = _resolve_scores(forecasts, scores)
        order = sorted(range(len(scores)), key=lambda i: scores[i])
        keep = sorted(order[n_drop:])
        kept_arrays = [arrays[i] for i in keep]
        return sum(kept_arrays) / len(kept_arrays)
