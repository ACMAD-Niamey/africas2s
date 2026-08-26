import numpy as np

from .base import StrategyBase
from ..registry import register_strategy


def _as_array(f):
    return f.forecast if hasattr(f, "forecast") else f


@register_strategy("uniform")
class UniformStrategy(StrategyBase):
    def combine(self, forecasts, obs=None, *, weights=None, **kwargs):
        arrays = [_as_array(f) for f in forecasts]
        if weights is None:
            return sum(arrays) / len(arrays)
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(arrays):
            raise ValueError(
                f"weights has length {len(weights)}, but {len(arrays)} forecasts."
            )
        result = arrays[0] * float(weights[0])
        for arr, wi in zip(arrays[1:], weights[1:]):
            result = result + arr * float(wi)
        return result
