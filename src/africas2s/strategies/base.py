from abc import ABC, abstractmethod
import numpy as np
import xarray as xr


class StrategyBase(ABC):
    @abstractmethod
    def combine(self, forecasts: list, obs: xr.DataArray = None, *, weights=None, **kwargs) -> xr.DataArray:
        ...

    def fit(self, forecasts, obs, years=None, *, primary_metric="rpss") -> np.ndarray:
        """Return a per-member weight vector summing to 1.

        Default: uniform. Override in subclasses that derive weights from data.
        ``years`` is an optional iterable restricting the fit to a subset of
        the year axis (used by the nested-CV inner loop).
        """
        n = len(forecasts)
        return np.full(n, 1.0 / n)

    def self_shrinks(self) -> bool:
        """Strategies whose own fit already shrinks toward a prior return True
        so the outer-loop shrinkage isn't applied on top of theirs.
        """
        return False
