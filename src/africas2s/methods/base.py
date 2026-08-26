import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import xarray as xr

class MethodBase(ABC):
    #: Methods whose training is expensive (DL methods: hours to days) set this
    #: True. ``downscale()`` then refuses to fit them inline and points the user
    #: at ``africas2s.training.train(...)`` + ``downscale(weights_path=...)``.
    #: Fast statistical methods leave it False and keep the unified fit/predict
    #: path. See §10.2 (#27).
    requires_training = False

    @abstractmethod
    def fit(self, hindcast, obs, **kwargs):
        ...

    @abstractmethod
    def predict(self, forecast, **kwargs):
        ...

    @property
    def is_trained(self) -> bool:
        """True once ``fit()`` has produced state.

        Uses the scikit-learn convention (mirrors ``check_is_fitted``): a
        method is fitted iff it has any public attribute whose name ends in a
        single trailing underscore (e.g. ``x_mean_``, ``obs_clim_``). Every
        method in this package names fit-produced state this way.
        """
        return any(
            k.endswith("_") and not k.endswith("__") for k in vars(self)
        )

    def save(self, path: str | Path) -> None:
        """Persist fitted state via pickle.

        Default pickles ``self.__dict__`` (so constructor params and fitted
        state both survive a reload). Override for non-picklable methods (e.g.
        DL methods holding a live torch model).
        """
        if not self.is_trained:
            raise RuntimeError(
                f"{type(self).__name__} is not fitted; nothing to save. "
                "Call fit() before save()."
            )
        with open(Path(path), "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, path: str | Path) -> "MethodBase":
        """Restore fitted state written by ``save``. Returns ``self``."""
        with open(Path(path), "rb") as f:
            self.__dict__.update(pickle.load(f))
        return self


class ProbabilisticMethodBase(MethodBase):
    """Base for methods that natively produce an ensemble or distribution.

    These methods can bypass the Gaussian-fit-to-deterministic-forecast tercile
    conversion: the tercile path takes the ensemble directly and counts members
    (see ``downscale(output_type="tercile")``).

    Subclasses implement :meth:`predict_distribution`. They inherit a default
    :meth:`predict` that returns the ensemble mean (a deterministic best guess
    for callers that don't need uncertainty). A subclass whose
    ``predict_distribution`` returns distributional *parameters* rather than an
    ensemble (e.g. EMOS returning mu/scale) must override ``predict`` itself.
    """

    @abstractmethod
    def predict_distribution(self, forecast, **kwargs):
        """Return the full predictive distribution for ``forecast``.

        Either an ensemble with a ``member`` dimension (e.g. CorrDiff draws,
        XGBoost-quantile members) or distributional parameters. The
        member-counting tercile path expects the ensemble form.
        """
        ...

    def predict(self, forecast, **kwargs):
        """Deterministic best guess: the mean over the ensemble ``member`` axis."""
        return self.predict_distribution(forecast, **kwargs).mean("member")
