"""Train / inference separation (§10.2, #27).

Deep-learning methods can take hours to days to train but only seconds to run
inference. Conflating the two inside ``downscale()`` would force an expensive
refit on every call. This package splits them:

- ``train(method_name, hindcast, obs, save_to=PATH)`` fits a method once and
  writes a checkpoint.
- ``downscale(..., weights_path=PATH)`` loads that checkpoint and runs inference
  only — no refit.

Statistical methods (CCA, BCSD, QM, ...) train in seconds, so they keep using
the unified ``downscale()`` fit+predict path. They can still be pre-trained via
``train()`` if you want to cache a fitted model, but they are not *required* to
be — only methods with ``requires_training = True`` (e.g. the §15 U-Net) are.
"""
from ..registry import get_method
from ..downscale import _METHOD_PARAMS


def train(method_name, hindcast, obs, save_to=None, *, verbose=True, **kwargs):
    """Fit ``method_name`` on ``(hindcast, obs)`` and optionally checkpoint it.

    Parameters
    ----------
    method_name : str
        A registered method name (e.g. ``"cca"``, ``"unet"``).
    hindcast, obs : xr.DataArray
        Training predictors and predictand.
    save_to : str | pathlib.Path | None
        If given, the fitted state is written here via ``MethodBase.save`` so a
        later ``downscale(..., weights_path=save_to)`` can run inference only.
    verbose : bool
        Print progress lines.
    **kwargs
        Forwarded to the method constructor (the recognised subset) and to
        ``fit()``.

    Returns
    -------
    MethodBase
        The fitted method instance.
    """
    method_cls = get_method(method_name)
    m = method_cls(**{k: v for k, v in kwargs.items() if k in _METHOD_PARAMS})
    if verbose:
        print(f"[africas2s.training] training {method_name}...")
    m.fit(hindcast, obs, **kwargs)
    if save_to is not None:
        m.save(save_to)
        if verbose:
            print(f"[africas2s.training] saved checkpoint -> {save_to}")
    return m
