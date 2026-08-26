"""Closed-form leave-one-out cross-validation for simple linear regression.

For a simple linear regression ``y ~ x`` the leave-one-out prediction of each point has a closed
form through the leverage ``h_i = 1/n + (x_i - x̄)² / Sxx`` (Allen's PRESS identity), so an
n-fold cross-validation costs a single full fit rather than n refits. That makes leave-one-out
skill cheap enough to permute thousands of times — which is exactly what significance testing over
a large predictor search needs (see :mod:`africas2s.metrics.significance`).

These are verification helpers on paired predictor/predictand *series*, not forecast-vs-obs metrics,
so they are plain functions rather than registered :class:`~africas2s.metrics.base.MetricBase`
scores.
"""
import numpy as np


def _as_1d(a):
    return np.asarray(a, dtype=float).ravel()


def leverage(x):
    """Leverage ``h_i`` of each point in a simple linear regression on ``x``.

    Depends only on the predictor, so it can be computed once and reused across many shuffles of
    the target. Returns NaNs when ``x`` has no variance (a degenerate fit).
    """
    x = _as_1d(x)
    n = x.size
    xbar = x.mean()
    sxx = float(((x - xbar) ** 2).sum())
    if sxx == 0.0:
        return np.full(n, np.nan)
    return 1.0 / n + (x - xbar) ** 2 / sxx


def loo_predict(x, y):
    """Leave-one-out predictions of ``y`` from a simple linear regression ``y ~ x``, closed-form.

    Returns an array the length of ``y``; every entry is the prediction for that point from a fit
    that excluded it. All-NaN when ``x`` has no variance.
    """
    x = _as_1d(x)
    y = _as_1d(y)
    n = x.size
    xbar = x.mean()
    sxx = float(((x - xbar) ** 2).sum())
    if sxx == 0.0:
        return np.full(n, np.nan)
    h = 1.0 / n + (x - xbar) ** 2 / sxx
    slope = float(((x - xbar) * (y - y.mean())).sum() / sxx)
    intercept = float(y.mean() - slope * xbar)
    resid = y - (intercept + slope * x)          # in-sample residual
    return y - resid / (1.0 - h)                 # PRESS: LOO residual = e_i / (1 - h_i)


def loo_corr(x, y, *, min_finite=5):
    """Correlation between the leave-one-out predictions of ``y ~ x`` and ``y``.

    A cross-validated measure of how well ``x`` predicts ``y`` out of sample: 1 is perfect,
    ~0 is no skill, and it is *negative* for a predictor no better than chance (leave-one-out
    predictions of a near-zero relationship anti-correlate with the target — the property that
    dictates a one-sided upper-tail permutation test, see :func:`~africas2s.metrics.significance.permutation_test`).

    Returns NaN for a degenerate fit, fewer than ``min_finite`` finite pairs, or a constant
    prediction.
    """
    y = _as_1d(y)
    pred = loo_predict(x, y)
    ok = np.isfinite(pred) & np.isfinite(y)
    if ok.sum() < min_finite or np.std(pred[ok]) == 0.0:
        return np.nan
    return float(np.corrcoef(pred[ok], y[ok])[0, 1])
