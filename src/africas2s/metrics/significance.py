"""Significance testing and multiple-comparison control for skill statistics.

Two tools for deciding whether a discovered predictor–predictand relationship is more than the
best of a large search:

* :func:`permutation_test` — is an observed statistic beyond what label-shuffling produces?
* :func:`fdr` — Benjamini–Hochberg control of the false-discovery rate across many tests.

Both are statistic-agnostic. The default statistic is the leave-one-out correlation
(:func:`africas2s.metrics.cross_validation.loo_corr`), which is what makes the permutation test
default to a one-sided upper tail, but any callable ``statistic(x, y) -> float`` can be supplied.
"""
import numpy as np

from .cross_validation import loo_corr

_ALTERNATIVES = ("greater", "less", "two-sided")


def permutation_test(x, y, *, statistic=loo_corr, alternative="greater", n=5000, rng=None):
    """Permutation p-value for the association between ``x`` and ``y`` under a shuffled null.

    The predictor ``x`` is held fixed and the target ``y`` is shuffled ``n`` times; the p-value is
    the fraction of shuffles whose statistic is at least as extreme as the observed one, with the
    conventional +1 in both numerator and denominator so it is never exactly zero.

    Parameters
    ----------
    x, y : array-like
        Paired predictor and target series.
    statistic : callable, default :func:`~africas2s.metrics.cross_validation.loo_corr`
        ``statistic(x, y) -> float``. Non-finite values from a shuffle are treated as not-extreme.
    alternative : {"greater", "less", "two-sided"}, default "greater"
        ``"greater"`` (upper tail) is correct for a leave-one-out skill statistic, whose null is
        biased negative — a zero-signal predictor yields leave-one-out predictions that
        *anti*-correlate with the target, so a two-sided ``|stat|`` test would be invalid. A
        genuinely predictive ``x``, of either underlying sign, lands in the upper tail.
    n : int, default 5000
        Number of shuffles.
    rng : int | numpy.random.Generator | None
        Seed or generator for reproducibility. A single generator shared across many calls
        preserves one stable draw sequence.

    Returns
    -------
    (observed_statistic, p_value) : tuple of float
        ``p_value`` is NaN when the observed statistic is not finite.
    """
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative must be one of {_ALTERNATIVES}, got {alternative!r}")
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    observed = statistic(x, y)
    if not np.isfinite(observed):
        return observed, np.nan

    count = 1
    n = int(n)
    for _ in range(n):
        s = statistic(x, rng.permutation(y))          # draw is consumed even if s is non-finite
        if not np.isfinite(s):
            continue
        if alternative == "greater":
            extreme = s >= observed
        elif alternative == "less":
            extreme = s <= observed
        else:  # two-sided
            extreme = abs(s) >= abs(observed)
        if extreme:
            count += 1
    return observed, count / (n + 1)


def fdr(pvalues, *, method="bh"):
    """Benjamini–Hochberg q-values for an array of p-values.

    ``q_i`` is the smallest false-discovery rate at which test ``i`` is rejected. NaN p-values are
    excluded from the test count ``m`` and returned as NaN, so a ragged leaderboard can be passed
    through directly.

    Returns an array the shape of ``pvalues``.
    """
    if method != "bh":
        raise ValueError(f"only method='bh' (Benjamini–Hochberg) is implemented, got {method!r}")
    p = np.asarray(pvalues, dtype=float)
    pf = p.ravel()
    finite = np.isfinite(pf)
    m = int(finite.sum())
    q = np.full(pf.shape, np.nan)
    if m == 0:
        return q.reshape(p.shape)
    # step down from the largest finite p-value, enforcing a monotonic non-increasing q. The BH
    # rank advances only over finite entries (NaNs sort to the end and are skipped without
    # consuming a rank), so q is correct on a leaderboard padded with NaNs.
    order = np.argsort(np.where(finite, pf, 2.0))
    running_min = 1.0
    rank = m
    for i in order[::-1]:
        if not finite[i]:
            continue
        running_min = min(running_min, pf[i] * m / rank)
        q[i] = running_min
        rank -= 1
    return q.reshape(p.shape)
