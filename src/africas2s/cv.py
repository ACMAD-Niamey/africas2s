"""Cross-validation schemes for hindcast skill evaluation.

Each scheme is a generator that yields `(train_years, test_year_or_years)`
tuples. `loyo` and `expanding` yield a single test year per fold; `lko` and
`blocked` yield a list of consecutive test years per fold.

Pick a scheme based on the temporal structure of your data:

- `loyo` — simple leave-one-year-out; window>1 also drops neighbours to
  break short-range autocorrelation.
- `lko` — leave k consecutive years out, sliding across the record. Good
  when you want every year scored but want to break autocorrelation across
  longer spans than `loyo` allows.
- `blocked` — non-overlapping contiguous blocks. Stricter than `lko`: every
  year is scored exactly once, and a `gap` parameter can fully isolate
  train and test sets across the block boundary.
- `expanding` — train on a growing prefix, predict the next year. Mirrors
  real-time operational use, where you only know the past.
"""

import warnings


def _validate_consecutive_years(years):
    """Return a list of consecutive integer years, or raise ValueError.

    Empty input is allowed (yields no folds downstream). Otherwise years
    must be integer-valued and strictly increase by 1. CV schemes here
    slice positionally and treat positions as calendar neighbours; gappy
    or non-monotonic input silently broke that contract before this guard.
    """
    out = []
    for y in years:
        y_int = int(y)
        if y_int != y:
            raise ValueError(f"years must be integer-valued; got {y!r}")
        out.append(y_int)
    for i in range(1, len(out)):
        if out[i] - out[i - 1] != 1:
            raise ValueError(
                f"years must be consecutive integers (gap of 1); "
                f"got {out[i - 1]} followed by {out[i]}"
            )
    return out


def loyo(years, window=1):
    """Leave-years-out CV. Yields (train_years, test_year) for each year.

    window: number of years to leave out (1=strict LOO, 3=target±1, etc.)
    """
    if window < 1:
        raise ValueError(f"window must be >= 1; got {window}")
    years = _validate_consecutive_years(years)
    hcw = (window - 1) // 2
    for i, test_year in enumerate(years):
        train_years = [y for j, y in enumerate(years) if abs(j - i) > hcw]
        yield train_years, test_year


def lko(years, k=3):
    """Leave-k-out (consecutive, sliding). Yields (train_years, [test_years]).

    A window of `k` consecutive years slides across the record. Each fold
    holds out the window from training and yields it as the test set.

    Note: `lko(k)` is NOT the same as `loyo(window=k)`. `loyo(window=k)`
    holds out k years centred on each target and yields a single test year;
    `lko(k)` holds out k years and yields all of them as test years.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    years = _validate_consecutive_years(years)
    n = len(years)
    if k > n:
        raise ValueError(f"k={k} exceeds number of years ({n}).")
    for start in range(n - k + 1):
        test_years = years[start:start + k]
        train_years = years[:start] + years[start + k:]
        yield train_years, test_years


def blocked(years, block_size=5, gap=0):
    """Blocked CV — non-overlapping contiguous blocks. Yields (train, [test]).

    Years are partitioned into blocks of `block_size`. Each fold uses one
    block as the test set; the train set is everything outside the block
    plus an optional `gap` of additional years on each side excluded to
    break autocorrelation across the boundary.

    A trailing partial block (if `len(years) % block_size != 0`) is dropped.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1; got {block_size}")
    if gap < 0:
        raise ValueError(f"gap must be >= 0; got {gap}")
    years = _validate_consecutive_years(years)
    n = len(years)
    n_blocks = n // block_size
    for b in range(n_blocks):
        test_start = b * block_size
        test_end = test_start + block_size
        test_years = years[test_start:test_end]
        train_lo = max(0, test_start - gap)
        train_hi = min(n, test_end + gap)
        train_years = years[:train_lo] + years[train_hi:]
        yield train_years, test_years


def expanding(years, min_train=10):
    """Expanding-window CV — simulates real-time operations.

    For each year `i` such that `i >= min_train`, train on `years[:i]` and
    predict `years[i]`. Yields (train_years, test_year) tuples.

    Warns if fewer than 5 evaluation years remain after honouring
    `min_train` — a common gotcha when hindcasts are short.
    """
    if min_train < 1:
        raise ValueError(f"min_train must be >= 1; got {min_train}")
    years = _validate_consecutive_years(years)
    n = len(years)
    n_eval = max(0, n - min_train)
    if n_eval < 5:
        warnings.warn(
            f"expanding(min_train={min_train}) on {n} years yields only "
            f"{n_eval} evaluation years; consider lowering min_train.",
            stacklevel=2,
        )
    for i in range(min_train, n):
        yield years[:i], years[i]


_REGISTRY = {
    "loyo": loyo,
    "lko": lko,
    "blocked": blocked,
    "expanding": expanding,
}


def get_cv(name):
    if name in _REGISTRY:
        return _REGISTRY[name]
    raise KeyError(f"Unknown CV scheme: {name}")
