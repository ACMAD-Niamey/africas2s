import numpy as np
import pytest
import xarray as xr


# ===================================================================
# 7. Cross-validation
# ===================================================================

def test_loyo_yields_correct_folds():
    from africas2s.cv import loyo
    years = list(range(2000, 2010))
    folds = list(loyo(years))
    assert len(folds) == 10
    for train, test in folds:
        assert len(train) == 9
        assert test not in train


def test_loyo_no_leakage():
    from africas2s.cv import loyo
    years = list(range(2000, 2010))
    test_years_seen = []
    for train, test in loyo(years):
        assert test not in train
        test_years_seen.append(test)
    assert sorted(test_years_seen) == years


def test_lko_sliding_consecutive():
    """Leave-k-out yields sliding windows of k consecutive test years."""
    from africas2s.cv import lko
    years = list(range(2000, 2010))
    folds = list(lko(years, k=3))
    # Sliding window: positions 0..7 inclusive → 8 folds
    assert len(folds) == 8
    for train, test in folds:
        assert isinstance(test, list)
        assert len(test) == 3
        # Test years are consecutive
        assert test == sorted(test)
        for i in range(1, len(test)):
            assert test[i] - test[i - 1] == 1
        # No leakage
        assert all(t not in train for t in test)
        assert len(train) == len(years) - 3


def test_lko_default_k():
    from africas2s.cv import lko
    years = list(range(2000, 2010))
    folds = list(lko(years))  # default k=3
    assert all(len(test) == 3 for _, test in folds)


def test_lko_k_equals_n_yields_one_fold():
    from africas2s.cv import lko
    years = list(range(2000, 2005))
    folds = list(lko(years, k=5))
    assert len(folds) == 1
    train, test = folds[0]
    assert train == []
    assert test == years


def test_blocked_partitions_into_contiguous_blocks():
    """Blocked CV partitions years into non-overlapping contiguous blocks."""
    from africas2s.cv import blocked
    years = list(range(2000, 2010))
    folds = list(blocked(years, block_size=5))
    assert len(folds) == 2
    test_years_seen = []
    for train, test in folds:
        assert isinstance(test, list)
        assert len(test) == 5
        # Block is contiguous
        assert test == sorted(test)
        for i in range(1, len(test)):
            assert test[i] - test[i - 1] == 1
        # No leakage
        assert all(t not in train for t in test)
        test_years_seen.extend(test)
    # Each year appears exactly once across the partition
    assert sorted(test_years_seen) == years


def test_blocked_with_gap_excludes_neighbours():
    """A nonzero gap removes years adjacent to the test block from the train set."""
    from africas2s.cv import blocked
    years = list(range(2000, 2010))
    folds = list(blocked(years, block_size=2, gap=1))
    # First fold: test=[2000, 2001], train must not include 2002 (gap=1)
    train, test = folds[0]
    assert test == [2000, 2001]
    assert 2002 not in train
    assert 2003 in train  # outside the gap
    # Middle fold: test=[2004, 2005], train excludes 2003 and 2006
    train, test = folds[2]
    assert test == [2004, 2005]
    assert 2003 not in train
    assert 2006 not in train
    assert 2002 in train and 2007 in train


def test_blocked_drops_partial_trailing_block():
    """If years don't divide evenly, the trailing partial block is dropped."""
    from africas2s.cv import blocked
    years = list(range(2000, 2008))  # 8 years, block_size=3 → 2 full blocks of 3
    folds = list(blocked(years, block_size=3))
    assert len(folds) == 2
    test_years = [yr for _, test in folds for yr in test]
    assert 2006 not in test_years  # part of dropped trailing block
    assert 2007 not in test_years


def test_expanding_simulates_realtime():
    """Expanding window: train on years[:i], test year i, for i >= min_train."""
    from africas2s.cv import expanding
    years = list(range(2000, 2010))
    folds = list(expanding(years, min_train=4))
    # i ranges from 4..9 → 6 folds
    assert len(folds) == 6
    for train, test in folds:
        assert isinstance(test, int) or not isinstance(test, list)
        # train is strictly the prefix before test
        assert all(yr < test for yr in train)
        assert len(train) >= 4


def test_expanding_short_hindcast_warns():
    """Expanding with too few eval years should warn (issue pitfall)."""
    import warnings
    from africas2s.cv import expanding
    years = list(range(2000, 2010))
    # min_train=8 → only 2 evaluation years
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        list(expanding(years, min_train=8))
    assert any("evaluation" in str(w.message).lower() for w in caught), (
        f"expected warning about few evaluation years, got: {[str(w.message) for w in caught]}"
    )


def test_get_cv_registers_all_schemes():
    from africas2s.cv import get_cv, loyo, lko, blocked, expanding
    assert get_cv("loyo") is loyo
    assert get_cv("lko") is lko
    assert get_cv("blocked") is blocked
    assert get_cv("expanding") is expanding


def _run_cv_pipeline(scheme, scheme_kwargs, gcm, obs):
    """Helper: run an end-to-end CV pipeline using the given scheme.

    Returns (cv_predictions, cv_obs) aligned by year, so a downstream
    `to_tercile_cv()` + `skill()` call can score the result.
    """
    from africas2s.methods.cca import CCAMethod
    years = list(gcm.year.values)
    preds = []
    obs_pieces = []
    for fold in scheme(years, **scheme_kwargs):
        train_years, test = fold
        test_list = test if isinstance(test, list) else [test]
        m = CCAMethod(n_modes=2)
        m.fit(gcm.sel(year=train_years), obs.sel(year=train_years))
        for test_yr in test_list:
            forecast = gcm.sel(year=test_yr)
            pred = m.predict(forecast).mean("member")
            preds.append(pred.expand_dims(year=[test_yr]))
            obs_pieces.append(obs.sel(year=[test_yr]))
    cv_pred = xr.concat(preds, dim="year").sortby("year")
    cv_obs = xr.concat(obs_pieces, dim="year").sortby("year")
    return cv_pred, cv_obs


def test_blocked_cv_pipeline_end_to_end(synthetic_gcm_hindcast, synthetic_obs):
    """Integration: full CV pipeline with `blocked` CV → tercile → skill."""
    import africas2s
    from africas2s.cv import blocked
    from africas2s.tercile import to_tercile_cv
    cv_pred, cv_obs = _run_cv_pipeline(
        blocked, {"block_size": 2}, synthetic_gcm_hindcast, synthetic_obs,
    )
    # Blocked with block_size=2 on 10 years → 5 folds, every year scored once.
    assert len(cv_pred.year) == len(synthetic_gcm_hindcast.year)
    tercile = to_tercile_cv(cv_pred, cv_obs, method="bootstrap")
    np.testing.assert_allclose(tercile.sum("tercile").values, 1.0, atol=1e-6)
    rpss = float(africas2s.skill(tercile, cv_obs, metrics=["rpss"]).scores["rpss"])
    assert -1.5 < rpss < 1.0
    assert not np.isnan(rpss)


def test_expanding_cv_pipeline_end_to_end(synthetic_gcm_hindcast, synthetic_obs):
    """Integration: full CV pipeline with `expanding` window → tercile → skill."""
    import warnings
    import africas2s
    from africas2s.cv import expanding
    from africas2s.tercile import to_tercile_cv
    # min_train=4 → 6 evaluation years (suppressing the short-hindcast warning).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_pred, cv_obs = _run_cv_pipeline(
            expanding, {"min_train": 4}, synthetic_gcm_hindcast, synthetic_obs,
        )
    # Expanding only scores years past the min_train threshold.
    assert len(cv_pred.year) == 6
    tercile = to_tercile_cv(cv_pred, cv_obs, method="bootstrap")
    np.testing.assert_allclose(tercile.sum("tercile").values, 1.0, atol=1e-6)
    rpss = float(africas2s.skill(tercile, cv_obs, metrics=["rpss"]).scores["rpss"])
    assert -1.5 < rpss < 1.0
    assert not np.isnan(rpss)


# ===================================================================
# 7b. Input validation (regression guards against silent CV bugs)
# ===================================================================

def test_loyo_rejects_window_zero():
    """window=0 used to silently leak the test year into train:
    hcw = (0-1)//2 = -1 and abs(j-i) > -1 is always true, so the
    position filter became a no-op and the test_year stayed in train.
    """
    from africas2s.cv import loyo
    with pytest.raises(ValueError, match="window"):
        list(loyo([2000, 2001, 2002], window=0))


def test_lko_rejects_k_zero():
    """k=0 used to silently yield n+1 folds with empty test sets — no
    crash, no warning, but every fold's test list was [] so downstream
    pipelines silently scored nothing."""
    from africas2s.cv import lko
    with pytest.raises(ValueError, match="k"):
        list(lko([2000, 2001, 2002], k=0))


def test_blocked_rejects_negative_gap():
    """A negative gap inverted the slice math and put test years back
    into train — a silent leakage bug. With test=[2000, 2001], gap=-1
    gave train_lo=1, train_hi=1, so train = years[:1] + years[1:] = all years.
    """
    from africas2s.cv import blocked
    with pytest.raises(ValueError, match="gap"):
        list(blocked([2000, 2001, 2002, 2003], block_size=2, gap=-1))


def test_cv_rejects_non_consecutive_years():
    """CV schemes slice positionally and treat positions as calendar
    neighbours. Gappy input like [2000, 2001, 2005, 2006] silently broke
    that contract — lko's "k consecutive years" would yield [2001, 2005]
    as a test fold, claiming a contiguous window that wasn't contiguous
    in calendar time.
    """
    from africas2s.cv import loyo, lko, blocked, expanding
    gappy = [2000, 2001, 2005, 2006]
    for fn, kwargs in [
        (loyo, {}),
        (lko, {"k": 2}),
        (blocked, {"block_size": 2}),
        (expanding, {"min_train": 2}),
    ]:
        with pytest.raises(ValueError, match="consecutive"):
            list(fn(gappy, **kwargs))


def test_disciplined_to_tercile_cv_pipeline_end_to_end(synthetic_gcm_hindcast, synthetic_obs):
    """Integration for §6.5 default flip: a real LOYO+CCA hindcast scored via
    `to_tercile_cv()` (default cpt_boundaries=True) yields a valid RPSS.
    """
    import africas2s
    from africas2s.cv import loyo
    from africas2s.tercile import to_tercile_cv
    cv_pred, cv_obs = _run_cv_pipeline(
        loyo, {}, synthetic_gcm_hindcast, synthetic_obs,
    )
    # Use the cpt method to actually exercise the cpt_boundaries=True default;
    # leverages of zero are fine for this skill-validity check.
    n = len(cv_obs.year)
    leverages = np.zeros(n)
    tercile = to_tercile_cv(cv_pred, cv_obs, method="cpt", leverages=leverages)
    np.testing.assert_allclose(tercile.sum("tercile").values, 1.0, atol=1e-6)
    rpss = float(africas2s.skill(tercile, cv_obs, metrics=["rpss"]).scores["rpss"])
    assert -1.5 < rpss < 1.0
    assert not np.isnan(rpss)
