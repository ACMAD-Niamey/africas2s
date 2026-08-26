"""africas2s.mask_by_skill / dry_mask — forecast masking, lifted from ACMAD's combination.core."""
import numpy as np
import pytest
import xarray as xr

from africas2s import mask_by_skill, dry_mask


def _probs(nan=False):
    p = np.tile(np.array([0.2, 0.3, 0.5])[:, None, None], (1, 2, 2)).astype(float)
    return xr.DataArray(p, dims=("tercile", "lat", "lon"),
                        coords={"tercile": [0, 1, 2], "lat": [0.0, 1.0], "lon": [10.0, 11.0]})


def _skill(vals):
    return xr.DataArray(np.array(vals, float).reshape(2, 2), dims=("lat", "lon"),
                        coords={"lat": [0.0, 1.0], "lon": [10.0, 11.0]})


# ---- mask_by_skill: equivalence with ACMAD's apply_skill_mask -----------------------

def _acmad_apply_skill_mask(probs, skill, threshold):
    if threshold is None or threshold <= 0:
        return probs
    keep = skill.notnull() & (skill > threshold)
    return probs.where(keep)


def test_matches_acmad_apply_skill_mask():
    probs = _probs()
    skill = _skill([0.1, 0.4, np.nan, 0.6])
    for thr in (None, -1, 0, 0.3, 0.5):
        got = mask_by_skill(probs, skill, threshold=thr)
        ref = _acmad_apply_skill_mask(probs, skill, thr)
        assert got.equals(ref), f"threshold={thr}"


def test_blanks_low_skill_and_nan_skill():
    probs = _probs()
    skill = _skill([0.1, 0.4, np.nan, 0.6])
    out = mask_by_skill(probs, skill, threshold=0.3)
    assert bool(out.isel(lat=0, lon=0).isnull().all())     # 0.1 < 0.3 -> blanked
    assert bool(out.isel(lat=1, lon=0).isnull().all())     # NaN skill -> blanked
    assert not bool(out.isel(lat=0, lon=1).isnull().any())  # 0.4 > 0.3 -> kept


def test_threshold_off_is_noop():
    probs = _probs()
    skill = _skill([0.1, 0.4, 0.2, 0.6])
    assert mask_by_skill(probs, skill, threshold=None).equals(probs)
    assert mask_by_skill(probs, skill, threshold=0).equals(probs)


def test_keep_below():
    probs = _probs()
    skill = _skill([0.1, 0.4, 0.2, 0.6])
    out = mask_by_skill(probs, skill, threshold=0.3, keep="below")
    assert not bool(out.isel(lat=0, lon=0).isnull().any())  # 0.1 < 0.3 kept
    assert bool(out.isel(lat=0, lon=1).isnull().all())       # 0.4 > 0.3 blanked


def test_rejects_bad_keep():
    with pytest.raises(ValueError, match="keep must be"):
        mask_by_skill(_probs(), _skill([1, 1, 1, 1]), threshold=0.3, keep="sideways")


# ---- dry_mask ------------------------------------------------------------------------

def test_dry_mask_thresholds_total():
    clim = xr.DataArray(np.array([[50.0, 150.0], [80.0, 200.0]]), dims=("lat", "lon"),
                        coords={"lat": [0.0, 1.0], "lon": [10.0, 11.0]})
    m = dry_mask(clim, threshold=100.0)
    assert m.values.tolist() == [[True, False], [True, False]]


def test_dry_mask_regrids_to_like():
    clim = xr.DataArray(np.array([[10.0, 10.0], [200.0, 200.0]]), dims=("lat", "lon"),
                        coords={"lat": [0.0, 2.0], "lon": [10.0, 12.0]})
    like = xr.DataArray(np.zeros((3, 3)), dims=("lat", "lon"),
                        coords={"lat": [0.0, 1.0, 2.0], "lon": [10.0, 11.0, 12.0]})
    m = dry_mask(clim, threshold=100.0, like=like)
    assert m.sizes == {"lat": 3, "lon": 3}
    assert bool(m.sel(lat=0.0, lon=10.0)) and not bool(m.sel(lat=2.0, lon=12.0))
