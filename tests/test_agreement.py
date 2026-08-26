"""Agreement tests: AfricaS2S vs PyCPT reference values.

Run:  pytest africas2s/tests/test_agreement.py -m agreement -v
Requires: CDS credentials (~/.cdsapirc), network access
"""
import sys
from pathlib import Path
import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import africas2s
from africas2s.cv import loyo
from africas2s.registry import get_method
from africas2s.tercile import to_tercile_cv
from reproduce import fetch_obs, fetch_gcm, DEFAULTS

YEARS = list(range(DEFAULTS["years"][0], DEFAULTS["years"][1] + 1))

PYCPT_PEARSON, PYCPT_RPSS = +0.122, +0.130
X_EOF, Y_EOF, CCA_MODES = 5, 4, 3


@pytest.fixture(scope="module")
def data():
    cfg = dict(DEFAULTS)
    return fetch_obs(cfg).sel(year=YEARS), fetch_gcm(cfg).sel(year=YEARS)


@pytest.fixture(scope="module")
def cv_results(data):
    obs, gcm = data
    cca_kw = dict(x_eof_modes=X_EOF, y_eof_modes=Y_EOF, cca_modes=CCA_MODES)
    preds, leverages = [], []
    for train_yrs, test_yr in loyo(YEARS, window=5):
        m = get_method("cca")(**cca_kw)
        m.fit(gcm.sel(year=train_yrs), obs.sel(year=train_yrs))
        forecast = gcm.sel(year=[test_yr]).isel(year=0, drop=True)
        preds.append(m.predict(forecast).mean("member"))
        leverages.append(m.leverage(forecast))
    cv = xr.concat(preds, dim="year")
    cv["year"] = YEARS
    return cv, np.array(leverages)


@pytest.mark.agreement
def test_pearson_r_meets_pycpt(data, cv_results):
    obs, _ = data
    cv, _ = cv_results
    r = float(africas2s.skill(cv, obs, metrics=["pearson_r"], spatial=True).spatial["pearson_r"].mean())
    assert r >= PYCPT_PEARSON - 0.05, f"Pearson r {r:+.3f} below PyCPT {PYCPT_PEARSON:+.3f} - 0.05"


@pytest.mark.agreement
def test_rpss_not_catastrophic(data, cv_results):
    obs, _ = data
    cv, _ = cv_results
    rpss = float(africas2s.skill(
        to_tercile_cv(cv, obs, method="bootstrap"), obs, metrics=["rpss"], spatial=True
    ).spatial["rpss"].mean())
    assert rpss > -0.5, f"RPSS {rpss:+.3f} is unreasonably negative"


@pytest.mark.agreement
def test_tercile_probs_sum_to_one(data, cv_results):
    obs, _ = data
    cv, leverages = cv_results
    for method in ("cpt", "bootstrap", "gaussian_loo", "t"):
        kw = dict(leverages=leverages, n_modes=X_EOF) if method == "cpt" else {}
        tercile = to_tercile_cv(cv, obs, method=method, **kw)
        total = tercile.sum("tercile", skipna=False)
        valid = ~np.isnan(total.values)
        assert valid.any(), f"method={method}: all cells are NaN"
        np.testing.assert_allclose(total.values[valid], 1.0, atol=1e-6, err_msg=f"method={method}")


@pytest.mark.agreement
def test_rpss_loo_bounded(data, cv_results):
    """RPSS with CPT-matching options: LOO boundaries + bounded formula."""
    obs, _ = data
    cv, leverages = cv_results
    tercile = to_tercile_cv(cv, obs, method="cpt", leverages=leverages, n_modes=X_EOF)
    rpss = float(africas2s.skill(
        tercile, obs, metrics=["rpss"], spatial=True,
        loo_boundaries=True, bounded=True, cv_window=5,
    ).spatial["rpss"].mean())
    assert rpss > -1.0, f"Bounded RPSS {rpss:+.3f} is out of range"
