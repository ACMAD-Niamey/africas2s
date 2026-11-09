"""Parity of ``engine="cccma"`` with ECCC/CCCma's ``s2d_calibration`` package as
deployed at ICPAC (gcm_calibration_ICPAC v0.1.0-icpac).

The fixtures under ``tests/data/cccma/`` are the reference package's own inputs and
outputs (``MakeCalib`` + ``MakeFcst`` run on synthetic hindcasts; regenerate with
``scripts/reproduce_cccma.py``). Three cases: single-model precipitation (gamma
transform), two models with 6 and 4 members (NaN-padded ensemble), and single-model
temperature (no transform).

How strict the comparison is depends on where it runs:

* On the platform the fixture was produced on (recorded in its ``__meta__``; currently
  macOS/arm64) every array must match *exactly* — same bytes, same dtype — with one
  exception: the reference evaluates its lazy dask expressions once per consumer, and
  the summation order of its cyclic seasonal filter for season index 0 depends on dask's
  task placement (a transposed block that crosses a worker boundary is pickled into a
  contiguous copy, which turns numpy's pairwise sum into a sequential one). Within one
  reference run the saved smoothed field and the same field as consumed downstream can
  therefore differ by an ulp in season 0, and two runs of the reference on identical
  inputs differ there too (observed: SCGF 5 cells, PXGF-TR 25 cells). The engine uses
  one fixed order (the distributed run's sequential one), so smoothed-derived fields are
  compared exactly in seasons 1..11 and to 1e-13 relative in season 0.
* Elsewhere (CI runs x86 Linux) ``exp``, ``erf`` and the normal/gamma CDFs come from a
  different libm and different SIMD kernels and differ by an ulp, which propagates
  through the whole pipeline; the comparison is then ``rtol=1e-9`` (``1e-7`` for the
  in-sample CRPSS, which amplifies it). The reference itself would differ by the same
  amount on such a platform.
"""
import json
import pathlib
import platform

import numpy as np
import pytest
import xarray as xr

import africas2s
from africas2s.methods import cccma_calibration as cc

DATA = pathlib.Path(__file__).parent / "data" / "cccma"
CASES = ["precip_1m", "precip_2m", "temp_1m"]
_PUBLIC = {"month": "season", "ensemble_member": "member", "latitude": "lat", "longitude": "lon"}
_VARIANTS = [("none", "U"), ("constant", "C"), ("independent", "A"), ("smoothed", "F")]
_SIGMA = {"constant": "constant", "independent": None, "smoothed": 2}
_HERE = (platform.system(), platform.machine())


class Case:
    def __init__(self, name):
        z = np.load(DATA / f"{name}.npz", allow_pickle=False)
        self.z = z
        self.meta = json.loads(str(z["__meta__"]))
        self.cfg = self.meta["config"]
        made_on = self.meta.get("platform", {})
        self.exact = (made_on.get("system"), made_on.get("machine")) == _HERE
        self.init_month = self.cfg["fcst_initialization_month"]
        self.hind = self._public(self.ref("X"))          # (lat, lon, season, year, model, member)
        self.obs = self._public(self.ref("Y"))
        fc = self._public(self.ref("F")).expand_dims(season=self.hind["season"].values)
        self.fc = fc.where(fc["season"] == self.init_month)   # only the issued month is real

    def ref(self, key):
        dims = self.meta[key]
        coords = {d: self.meta["coords"][d] for d in dims if d in self.meta["coords"]}
        return xr.DataArray(self.z[key], dims=dims, coords=coords)

    @staticmethod
    def _public(da):
        return da.rename({k: v for k, v in _PUBLIC.items() if k in da.dims})

    def fit_kwargs(self):
        g = self.cfg["Gaussian_width"]
        return dict(gamma_transform=self.cfg["gamma_transform"], coefficient_width=g["calib"],
                    obs_variance_width=g["obsVar"], fcst_variance_width=g["fcstVar"],
                    golden_tolerance=self.cfg["Golden_tolerance"], small=self.cfg["SMALL"],
                    small_ic=self.cfg["SMALLIC"])

    def model_dict(self):
        """ereg-style ``{model: (hindcast, forecast)}`` with each model's real members."""
        out = {}
        for m in self.hind["model"].values:
            h = self.hind.sel(model=m).dropna("member", how="all")
            f = self.fc.sel(model=m).isel(member=slice(0, h.sizes["member"]))
            out[str(m)] = (h, f)
        return out

    def check(self, ours, ref, what, season_dim=None, rtol=1e-9):
        """Exact on the fixture's platform (season-0 rule for smoothed fields), ``rtol``
        elsewhere."""
        o = np.asarray(ours)
        r = np.asarray(ref)
        assert o.shape == r.shape, f"{what}: shape {o.shape} vs reference {r.shape}"
        assert o.dtype == r.dtype, f"{what}: dtype {o.dtype} vs reference {r.dtype}"
        same = (o == r) | (np.isnan(o) & np.isnan(r))
        if same.all():
            return
        if not self.exact:
            np.testing.assert_allclose(o, r, rtol=rtol, atol=0, equal_nan=True,
                                       err_msg=f"{what}: beyond the cross-platform ulp tolerance")
            return
        bad = np.argwhere(~same)
        if season_dim is None:
            raise AssertionError(f"{what}: {len(bad)} values differ from the reference bytes")
        seasons = set(bad[:, season_dim].tolist())
        assert seasons == {0}, (
            f"{what}: {len(bad)} values differ outside season 0 (seasons {sorted(seasons)})")
        np.testing.assert_allclose(o[~same], r[~same], rtol=1e-13, atol=0,
                                   err_msg=f"{what}: season-0 values beyond the reference's own wobble")


@pytest.fixture(scope="module", params=CASES)
def case(request):
    return Case(request.param)


@pytest.fixture(scope="module")
def fitted(case):
    return cc.fit(case.hind, case.obs, **case.fit_kwargs())


def _season_axis(da):
    return da.dims.index("month") if "month" in da.dims else None


# ── fit: coefficients and in-sample probabilities ─────────────────────────────

def test_hindcast_statistics_match_reference(case, fitted):
    for key, ours in [("XAVG", fitted.xavg), ("XANOU", fitted.xanou), ("YSDT", fitted.diagnostics["YSDT"]),
                      ("XVARAU", fitted.diagnostics["XVARAU"])]:
        ref = case.ref(key)
        case.check(ours.transpose(*ref.dims).values, ref.values, key)
    for key, ours in [("YSDF", fitted.diagnostics["YSDF"]), ("XVARAS", fitted.diagnostics["XVARAS"]),
                      ("XVARTS", fitted.diagnostics["XVARTS"]), ("XSDEMS", fitted.diagnostics["XSDEMS"])]:
        ref = case.ref(key)
        case.check(ours.transpose(*ref.dims).values, ref.values, key, _season_axis(ref))


def test_gamma_parameters_match_reference(case, fitted):
    if not case.cfg["gamma_transform"]:
        pytest.skip("no gamma transform in this case")
    ref = case.ref("YPARA")
    case.check(fitted.ypara.transpose(*ref.dims).values, ref.values, "YPARA")
    ref = case.ref("YAVC")
    case.check(fitted.yavc.transpose(*ref.dims).values, ref.values, "YAVC")


def test_rescaling_coefficients_match_reference(case, fitted):
    for name in ("constant", "independent", "smoothed"):
        ref = case.ref({"constant": "SCGC", "independent": "SCGA", "smoothed": "SCGF"}[name])
        case.check(fitted.scg[name].transpose(*ref.dims).values, ref.values, f"SCG {name}",
                   _season_axis(ref))
        ref = case.ref({"constant": "XSDEMC", "independent": "XSDEMA", "smoothed": "XSDEMF"}[name])
        case.check(fitted.xsdem[name].transpose(*ref.dims).values, ref.values, f"XSDEM {name}",
                   _season_axis(ref))


def test_insample_probabilities_match_reference(case, fitted):
    for name, suffix in _VARIANTS:
        ref = case.ref(f"PXG{suffix}-TR")
        ours = fitted.insample_probabilities[name].transpose(*ref.dims)
        case.check(ours.values, ref.values, f"PXG{suffix}-TR", _season_axis(ref))


def test_insample_crpss_match_reference(case, fitted):
    for name, suffix in _VARIANTS:
        ref = case.ref(f"RPSG{suffix}")
        ours = fitted.crpss[name].transpose(*ref.dims)
        # CRPSS = 1 - CRPS/CRPS_clim amplifies ulp noise: exact outside season 0 and loose
        # inside it on the fixture platform, loose everywhere elsewhere.
        o, r = ours.values, ref.values
        same = (o == r) | (np.isnan(o) & np.isnan(r))
        if same.all():
            continue
        if case.exact:
            assert set(np.argwhere(~same)[:, _season_axis(ref)].tolist()) == {0}
            np.testing.assert_allclose(o[~same], r[~same], rtol=1e-11, atol=0)
        else:
            np.testing.assert_allclose(o, r, rtol=1e-7, atol=1e-12, equal_nan=True)


# ── forecast mode (MakeFcst) ──────────────────────────────────────────────────

def test_forecast_probabilities_match_reference(case, fitted):
    for name, suffix in _VARIANTS:
        ours = cc.predict_tercile(fitted, case.fc, coefficient=name).sel(season=case.init_month)
        ref = case.ref(f"Fcst_PXG{suffix}-TR")
        case.check(ours.transpose("tercile", "lat", "lon").values, ref.values, f"Fcst_PXG{suffix}-TR")


def test_forecast_ensemble_mean_matches_reference(case, fitted):
    for name, suffix in _VARIANTS:
        ours = cc.predict_deterministic(fitted, case.fc, coefficient=name).sel(season=case.init_month)
        ref = case.ref(f"Fcst_EMG{suffix}")
        case.check(ours.transpose("lat", "lon").values, ref.values, f"Fcst_EMG{suffix}")


# ── the one-call wrapper ──────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["constant", "independent", "smoothed"])
def test_calibrate_wrapper_forecast(case, name):
    suffix = dict(_VARIANTS)[name]
    distribution = "gamma" if case.cfg["gamma_transform"] else "normal"
    probs = africas2s.calibrate(
        case.model_dict(), case.obs, method="smoothed_regression", engine="cccma",
        output_type="tercile", temporal_sigma=_SIGMA[name], distribution=distribution,
        small_ic=case.cfg["SMALLIC"],
    )
    assert probs.dims == ("season", "tercile", "lat", "lon")
    assert probs.attrs["engine"] == "cccma" and probs.attrs["coefficient"] == name
    assert probs.attrs["units"] == "1"
    ref = case.ref(f"Fcst_PXG{suffix}-TR")
    case.check(probs.sel(season=case.init_month).transpose("tercile", "lat", "lon").values,
               ref.values, f"calibrate Fcst_PXG{suffix}-TR")
    # the other 11 seasons carry no forecast and are NaN, not garbage
    others = probs.drop_sel(season=case.init_month)
    assert np.isnan(others.values).all()

    det = africas2s.calibrate(
        case.model_dict(), case.obs, method="smoothed_regression", engine="cccma",
        output_type="deterministic", temporal_sigma=_SIGMA[name], distribution=distribution,
        small_ic=case.cfg["SMALLIC"],
    )
    ref = case.ref(f"Fcst_EMG{suffix}")
    case.check(det.sel(season=case.init_month).transpose("lat", "lon").values, ref.values,
               f"calibrate Fcst_EMG{suffix}")


@pytest.mark.parametrize("name", ["constant", "independent", "smoothed"])
def test_calibrate_wrapper_hindcast_year(case, name):
    suffix = dict(_VARIANTS)[name]
    distribution = "gamma" if case.cfg["gamma_transform"] else "normal"
    hind = {k: h for k, (h, _f) in case.model_dict().items()}
    year = int(case.obs["year"].values[5])
    probs = africas2s.calibrate(
        hind, case.obs, method="smoothed_regression", engine="cccma", output_type="tercile",
        temporal_sigma=_SIGMA[name], distribution=distribution, forecast_year=year,
        small_ic=case.cfg["SMALLIC"],
    )
    ref = case.ref(f"PXG{suffix}-TR").sel(year=year)
    case.check(probs.transpose("tercile", "lat", "lon", "season").values,
               ref.transpose("tercile", "latitude", "longitude", "month").values,
               f"calibrate PXG{suffix}-TR[{year}]", season_dim=3)


def test_fixture_platform_is_recorded(case):
    """The exact-bytes branch only fires on the fixture's own platform; make sure the
    bundle says which one that is, so a regeneration elsewhere switches the rule."""
    made_on = case.meta["platform"]
    assert made_on["system"] and made_on["machine"]
