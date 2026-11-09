"""CCCma seasonal-to-decadal (s2d) calibration engine: the operational form of the
Kharin et al. post-processing (ensemble-mean rescaling by CRPS-optimised coefficients,
smoothed across the seasonal cycle, with a variance-preserving spread adjustment),
ported from ECCC/CCCma's ``s2d_calibration`` package as deployed at ICPAC
(``gcm_calibration_ICPAC`` v0.1.0-icpac, wrapping
``CCCma_s2d_postprocessing/Calibration @ f728ce17``: ``MakeCalib.py``, ``MakeFcst.py``,
``utils/calib.py``).

This module is a *numerical* port, not a re-derivation: every reduction, dtype
promotion, scalar type and operation order of the reference is kept so that, for the
same input arrays, the fitted coefficients and the tercile probabilities come out
bit-for-bit identical to the reference package on the same platform (see
``tests/test_cccma_parity.py``; across platforms libm/SIMD ``exp``/``erf`` differ by
an ulp, so the agreement there is ~1e-15 relative rather than exact).
Where the reference makes a choice that looks odd in isolation (e.g. the float32
``X.astype('float32')`` after the gamma transform, the ``np.float64`` tercile constant
promoting float32 anomalies, the ``python float`` mid-bracket in the per-season golden
search), it is reproduced on purpose and commented. Variable names follow the reference
(``XANOU`` unweighted multi-model mean anomaly, ``XVARAS`` smoothed signal variance,
``SCGC``/``SCGA``/``SCGF`` constant/independent/smoothed rescaling coefficients, ...)
so the two code bases can be read side by side.

Public entry points use africas2s' dims — ``(season, year, [model,] member, lat, lon)``
hindcast, ``(season, year, lat, lon)`` obs, ``(season, [model,] member, lat, lon)``
forecast — and translate to the reference's ``(latitude, longitude, month, year, model,
ensemble_member)`` layout internally. ``calibrate(method="smoothed_regression",
engine="cccma")`` is the one-call wrapper (see ``africas2s.calibrate``).

Not ported (not reachable from the africas2s API): probability-of-exceedance output,
the saved empirical/parametric tercile boundaries, the CanSIPS-Datamart temperature
special case, and percent-anomaly deterministic output. Cells where the reference would
fill a *random* normal deviate (gamma transform impossible) are NaN here instead. As in
the reference, a cell with no usable data gets *climatological* tercile probabilities
(1/3 each: its standardised anomaly and spread are set to 0), not NaN — blank such cells
with ``CCCMAFit.masks["SMSK"]`` (the reference's seasonal mask, which it computes but
never applies) if you need them missing.

Contains information licensed under the Open Government Licence – Canada.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import xarray as xr
from scipy.special import comb, erf
from scipy.stats import gamma as _gamma_dist, norm

# ----------------------------------------------------------------------------
# dims: africas2s <-> reference
# ----------------------------------------------------------------------------

_TO_INTERNAL = {"season": "month", "member": "ensemble_member", "lat": "latitude", "lon": "longitude"}
_TO_PUBLIC = {v: k for k, v in _TO_INTERNAL.items()}
_HINDCAST_ORDER = ("latitude", "longitude", "month", "year", "model", "ensemble_member")
_FORECAST_ORDER = ("latitude", "longitude", "month", "model", "ensemble_member")

COEFFICIENTS = ("none", "constant", "independent", "smoothed")
"""Rescaling variants: ``none`` = uncalibrated (reference GU), ``constant`` = one
year-round CRPS-optimal coefficient (GC / ``SCGC``), ``independent`` = one per season
(GA / ``SCGA``), ``smoothed`` = the per-season coefficients Gaussian-smoothed across the
seasonal cycle (GF / ``SCGF``)."""
_SUFFIX = {"none": "U", "constant": "C", "independent": "A", "smoothed": "F"}

# The reference's Fortran special values: cells within SPVALT of SPVAL are missing.
_SPVAL, _SPVALT = 1e38, 1e32


def _internal(da, order, add_model=True):
    da = da.rename({k: v for k, v in _TO_INTERNAL.items() if k in da.dims})
    if add_model and "model" not in da.dims:
        # reference: X.expand_dims('model').assign_coords(model=[1])
        da = da.expand_dims("model").assign_coords(model=[1])
    dims = [d for d in order if d in da.dims]
    extra = [d for d in da.dims if d not in dims]
    if extra:
        raise ValueError(f"cccma: unexpected dims {extra}; expected a subset of "
                         f"{[_TO_PUBLIC.get(d, d) for d in order]}.")
    da = da.transpose(*dims)
    # The reference reads its inputs from NetCDF in exactly this order, so the arrays it
    # reduces are C-contiguous in this layout; numpy's summation order depends on it.
    return da.copy(data=np.ascontiguousarray(da.values))


def _public(da):
    out = da.rename({k: v for k, v in _TO_PUBLIC.items() if k in da.dims})
    if "model" in out.dims and out.sizes["model"] == 1:
        out = out.isel(model=0, drop=True)
    lead = [d for d in ("year", "season", "tercile", "model", "member") if d in out.dims]
    return out.transpose(*lead, ...)


# ----------------------------------------------------------------------------
# small numerical pieces (utils/calib.py)
# ----------------------------------------------------------------------------

_C4 = np.array([
    0.0000000, 0.7978846, 0.8862269, 0.9213177, 0.9399856,
    0.9515329, 0.9593688, 0.9650305, 0.9693107, 0.9726593,
    0.9753501, 0.9775594, 0.9794056, 0.9809714, 0.9823162,
    0.9834835, 0.9845064, 0.9854100, 0.9862141, 0.9869343,
    0.9875829, 0.9881703, 0.9887045, 0.9891927, 0.9896404,
    0.9900525, 0.9904330, 0.9907856, 0.9911130, 0.9914181,
    0.9917028, 0.9919693, 0.9922192, 0.9924540, 0.9926751,
    0.9928836, 0.9930805, 0.9932668, 0.9934434, 0.9936109,
    0.9937701, 0.9939216, 0.9940659, 0.9942034, 0.9943348,
    0.9944603, 0.9945804, 0.9946954, 0.9948056, 0.9949113,
    0.9950128, 0.9951103, 0.9952041, 0.9952944, 0.9953813,
    0.9954651, 0.9955459, 0.9956239, 0.9956991, 0.9957719,
    0.9958422, 0.9959102, 0.9959760, 0.9960398, 0.9961015,
    0.9961614, 0.9962194, 0.9962757, 0.9963304, 0.9963835,
    0.9964351, 0.9964852, 0.9965339, 0.9965813, 0.9966274,
    0.9966723, 0.9967160, 0.9967586, 0.9968001, 0.9968405,
    0.9968800, 0.9969184, 0.9969559, 0.9969926, 0.9970283,
    0.9970632, 0.9970973, 0.9971306, 0.9971632, 0.9971950,
    0.9972261, 0.9972566, 0.9972864, 0.9973155, 0.9973440,
    0.9973719, 0.9973993, 0.9974260, 0.9974523, 0.9974780,
])


def c4_factor(sample_size):
    """``c4(N)`` bias factor of the sample standard deviation (reference ``C4`` table)."""
    return _C4[sample_size - 1]


def adjusted_tercile(sample_size):
    """Standard-normal tercile scaled for the small-sample bias of the sample standard
    deviation: ``TRCSTN = 0.4307273 / c4(N)`` for ``N <= 100``. Returns
    ``(TRCSTN, P_below, P_above)``; ``TRCSTN`` is an ``np.float64`` scalar because the
    reference reads it from a numpy table — that scalar type is what promotes the float32
    anomalies to float64 in ``tercile_probabilities``."""
    tercile = 0.4307273  # norm.ppf(1/3)
    if sample_size <= 100:
        adjusted = tercile / c4_factor(sample_size)
    else:
        adjusted = tercile
    return adjusted, norm.cdf(-adjusted), norm.cdf(adjusted)


def gaussian_filter_weights(width, n_seasons=12):
    """Cyclic filter weights over the seasonal cycle (reference
    ``gaussian_filter_coefficients``): ``width < 0`` uniform average, ``0`` identity,
    ``> 0`` Gaussian ``exp(-k^2 / 2 width^2)`` normalised over the ``n_seasons`` offsets."""
    nmon = int(n_seasons)
    w = np.zeros(nmon)
    if width < 0:
        w[:] = 1.0 / nmon
    elif width == 0:
        w[0] = 1.0
    else:
        w[0] = 1.0
        s = w[0]
        half = nmon // 2
        for ns in range(1, half + 1):
            ns1 = nmon - ns
            w[ns] = np.exp(-ns ** 2 / (2 * width ** 2))
            s += w[ns]
            if ns1 > ns:
                w[ns1] = w[ns]
                s += w[ns1]
        w /= s
    return w


def resolve_filter(width_or_weights, n_seasons=12):
    """``width_or_weights`` is either a scalar width for :func:`gaussian_filter_weights`
    (the reference's ``Gaussian_width``) or an explicit length-``n_seasons`` weight
    vector (the reference's ``custom_smoothing_filter``; first entry is the season
    itself, then increasing offsets)."""
    if np.ndim(width_or_weights) == 0:
        return gaussian_filter_weights(float(width_or_weights), n_seasons)
    w = np.asarray(width_or_weights, dtype=float)
    if w.shape != (n_seasons,):
        raise ValueError(f"cccma: a custom smoothing filter needs {n_seasons} weights, "
                         f"got shape {w.shape}.")
    return w


def cyclic_filter(da, weights, dim="month", condition=None):
    """Smooth ``da`` along the cyclic ``dim`` with ``weights`` (reference
    ``apply_temporal_filter``): for season ``s`` the smoothed value is
    ``sum_k w[k] * da[..., (s + k) % n]``; where ``condition`` is False the original
    value is kept.

    The reference builds that product by label-aligning a rotated weight vector with a
    dask-backed ``da`` and reducing over the season axis; under its distributed
    scheduler the product block reaches numpy with the season axis slowest, so the
    ``n`` terms are accumulated *sequentially* in ``k`` order (not pairwise). That order
    is reproduced here, as is the cast back to ``da``'s dtype of the reference's ``.loc``
    assignment. (A single-process dask run sums season 0 pairwise instead; the two
    orders differ by an ulp now and then.)"""
    if condition is None:
        condition = xr.full_like(da, 0, dtype=bool)
    else:
        condition, _ = xr.broadcast(condition, da)
    smooth = xr.zeros_like(da)
    n = da.sizes[dim]
    axis = da.get_axis_num(dim)
    vals = np.moveaxis(da.values, axis, -1)                            # (..., season)
    w = np.asarray(weights)
    out = np.empty(vals.shape, dtype=np.result_type(w.dtype, vals.dtype))
    for s in range(n):
        acc = None
        for k in range(n):
            term = w[k] * vals[..., (s + k) % n]
            term = np.where(np.isnan(term), 0, term)
            acc = term if acc is None else acc + term
        out[..., s] = acc
    smooth.values[...] = np.moveaxis(out, -1, axis)
    return xr.where(condition, smooth, da)


def cdfstn(x):
    """Standard-normal CDF via ``erf`` (reference ``CDFSTN``), clamped at |x| > 10."""
    rthalf = 0.707106781154
    result = xr.full_like(x, np.nan)
    result = xr.where(x > 10, 1.0, result)
    result = xr.where(x < -10, 0.0, result)
    result = xr.where((x >= -10) & (x <= 10), 0.5 + 0.5 * erf(x * rthalf), result)
    return result


def gamma_possible(avg, sdt, small_sd, small_cv):
    """1 where a gamma fit is meaningful (positive mean, spread and CV), else 0."""
    max_avg = np.maximum(1e-36, avg)
    condition = (avg <= small_sd) | (sdt <= small_sd) | ((sdt / max_avg) <= small_cv)
    return xr.where(condition, 0, 1)


def gamma_lmom_fit(values):
    """Two-parameter gamma fit by L-moments — a transcription of
    ``lmoments3.distr.gam.lmom_fit`` (Hosking's rational approximation) with the sample
    L-moments of ``lmoments3._samlmusmall``, kept operation-for-operation so no
    dependency on lmoments3 is needed. Returns ``(shape, scale)``; NaN where lmoments3
    would raise (``l1 <= l2`` or ``l2 <= 0``)."""
    x = np.asarray(values, dtype=np.float64)
    n = len(x)
    x = np.sort(x)
    l1 = np.sum(x) / comb(n, 1, exact=True)
    comb1 = range(n)
    coefl2 = 0.5 / comb(n, 2, exact=True)
    sum_xtrans = sum([(comb1[i] - comb1[n - i - 1]) * x[i] for i in range(n)])
    l2 = coefl2 * sum_xtrans
    if l1 <= l2 or l2 <= 0:
        return np.nan, np.nan
    a1, a2, a3 = -0.3080, -0.05812, 0.01765
    b1, b2, b3, b4 = 0.7213, -0.5947, -2.1817, 1.2113
    cv = l2 / l1
    if cv >= 0.5:
        t = 1 - cv
        alpha = t * (b1 + t * b2) / (1 + t * (b3 + t * b4))
    else:
        t = math.pi * cv ** 2
        alpha = (1 + a1 * t) / (t * (1 + t * (a2 + t * a3)))
    return alpha.item(), (l1 / alpha).item()


def _gamma_params_block(data):
    data = np.asarray(data).ravel()
    finite = data[np.isfinite(data)]
    if len(finite) <= 3 or np.count_nonzero(finite > 0) < 2 or len(np.unique(finite)) == 1:
        return np.array([np.nan, np.nan])
    return np.array(gamma_lmom_fit(finite))


def gamma_params_lmom(xs, core_dims):
    """Per-cell gamma ``(SHAPE, SCALE)`` of the climatology-normalised sample ``xs``
    pooled over ``core_dims`` (reference ``gamma_param_lmom``; float32 output)."""
    params = xr.apply_ufunc(
        _gamma_params_block, xs,
        input_core_dims=[list(core_dims)], output_core_dims=[["PARA"]],
        vectorize=True, output_dtypes=[np.float32],
    )
    params.coords["PARA"] = ["SHAPE", "SCALE"]
    return params


def _gamma_transform_block(data, parameters, pmin, pmax):
    shape, scale = parameters[0], parameters[1]
    cdf = _gamma_dist.cdf(data, a=shape, scale=scale)
    cdf = np.clip(cdf, pmin, pmax)
    return norm.ppf(cdf)


def gamma_transform_field(xs, xgam, parameters, ntime, core_dims, n_members=None):
    """Map the normalised sample through its fitted gamma CDF into standard-normal
    quantiles, clipped at ``0.5 / (NTIME * NENS)`` (reference ``gamma_transform``; the
    block output is float32, the fill where ``xgam != 1`` is a float64 scalar, so the
    result is float64 — kept). The reference fills impossible cells with a random normal
    deviate; here they are NaN."""
    denom = ntime * n_members if n_members is not None else ntime
    pmin = 0.5 / denom
    pmax = 1.0 - 0.5 / denom
    transformed = xr.apply_ufunc(
        _gamma_transform_block, xs, parameters, pmin, pmax,
        input_core_dims=[list(core_dims), ["PARA"], [], []],
        output_core_dims=[list(core_dims)],
        vectorize=True, output_dtypes=[np.float32],
    )
    return xr.where(xgam == 1, transformed, np.float64(np.nan))


def tercile_probabilities(ano, sde, trcstn):
    """Below/near/above probabilities of ``N(ano, sde)`` against ``±trcstn``
    (reference ``tercile_probabilities``; ``tercile`` axis first)."""
    condition = sde > 0
    safe_sde = xr.where(sde == 0, float("nan"), sde)
    p1 = xr.where(condition, cdfstn((-trcstn - ano) / safe_sde), 1 / 3)
    p3 = xr.where(condition, 1 - cdfstn((trcstn - ano) / safe_sde), 1 / 3)
    p2 = xr.where(condition, 1 - p1 - p3, 1 / 3)
    out = xr.concat([p1, p2, p3], dim="tercile")
    return out.assign_coords(tercile=[0, 1, 2])


def crpsstn(x, amu, sig):
    """CRPS of ``x`` against ``N(amu, sig)`` (reference ``CRPSSTN``)."""
    z = (x - amu) / sig
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    pi_invert = 1 / np.sqrt(np.pi)
    return sig * (z * (2 * cdf_z - 1) + 2 * pdf_z - pi_invert)


def _mean(da, dims):
    """``da.mean(dims)`` with dask's single-chunk arithmetic: the reference computes on
    dask arrays, whose float32 mean divides a float32 sum by a float32 count *in
    float32*, whereas numpy's ``nanmean`` divides in float64 and rounds. Identical for
    float64 input."""
    dims = [dims] if isinstance(dims, str) else list(dims)
    if da.dtype != np.float32:
        return da.mean(dims)
    axes = tuple(da.get_axis_num(d) for d in dims)
    vals = da.values
    total = np.nansum(vals, axis=axes, dtype=np.float32)
    count = np.sum(~np.isnan(vals), axis=axes, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(total, count, dtype=np.float32)
    keep = [d for d in da.dims if d not in dims]
    coords = {k: v for k, v in da.coords.items() if set(v.dims) <= set(keep)}
    return xr.DataArray(out, dims=keep, coords=coords)


def _nanquantile(da, core_dims, q):
    return xr.apply_ufunc(
        np.nanquantile, da, input_core_dims=[list(core_dims)], output_core_dims=[[]],
        kwargs={"q": q, "method": "median_unbiased"}, vectorize=True,
    )


def _mask_missing(ds):
    """Reference ``mask_missing_data``: ``(location mask, seasonal mask)`` of 1/NaN."""
    masked_all = xr.where(np.abs(ds - _SPVAL) >= _SPVALT, ds, np.nan)
    is_nan = masked_all.isnull()
    if "ensemble_member" in ds.dims:
        dims1 = [d for d in ds.dims if d not in ("latitude", "longitude", "ensemble_member")]
        nan_loc = is_nan.all(dim=["ensemble_member"]).any(dim=dims1)
        dims2 = [d for d in ds.dims if d not in ("latitude", "longitude", "month", "ensemble_member")]
        nan_seas = is_nan.all(dim=["ensemble_member"]).any(dim=dims2)
    else:
        dims1 = [d for d in ds.dims if d not in ("latitude", "longitude")]
        nan_loc = is_nan.any(dim=dims1)
        dims2 = [d for d in ds.dims if d not in ("latitude", "longitude", "month")]
        nan_seas = is_nan.any(dim=dims2)
    return xr.where(nan_loc, np.nan, 1), xr.where(nan_seas, np.nan, 1)


# ----------------------------------------------------------------------------
# CRPS-optimal rescaling coefficient: golden-section search (per cell, numpy)
# ----------------------------------------------------------------------------

def _crps_np(x, amu, sig):
    z = (x - amu) / sig
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    pi_invert = 1 / np.sqrt(np.pi)
    return sig * (z * (2 * cdf_z - 1) + 2 * pdf_z - pi_invert)


def _nansum_all(a):
    # xarray's skipna sum: zero the NaNs, then np.sum over every axis
    return np.sum(np.where(np.isnan(a), 0, a), axis=tuple(range(a.ndim)))


def bsopt(f, xano, yano, xvart, xvara, ysdt, ntime, small):
    """Mean CRPS over (year[, month]) of the forecast ``N(f * xano / sdt,
    sqrt(xvart - xvara f^2) / sdt)`` against the standardised obs anomaly (reference
    ``BSOPT``). Arrays are one cell: ``xano``/``yano`` ``(month, year)`` or ``(year,)``,
    the variances ``(month,)`` or scalars. Mirrors the reference's xarray arithmetic in
    numpy, including the dtype of ``f`` (an ``np.float32``/``np.float64`` scalar from the
    bracket, or the Python-float mid bracket)."""
    seasonal = xano.ndim == 2
    xsdt = np.sqrt(xvart)
    condition = np.logical_and(xsdt > small, ysdt > small)
    nseas = np.where(condition, 1, 0).sum() if seasonal else np.where(condition, 1, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        xsden_ = np.where(xsdt == 0, np.nan, np.sqrt(xvart - xvara * np.square(f)) / xsdt)
        xsden_ = np.where(condition, xsden_, np.nan)
        xsden = np.where(xsden_ > 0, xsden_, small)
        if seasonal:
            xanon = np.where(xsdt[:, None] == 0, np.nan, f * xano / xsdt[:, None])
            yanon = np.where(ysdt[:, None] == 0, np.nan, yano / ysdt[:, None])
            crps = _crps_np(yanon, xanon, xsden[:, None])
        else:
            xanon = np.where(xsdt == 0, np.nan, f * xano / xsdt)
            yanon = np.where(ysdt == 0, np.nan, yano / ysdt)
            crps = _crps_np(yanon, xanon, xsden)
        total = _nansum_all(crps)
        return np.where(nseas == 0, np.nan, total / (ntime * nseas))


def golden(func, brack, args=(), tol=np.sqrt(np.finfo(float).eps), itmax=1000):
    """Golden-section minimisation of ``func(x, *args)`` on the bracket ``(ax, bx, cx)``
    (reference ``golden``). Returns the minimiser."""
    r = 0.61803399
    c = 1.0 - r
    ax, bx, cx = brack
    x0, x3 = ax, cx
    if np.abs(cx - bx) > np.abs(bx - ax):
        x1 = bx
        x2 = bx + c * (cx - bx)
    else:
        x2 = bx
        x1 = bx - c * (bx - ax)
    f1 = func(x1, *args)
    f2 = func(x2, *args)
    it = 1
    while np.abs(x3 - x0) > tol * (np.abs(x1) + np.abs(x2)):
        if np.any(f2 < f1):
            x0, x1 = x1, x2
            x2 = r * x1 + c * x3
            f1, f2 = f2, func(x2, *args)
        else:
            x3, x2 = x2, x1
            x1 = r * x2 + c * x0
            f2, f1 = f1, func(x1, *args)
        it += 1
        if it > itmax:
            break
    return x1 if np.any(f1 < f2) else x2


def _scgc(xanou, yano, xvarts, xvaras, ysdf, ntime, small, tol):
    """Reference ``get_SCGC`` (seasonal): one year-round coefficient per cell, bracket
    from the smoothed variance ratio (min over months), mid bracket 0.5 in the array's
    own dtype."""
    with np.errstate(divide="ignore", invalid="ignore"):
        fmax_ = xr.where(xvaras > 0.0,
                         np.minimum(2.0, np.maximum(1.0, np.sqrt(xvarts / xvaras) - small)),
                         1.0 + small)
    fmax = xr.where(fmax_ < 1e38, fmax_, 1e38)
    if "month" in xanou.dims:
        fmax = fmax.min(dim="month")
    fmin = -fmax
    fmid = 0.5 * xr.ones_like(fmin)
    nlat, nlon = xanou.sizes["latitude"], xanou.sizes["longitude"]
    xa, ya = xanou.values, yano.values
    vt, va, sd = xvarts.values, xvaras.values, ysdf.values
    results = []
    for i in range(nlat):
        for j in range(nlon):
            ysd = sd[i, j]
            if (ysd == 0).all():
                results.append(np.nan)
                continue
            brack = (fmin.values[i, j], fmid.values[i, j], fmax.values[i, j])
            results.append(golden(bsopt, brack, args=(xa[i, j], ya[i, j], vt[i, j], va[i, j], ysd, ntime, small), tol=tol))
    fopt = xr.DataArray(np.array(results).reshape(nlat, nlon), dims=["latitude", "longitude"],
                        coords={"latitude": xanou.latitude, "longitude": xanou.longitude})
    if "month" in xanou.dims:
        return fopt.expand_dims(month=xanou.month.values).transpose("latitude", "longitude", "month").copy()
    return fopt


def _scgc_decadal(xanou, yano, xvarts, xvaras, ysdf, ntime, small, tol):
    """Reference ``get_SCGC_decadal`` (no season axis): per-cell bracket, with the
    zero-variance short-cuts of the per-season search."""
    nlat, nlon = xanou.sizes["latitude"], xanou.sizes["longitude"]
    xa, ya = xanou.values, yano.values
    vt, va, sd = xvarts.values, xvaras.values, ysdf.values
    results = []
    for i in range(nlat):
        for j in range(nlon):
            xvart, xvara, ysd = vt[i, j], va[i, j], sd[i, j]
            if (ysd == 0).all():
                results.append(np.nan)
            elif (xvart == 0) | (xvara <= 0):
                results.append(0)
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    fmax = np.minimum(2.0, np.maximum(1.0, np.sqrt(np.clip(xvart / xvara, 0, None)) - small))
                results.append(golden(bsopt, (-fmax, 0.5, fmax),
                                      args=(xa[i, j], ya[i, j], xvart, xvara, ysd, ntime, small), tol=tol))
    return xr.DataArray(np.array(results).reshape(nlat, nlon), dims=["latitude", "longitude"],
                        coords={"latitude": xanou.latitude, "longitude": xanou.longitude})


def _scga(xanou, yano, xvartu, xvarau, ysdf, ntime, small, tol):
    """Reference ``get_SCGA``: one coefficient per cell and season from the *unsmoothed*
    variances; the mid bracket is the Python float 0.5 (which makes the first CRPS
    evaluation run in float64 even for float32 fields — kept)."""
    nlat, nlon, nmon = xanou.sizes["latitude"], xanou.sizes["longitude"], xanou.sizes["month"]
    xa, ya = xanou.values, yano.values
    vt, va, sd = xvartu.values, xvarau.values, ysdf.values
    results = []
    for i in range(nlat):
        for j in range(nlon):
            for s in range(nmon):
                xvart, xvara, ysd = vt[i, j, s], va[i, j, s], sd[i, j, s]
                if (ysd == 0).all():
                    results.append(np.nan)
                elif (xvart == 0) | (xvara <= 0):
                    results.append(0)
                else:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        fmax = np.minimum(2.0, np.maximum(1.0, np.sqrt(np.clip(xvart / xvara, 0, None)) - small))
                    results.append(golden(bsopt, (-fmax, 0.5, fmax),
                                          args=(xa[i, j, s], ya[i, j, s], xvart, xvara, ysd, ntime, small), tol=tol))
    return xr.DataArray(np.array(results).reshape(nlat, nlon, nmon), dims=["latitude", "longitude", "month"],
                        coords={"latitude": xanou.latitude, "longitude": xanou.longitude, "month": xanou.month})


# ----------------------------------------------------------------------------
# fit (MakeCalib) / predict (MakeFcst)
# ----------------------------------------------------------------------------

@dataclass
class CCCMAFit:
    """Everything ``MakeCalib`` writes for ``MakeFcst`` plus the in-sample diagnostics.
    Fields are in the reference's internal layout and names; use the public helpers
    (:func:`predict_tercile`, :func:`predict_deterministic`, :func:`hindcast_tercile`,
    :func:`hindcast_deterministic`) or :meth:`coefficient` for africas2s-shaped views."""
    ntime: int
    gamma_transform: bool
    small: float
    trcstn: float
    seasonal: bool
    xavg: xr.DataArray            # model climatology (transformed space when gamma)
    xavc: xr.DataArray            # model climatology, untransformed
    yavg: xr.DataArray            # obs climatology (transformed space when gamma)
    yavc: xr.DataArray            # obs climatology, untransformed
    xsdtms: xr.DataArray          # multi-model total std, smoothed
    xsdem: dict                   # coefficient -> adjusted spread (XSDEMS/C/A/F)
    scg: dict                     # coefficient -> rescaling coefficient (1 / SCGC / SCGA / SCGF)
    xanou: xr.DataArray           # hindcast ensemble-mean anomalies (month, year)
    xpara: xr.DataArray | None = None
    ypara: xr.DataArray | None = None
    insample_probabilities: dict = field(default_factory=dict)   # coefficient -> PXG?-TR
    crpss: dict = field(default_factory=dict)                    # coefficient -> in-sample CRPSS
    masks: dict = field(default_factory=dict)                    # XFLT, YFLT, SMSK
    diagnostics: dict = field(default_factory=dict)              # other reference fields

    def coefficient(self, which="smoothed"):
        """Rescaling coefficient field ``(season, lat, lon)`` (``SCGF`` etc.)."""
        return _public(self.scg[_check_coefficient(which, self.seasonal)])


def _check_coefficient(which, seasonal):
    if which not in COEFFICIENTS:
        raise ValueError(f"cccma: coefficient must be one of {COEFFICIENTS}, got {which!r}.")
    if not seasonal and which in ("independent", "smoothed"):
        raise ValueError(
            f"cccma: coefficient={which!r} needs a season axis; without one only "
            "'constant' (one coefficient per cell) and 'none' exist.")
    return which


def fit(hindcast, obs, *, gamma_transform=False, coefficient_width=2, obs_variance_width=1,
        fcst_variance_width=1, golden_tolerance=1e-3, small=1e-6, small_ic=None):
    """Fit the CCCma calibration on a hindcast (reference ``MakeCalib.calibrate``).

    hindcast: ``(season, year, [model,] member, lat, lon)`` — several models stack on a
    ``model`` dim with NaN-padded members (see :func:`stack_models`); obs: ``(season,
    year, lat, lon)``. The ``season`` axis is the 12 initialisation months of the
    reference (any labels, size 12) or absent (the reference's decadal path: no
    smoothing, ``constant`` coefficient only). Years must match (positionally the
    reference uses ``last - first + 1`` consecutive years).

    gamma_transform: map both fields through per-cell L-moment gamma fits into normal
    space first (precipitation). coefficient_width / obs_variance_width /
    fcst_variance_width: the reference's ``Gaussian_width`` ``calib``/``obsVar``/
    ``fcstVar`` (scalar width, or a 12-vector of custom weights). golden_tolerance,
    small: ``Golden_tolerance``, ``SMALL``. small_ic: ``SMALLIC`` (``None`` → the
    reference template's ``-1e-9`` with the gamma transform, ``0`` without).
    """
    X = _internal(hindcast, _HINDCAST_ORDER)
    Y = _internal(obs, ("latitude", "longitude", "month", "year"), add_model=False)
    if (X.sizes["year"] != Y.sizes["year"]) or not np.array_equal(X["year"].values, Y["year"].values):
        raise ValueError("cccma: hindcast and obs must carry the same years.")
    seasonal = "month" in X.dims
    if seasonal and "month" not in Y.dims:
        raise ValueError("cccma: obs lack the season axis the hindcast has.")
    if seasonal and X.sizes["month"] != 12:
        raise ValueError(
            f"cccma: the season axis must hold the 12 initialisation months of the "
            f"reference (got {X.sizes['month']}); drop it for a single-season fit.")

    SMALL = float(small)
    if small_ic is None:
        small_ic = -1e-9 if gamma_transform else 0.0
    SMALLIC = float(small_ic)
    if SMALLIC != 0:
        SMALLSD = abs(SMALLIC)
        SMALLTE = abs(SMALLIC) if SMALLIC > 0 else 0
        SMALLSCL = SMALLSD
        SMALLCV = SMALLSD
    else:
        SMALLSD = SMALLTE = SMALLSCL = SMALLCV = 1e-6

    NTIME = X.sizes["year"]
    NMOD = X.sizes["model"]
    NENS_per_model = xr.DataArray(
        np.array([X.isel(model=i).dropna(dim="ensemble_member", how="all").sizes["ensemble_member"]
                  for i in range(NMOD)]),
        dims=["model"], coords={"model": X.model.values})
    NENS_total = sum(NENS_per_model).item()

    TRCSTN, PBNSTN, PANSTN = adjusted_tercile(int(NTIME))
    nmon = X.sizes["month"] if seasonal else 12
    FLTX = resolve_filter(fcst_variance_width, nmon)
    FLTY = resolve_filter(obs_variance_width, nmon)
    FLTC = resolve_filter(coefficient_width, nmon)

    XFLT, SMSK_X = _mask_missing(X)
    YFLT, SMSK_Y = _mask_missing(Y)
    SMSK = (SMSK_X == 1.0) & (SMSK_Y == 1.0)

    XAVC = XSDC = YAVC = YSDC = XPARA = YPARA = None
    condition_XY = None
    gamma_done = False
    while True:
        YAVG = Y.mean(dim=["year"])
        YANO = Y - YAVG
        YVAR = (np.square(YANO)).sum(dim=["year"]) / (NTIME - 1)
        YSDT = np.sqrt(YVAR)

        XAVG = _mean(X, ["ensemble_member", "year"])
        XANO = _mean(X - XAVG, ["ensemble_member"])
        XANOU = _mean(XANO, ["model"])

        XSST = (np.square(X - XAVG)).sum(dim=["ensemble_member", "year"])
        XSSE = (np.square(X - XAVG - XANO)).sum(dim=["ensemble_member", "year"])
        XSSEU = (np.square(X - XAVG - XANOU)).sum(dim=["ensemble_member", "year", "model"])
        XSSA = XSST - XSSE
        XSSTU = XSST.sum(dim=["model"])

        XSDE = XSSE / (NTIME * (NENS_per_model - 1))
        XSDA = XSSA / ((NTIME - 1) * NENS_per_model) - XSDE / NENS_per_model
        XSDT = XSDE + XSDA
        XSDE = np.sqrt(XSDE)
        XSDT = np.sqrt(XSDT)
        XSDM_ = (np.square(XANO)).sum(dim=["year"])
        XSDM = np.sqrt(XSDM_ / (NTIME - 1))

        if not gamma_done:
            XTRE1 = _nanquantile(X, ["year", "ensemble_member"], PBNSTN)
            XTRE2 = _nanquantile(X, ["year", "ensemble_member"], PANSTN)
            YTRE1 = _nanquantile(Y, ["year"], PBNSTN)
            YTRE2 = _nanquantile(Y, ["year"], PANSTN)

            condition_variance_X = (XSDM < SMALLSD) | ((XTRE2 - XTRE1) < SMALLTE)
            condition_variance_Y = (YSDT < SMALLSD) | ((YTRE2 - YTRE1) < SMALLTE)
            SMSK = xr.where(condition_variance_X.any(dim="model"), np.nan, SMSK)
            SMSK = xr.where(condition_variance_Y, np.nan, SMSK)
            if seasonal:
                NZERO_X = condition_variance_X.sum(dim=["month"])
                XFLT = xr.where((NZERO_X > 0).any(dim="model"), np.nan, XFLT)
                NZERO_Y = condition_variance_Y.sum(dim=["month"])
                YFLT = xr.where(NZERO_Y > 0, np.nan, YFLT)
            if gamma_transform:
                YSmax = Y.max(dim=["year"])
                YSmax2nd = Y.where(Y != YSmax).max(dim=["year"])
                condition_ntime_zeros = YSmax2nd < SMALL
                SMSK = xr.where(condition_ntime_zeros, np.nan, SMSK)
                if seasonal:
                    NZERO = condition_ntime_zeros.sum(dim=["month"])
                    YFLT = xr.where(NZERO > 0, np.nan, YFLT)
            condition_XY = (XFLT == 1.0) & (YFLT == 1.0)

        YVAF = cyclic_filter(YVAR, FLTY, "month", condition_XY) if seasonal else YVAR
        YSDF = np.sqrt(YVAF)

        if gamma_transform and not gamma_done:
            XAVC, XSDC, YAVC, YSDC = XAVG.copy(), XSDT.copy(), YAVG.copy(), YSDF.copy()
            XS = X / XAVG
            YS = Y / YAVG
            YGAM = gamma_possible(YAVG, YSDT, SMALLSD, SMALLCV)
            YPARA = gamma_params_lmom(YS, ["year"])
            YPARA.loc[dict(PARA="SHAPE")] = xr.where(YGAM == 1, YPARA.sel(PARA="SHAPE"), 1)
            YPARA.loc[dict(PARA="SCALE")] = xr.where(YGAM == 1, YPARA.sel(PARA="SCALE"), SMALLSCL)
            YPARA = YPARA.copy(deep=True)
            Y = gamma_transform_field(YS, YGAM, YPARA, NTIME, ["year"])
            XGAM = gamma_possible(XAVG, XSDT, SMALLSD, SMALLCV)
            XPARA = gamma_params_lmom(XS, ["ensemble_member", "year"])
            XPARA.loc[dict(PARA="SHAPE")] = xr.where(XGAM == 1, XPARA.sel(PARA="SHAPE"), 1)
            XPARA.loc[dict(PARA="SCALE")] = xr.where(XGAM == 1, XPARA.sel(PARA="SCALE"), SMALLSCL)
            X = gamma_transform_field(XS, XGAM, XPARA, NTIME, ["ensemble_member", "year"],
                                n_members=X.sizes["ensemble_member"])
            X = X.astype("float32")   # reference: X = X.astype('float32') after reloading X_gamma
            gamma_done = True
            continue
        break

    XSSAU = XSSTU - XSSEU
    XVAREU = XSSEU / (NTIME * (NENS_total - 1) - NMOD + 1)
    XVARAU = XSSAU / ((NTIME - 1) * NENS_total) - XVAREU / (NENS_total)
    XVARTU = XVARAU + XVAREU
    if seasonal:
        XVARAS = cyclic_filter(XVARAU, FLTX, "month", condition_XY)
        XVARES = cyclic_filter(XVAREU, FLTX, "month", condition_XY)
        XVARTS = cyclic_filter(XVARTU, FLTX, "month", condition_XY)
    else:
        XVARAS, XVARES, XVARTS = XVARAU, XVAREU, XVARTU
    XSDTMU = np.sqrt(XVARTU)
    XSDTMS = np.sqrt(XVARTS)
    XSDEMU = np.sqrt(XVAREU)
    XSDEMS = np.sqrt(XVARES)
    YANON = xr.where(YSDT > 1e-6, YANO / YSDF, 0)

    RPO = crpsstn(YANON, 0, 1).mean(dim=["year"]).copy()

    TOL = float(golden_tolerance)
    if seasonal:
        SCGC = _scgc(XANOU, YANO, XVARTS, XVARAS, YSDF, NTIME, SMALL, TOL)
    else:
        SCGC = _scgc_decadal(XANOU, YANO, XVARTS, XVARAS, YSDF, NTIME, SMALL, TOL)
    XVAREC = XVARTS - XVARAS * np.square(SCGC)
    XSDEMC = xr.where((XVAREC > 0), np.sqrt(XVAREC), SMALL)

    scg = {"none": 1, "constant": SCGC}
    xsdem = {"none": XSDEMS, "constant": XSDEMC}
    if seasonal:
        SCGA = _scga(XANOU, YANO, XVARTU, XVARAU, YSDF, NTIME, SMALL, TOL)
        XVAREA = XVARTS - XVARAS * np.square(SCGA)
        XSDEMA = xr.where((XVAREA > 0), np.sqrt(XVAREA), SMALL)
        SCGF = cyclic_filter(SCGA, FLTC, "month", condition_XY)
        XVAREF = XVARTS - XVARAS * np.square(SCGF)
        XSDEMF = xr.where((XVAREF > 0), np.sqrt(XVAREF), SMALL)
        scg.update(independent=SCGA, smoothed=SCGF)
        xsdem.update(independent=XSDEMA, smoothed=XSDEMF)

    # in-sample probabilities and CRPSS (reference PXG?-TR, RPSG?)
    insample, crpss = {}, {}
    condition_XSDTMS = (XSDTMS > SMALL)
    for name in scg:
        if name == "none":
            ANO = xr.where(condition_XSDTMS, XANOU / XSDTMS, 0)
            SDE = xr.where(condition_XSDTMS, XSDEMU / XSDTMS, 0)
        else:
            ANO = xr.where(condition_XSDTMS, XANOU * scg[name] / XSDTMS, 0)
            SDE = xr.where(condition_XSDTMS, xsdem[name] / XSDTMS, 0)
        insample[name] = tercile_probabilities(ANO, SDE, TRCSTN)
        RPG = crpsstn(YANON, ANO, SDE).mean(dim=["year"]).copy()
        crpss[name] = 1 - RPG / RPO

    return CCCMAFit(
        ntime=int(NTIME), gamma_transform=bool(gamma_transform), small=SMALL, trcstn=TRCSTN,
        seasonal=seasonal, xavg=XAVG, xavc=XAVC if XAVC is not None else XAVG,
        yavg=YAVG, yavc=YAVC if YAVC is not None else YAVG, xsdtms=XSDTMS, xsdem=xsdem, scg=scg,
        xanou=XANOU, xpara=XPARA, ypara=YPARA, insample_probabilities=insample, crpss=crpss,
        masks={"XFLT": XFLT, "YFLT": YFLT, "SMSK": SMSK},
        diagnostics={"XSDTMU": XSDTMU, "XSDEMU": XSDEMU, "XSDEMS": XSDEMS, "XVARAU": XVARAU,
                     "XVAREU": XVAREU, "XVARTU": XVARTU, "XVARAS": XVARAS, "XVARES": XVARES,
                     "XVARTS": XVARTS, "YSDT": YSDT, "YSDF": YSDF, "YANON": YANON, "RPO": RPO,
                     "XSDC": XSDC, "YSDC": YSDC, "XSDT": XSDT, "XSDE": XSDE, "XSDM": XSDM},
    )


def _gamma_ppf(x, a, scale):
    return _gamma_dist.ppf(x, a=a, scale=scale)


def _gamma_back(z, para, avc):
    """Normal space -> physical units through the fitted gamma: ``gamma.ppf(Phi(z)) * AVC``
    (reference ``EMG? = gamma_ppf(norm.cdf(ENM?_def), YSHPc, YSCLc) * YAVC``)."""
    shp = para.sel(PARA="SHAPE")
    scl = para.sel(PARA="SCALE")
    cdf = xr.apply_ufunc(norm.cdf, z)
    return xr.apply_ufunc(_gamma_ppf, cdf, shp, scl, output_dtypes=[z.dtype]) * avc


def _forecast_anomaly(fit_, forecast):
    """Reference ``MakeFcst`` up to ``XANOU``: gamma-transform the forecast members with
    the hindcast parameters, ensemble mean per model, anomaly from the model climatology
    (``XENM - XAVG``, not the hindcast's ``(X - XAVG).mean``). Returns ``(XANO, XANOU)``."""
    if "member" not in forecast.dims:
        forecast = forecast.expand_dims("member")      # a single realisation is a 1-member ensemble
    X = _internal(forecast, _FORECAST_ORDER)
    if fit_.seasonal:
        if "month" not in X.dims:
            raise ValueError("cccma: the forecast needs the fit's season axis (reindex it onto the "
                             "hindcast seasons; absent seasons may be NaN).")
        if not np.array_equal(X["month"].values, fit_.xavg["month"].values):
            raise ValueError("cccma: forecast seasons must match the hindcast seasons.")
    if X.sizes["model"] != fit_.xavg.sizes["model"]:
        raise ValueError("cccma: the forecast must carry the same models as the hindcast.")
    X = X.assign_coords(model=fit_.xavg["model"].values)
    if fit_.gamma_transform:
        XGAM = xr.ones_like(fit_.xavc)
        XS = X / fit_.xavc
        X = gamma_transform_field(XS, XGAM, fit_.xpara, fit_.ntime, ["ensemble_member"],
                            n_members=X.sizes["ensemble_member"])
    XENM = _mean(X, ["ensemble_member"])
    XANO = XENM - fit_.xavg
    XANOU = _mean(XANO, ["model"])
    return XANO, XANOU


def predict_tercile(fit_, forecast, coefficient="smoothed"):
    """Tercile probabilities ``(season, tercile, lat, lon)`` for an out-of-sample
    forecast ensemble (reference ``MakeFcst`` ``Fcst_PXG?-TR``)."""
    which = _check_coefficient(coefficient, fit_.seasonal)
    _, XANOU = _forecast_anomaly(fit_, forecast)
    XANOC = fit_.scg[which] * XANOU if which != "none" else XANOU
    condition = fit_.xsdtms > fit_.small
    ANO = xr.where(condition, XANOC / fit_.xsdtms, 0)
    SDE = xr.where(condition, fit_.xsdem[which] / fit_.xsdtms, 0)
    return _public(tercile_probabilities(ANO, SDE, fit_.trcstn))


def predict_deterministic(fit_, forecast, coefficient="smoothed"):
    """Calibrated ensemble-mean forecast in obs units ``(season, lat, lon)``: the
    rescaled multi-model anomaly added to the obs climatology, mapped back through the
    obs gamma when the fit was transformed (reference ``Fcst_EMG?``)."""
    which = _check_coefficient(coefficient, fit_.seasonal)
    _, XANOU = _forecast_anomaly(fit_, forecast)
    XANOC = fit_.scg[which] * XANOU if which != "none" else XANOU
    ENM = XANOC + fit_.yavg
    if fit_.gamma_transform:
        ENM = _gamma_back(ENM, fit_.ypara, fit_.yavc)
    else:
        ENM = ENM.copy()
    return _public(ENM)


def hindcast_tercile(fit_, coefficient="smoothed", year=None):
    """In-sample tercile probabilities of the hindcast years (reference ``PXG?-TR``):
    ``(year, season, tercile, lat, lon)``, or one year's ``(season, tercile, lat, lon)``."""
    which = _check_coefficient(coefficient, fit_.seasonal)
    p = fit_.insample_probabilities[which]
    if year is not None:
        p = p.sel(year=year, drop=True)
    return _public(p)


def hindcast_deterministic(fit_, coefficient="smoothed", year=None):
    """In-sample calibrated ensemble mean of the hindcast years, obs units — the
    hindcast-mode analogue of :func:`predict_deterministic` (no reference counterpart:
    ``MakeCalib`` writes probabilities only)."""
    which = _check_coefficient(coefficient, fit_.seasonal)
    XANOU = fit_.xanou if year is None else fit_.xanou.sel(year=year, drop=True)
    XANOC = fit_.scg[which] * XANOU if which != "none" else XANOU
    ENM = XANOC + fit_.yavg
    if fit_.gamma_transform:
        ENM = _gamma_back(ENM, fit_.ypara, fit_.yavc)
    return _public(ENM)


def stack_models(fields, member_dim="member"):
    """Stack per-model ``(…, member, …)`` DataArrays on a new ``model`` dim with members
    renumbered positionally and NaN-padded to the largest ensemble — the reference's
    multi-model layout (``NENS_per_model`` counts the non-NaN members). ``fields`` is a
    ``{model: DataArray}`` mapping."""
    stacked = []
    for name, da in fields.items():
        if member_dim not in da.dims:
            da = da.expand_dims(member_dim)
        da = da.assign_coords({member_dim: np.arange(da.sizes[member_dim])})
        stacked.append(da.expand_dims(model=[name]))
    return xr.concat(stacked, dim="model", join="outer")
