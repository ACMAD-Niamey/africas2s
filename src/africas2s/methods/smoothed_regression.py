"""Kharin et al. (2017) deterministic postprocessing: per-gridpoint ensemble-mean
regression coefficients smoothed across the seasonal cycle to reduce sampling error.
Reuses EnsembleRegressionMethod for the per-cell fit; adds only the season smoothing.
"""
import numpy as np
import xarray as xr
from scipy import optimize, stats
from scipy.ndimage import gaussian_filter1d

from .ensemble_regression import EnsembleRegressionMethod

_EPS = 1e-12


def _smooth_seasonal_coefficients(a, temporal_sigma):
    """a: xr.DataArray (season, lat, lon). Cyclic smoothing over the season axis.
    None -> unchanged; float -> gaussian_filter1d(sigma, mode='wrap').

    The time-invariant 'constant' case is NOT handled here: it is a pooled regression
    over the underlying data (not a transform of the per-season coefficients), so it is
    computed in seasonal_coefficients()."""
    if temporal_sigma is None:
        return a.copy()
    if temporal_sigma == "constant":
        raise ValueError(
            "temporal_sigma='constant' is a pooled regression over the data; use "
            "seasonal_coefficients(...), not _smooth_seasonal_coefficients(...)."
        )
    axis = a.get_axis_num("season")
    vals = gaussian_filter1d(a.values, sigma=float(temporal_sigma), axis=axis, mode="wrap")
    return a.copy(data=vals)


def _pooled_coefficient(predictor_hindcast, obs):
    """Single per-gridpoint slope from all seasons pooled (the paper's time-invariant C):
    remove each season's climatology, pool the (season, year) anomalies into one sample,
    and fit one slope per cell. Reuses EnsembleRegressionMethod on the pooled anomalies
    (their pooled mean is 0, so its intercept-OLS slope equals the pooled Cov/Var).
    Returns (lat, lon)."""
    fbar = predictor_hindcast.mean("member") if "member" in predictor_hindcast.dims \
        else predictor_hindcast
    fa = fbar - fbar.mean("year")                      # per-season anomalies
    oa = obs - obs.mean("year")
    # Pool (season, year) into a single sample dim named "year" (what EnsembleRegressionMethod
    # expects). Drop the stack MultiIndex first so its retained "year" level doesn't collide.
    fa_p = fa.stack(_sy=("season", "year")).reset_index("_sy", drop=True).rename({"_sy": "year"})
    oa_p = oa.stack(_sy=("season", "year")).reset_index("_sy", drop=True).rename({"_sy": "year"})
    m = EnsembleRegressionMethod()
    m.fit(fa_p, oa_p)
    return xr.DataArray(m.slope_, dims=(m.lat_dim_, m.lon_dim_), coords=m.predictand_coords_)


def seasonal_coefficients(predictor_hindcast, obs, temporal_sigma=None):
    """Per-gridpoint ensemble-mean regression slope a = Cov(Fbar, O)/Var(Fbar), as a
    function of the seasonal cycle. predictor_hindcast (season, year, member, lat, lon),
    obs (season, year, lat, lon) on the same grid -> coefficient (season, lat, lon).

    temporal_sigma:
      None       -> per-season coefficients, unsmoothed.
      float >= 0 -> per-season coefficients, cyclically Gaussian-smoothed across seasons.
      'constant' -> a single time-invariant coefficient (pooled regression over all
                    seasons, the paper's C), broadcast across the season axis. This pooled
                    fit is the paper's time-invariant coefficient and differs slightly from
                    the large-sigma limit of the Gaussian dial (which averages the
                    per-season slopes); the paper notes the pooled fit scores slightly
                    better."""
    seasons = predictor_hindcast["season"]
    if temporal_sigma == "constant":
        pooled = _pooled_coefficient(predictor_hindcast, obs)          # (lat, lon)
        return pooled.expand_dims(season=seasons.values).transpose("season", ...)
    slopes = []
    for s in seasons.values:
        m = EnsembleRegressionMethod()
        m.fit(predictor_hindcast.sel(season=s), obs.sel(season=s))
        slope = xr.DataArray(m.slope_, dims=(m.lat_dim_, m.lon_dim_),
                             coords=m.predictand_coords_).assign_coords(season=s)
        slopes.append(slope)
    a = xr.concat(slopes, dim="season")
    return _smooth_seasonal_coefficients(a, temporal_sigma)


# --- probabilistic calibration (round 2): mean-scaling a + spread-scaling b ---

def fit_ab(mu_f, sigma_f, o, *, constrained=True):
    """Mean/spread scaling for one gridpoint-season -> (a, b), b >= 0.
    constrained: a = Cov(mu_f,o)/Var(mu_f); b sized so a^2 Var(mu_f) + b^2 mean(sigma_f^2)
    matches Var(o). unconstrained: minimize mean Gaussian CRPS over (a,b)."""
    mu_f = np.asarray(mu_f, float); sigma_f = np.asarray(sigma_f, float); o = np.asarray(o, float)
    ok = np.isfinite(mu_f) & np.isfinite(sigma_f) & np.isfinite(o)
    if ok.sum() < 3:
        return np.nan, np.nan
    mu_f, sigma_f, o = mu_f[ok], sigma_f[ok], o[ok]
    var_mu, var_o = np.var(mu_f), np.var(o)
    if var_mu < _EPS or np.mean(sigma_f ** 2) < _EPS:
        return np.nan, np.nan
    if constrained:
        a = np.cov(mu_f, o, bias=True)[0, 1] / var_mu
        b = np.sqrt(max(var_o - a ** 2 * var_mu, 0.0) / np.mean(sigma_f ** 2))
        return float(a), float(b)
    from ..metrics.crpss import crps_normal
    a0, b0 = fit_ab(mu_f, sigma_f, o, constrained=True)
    if not np.isfinite(a0):
        a0, b0 = 1.0, 1.0
    res = optimize.minimize(lambda p: crps_normal(p[0] * mu_f, abs(p[1]) * sigma_f, o).mean(),
                            x0=[a0, max(b0, 1e-3)], method="Nelder-Mead")
    return float(res.x[0]), float(abs(res.x[1]))


def fit_ab_field(mu_f, sigma_f, o, *, constrained=True):
    """(season,year,lat,lon) -> (a, b) each (season,lat,lon)."""
    ns, ny, nla, nlo = mu_f.shape
    a = np.full((ns, nla, nlo), np.nan); b = np.full((ns, nla, nlo), np.nan)
    for s in range(ns):
        for i in range(nla):
            for j in range(nlo):
                a[s, i, j], b[s, i, j] = fit_ab(mu_f[s, :, i, j], sigma_f[s, :, i, j],
                                                o[s, :, i, j], constrained=constrained)
    return a, b


def smooth_ab(a, b, temporal_sigma):
    """Smooth a and b across the SEASON axis (axis 0). None -> unchanged; float -> cyclic
    Gaussian (via _smooth_seasonal_coefficients on the moved axis); 'constant' -> mean across
    seasons (matches the Phase-A experiment; NOT the deterministic pooled 'constant')."""
    def _s(x):
        x = np.asarray(x, float)
        if temporal_sigma is None:
            return x.copy()
        if temporal_sigma == "constant":
            return np.broadcast_to(np.nanmean(x, axis=0), x.shape).copy()
        xl = np.moveaxis(x, 0, -1)
        # _smooth_seasonal_coefficients expects an xr.DataArray with a "season" dim
        # (it uses .get_axis_num("season") / .values); wrap the moved-axis ndarray so
        # the existing cyclic-Gaussian smoother is reused with identical math, then
        # unwrap back to a plain ndarray and move the season axis back to position 0.
        dims = [f"_d{i}" for i in range(xl.ndim - 1)] + ["season"]
        da = xr.DataArray(xl, dims=dims)
        smoothed = _smooth_seasonal_coefficients(da, temporal_sigma).values
        return np.moveaxis(smoothed, -1, 0)
    return _s(a), _s(b)


def normal_category_probs(mu_cal, sigma_cal, t_lo, t_hi):
    """Below/near/above probs from N(mu_cal, sigma_cal), tercile bounds t_lo<t_hi. Category axis first."""
    mu_cal = np.asarray(mu_cal, float); sigma_cal = np.maximum(np.asarray(sigma_cal, float), _EPS)
    below = stats.norm.cdf(t_lo, mu_cal, sigma_cal)
    above = 1.0 - stats.norm.cdf(t_hi, mu_cal, sigma_cal)
    near = np.clip(1.0 - below - above, 0.0, 1.0)
    return np.stack([below, near, above], axis=0)


def fit_gamma(x):
    """Method-of-moments two-parameter gamma fit to positive values -> (shape, scale)."""
    x = np.asarray(x, float); x = x[np.isfinite(x) & (x > 0)]
    if x.size < 3:
        return np.nan, np.nan
    m, v = x.mean(), x.var()
    if v < _EPS or m < _EPS:
        return np.nan, np.nan
    return float(m * m / v), float(v / m)


def gamma_to_normal(x, shape, scale):
    cdf = stats.gamma.cdf(np.asarray(x, float), a=shape, scale=scale)
    return stats.norm.ppf(np.clip(cdf, 1e-6, 1 - 1e-6))


def normal_to_gamma(x_hat, shape, scale):
    return stats.gamma.ppf(np.clip(stats.norm.cdf(np.asarray(x_hat, float)), 1e-6, 1 - 1e-6),
                           a=shape, scale=scale)
