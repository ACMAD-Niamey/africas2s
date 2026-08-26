"""Ensemble-regression (eReg) calibration engine.

Per grid cell, regress the observation on the ensemble-mean hindcast over the
training years (ordinary least squares), then apply that fit to the forecast
ensemble mean:

    obs ~ a + b * mean_members(hindcast)
    forecast_calibrated = a + b * mean_members(forecast)

This is the MOS calibration used in ICPAC's operational EnsReg stream
(``regrEns``), reduced to its core: it removes mean and amplitude bias relative
to observations, then turns the calibrated forecast into tercile probabilities
via its own prediction-error variance (``predict_tercile``).

This class is the per-model engine; the public entry point is
``africas2s.calibrate(predictor, obs, method="ereg")``, which fits it per model
and averages the per-model tercile maps across models. It is a calibrate-family
method (tercile probabilities, no resolution change), not a ``downscale``/
``seasonal_mme`` method.
"""
import numpy as np
import xarray as xr
from scipy.stats import norm

from .base import MethodBase
from .._spatial import spatial_dims


def _spatial_dims(da):
    return spatial_dims(da, context="ensemble_regression")


class EnsembleRegressionMethod(MethodBase):
    """Per-cell OLS calibration of obs on the ensemble-mean hindcast.

    Parameters
    ----------
    clip_negative : bool
        If True, clamp calibrated predictions below zero to zero (sensible for
        precipitation, the ICPAC use case). Default False to stay variable-
        agnostic, matching CCA.
    """

    def __init__(self, clip_negative=False, **_ignored):
        self.clip_negative = clip_negative

    def fit(self, hindcast, obs, **kwargs):
        gcm_mean = hindcast.mean("member") if "member" in hindcast.dims else hindcast
        hlat, hlon = _spatial_dims(gcm_mean)
        olat, olon = _spatial_dims(obs)
        gcm_mean = gcm_mean.transpose("year", hlat, hlon)
        obs = obs.transpose("year", olat, olon)

        # eReg is a per-cell calibration, so predictor and predictand must share
        # a grid (unlike CCA, which maps between grids). Fail clearly otherwise.
        if (gcm_mean.sizes[hlat], gcm_mean.sizes[hlon]) != (obs.sizes[olat], obs.sizes[olon]):
            raise ValueError(
                "ensemble_regression requires the hindcast and obs on the same "
                f"grid; got hindcast {(gcm_mean.sizes[hlat], gcm_mean.sizes[hlon])} "
                f"vs obs {(obs.sizes[olat], obs.sizes[olon])}. Regrid the GCM "
                "onto the obs grid first (e.g. gcm.interp(lat=obs.lat, lon=obs.lon))."
            )

        n_years = gcm_mean.sizes["year"]
        X = gcm_mean.values.reshape(n_years, -1)
        Y = obs.values.reshape(n_years, -1)
        ncell = X.shape[1]

        slope = np.full(ncell, np.nan)
        intercept = np.full(ncell, np.nan)
        pev = np.full(ncell, np.nan)
        x_mean = np.full(ncell, np.nan)      # training predictor mean per cell
        sxx = np.full(ncell, np.nan)         # sum of squared predictor deviations
        n_eff = np.full(ncell, np.nan)       # finite paired years per cell

        # Per-cell OLS via closed form, NaN-aware. Cells with <3 paired finite
        # years or a constant predictor stay NaN (uncalibratable). We also store
        # the predictor mean / Sxx / n so predict_tercile can inflate the
        # residual variance for parameter-estimation uncertainty (Wilks 2006
        # eq 6.22): sigma^2 = pev * (1 + 1/n + (xf - xbar)^2 / Sxx).
        for g in range(ncell):
            xg, yg = X[:, g], Y[:, g]
            ok = np.isfinite(xg) & np.isfinite(yg)
            if ok.sum() < 3:
                continue
            xo, yo = xg[ok], yg[ok]
            xbar, ybar = xo.mean(), yo.mean()
            sxx_g = np.sum((xo - xbar) ** 2)
            if sxx_g < 1e-12:
                continue
            b = np.sum((xo - xbar) * (yo - ybar)) / sxx_g
            a = ybar - b * xbar
            slope[g] = b
            intercept[g] = a
            resid = yo - (a + b * xo)
            dof = max(ok.sum() - 2, 1)
            pev[g] = float(np.sum(resid ** 2) / dof)
            x_mean[g] = xbar
            sxx[g] = sxx_g
            n_eff[g] = ok.sum()

        shape = obs.isel(year=0).shape
        self.slope_ = slope.reshape(shape)
        self.intercept_ = intercept.reshape(shape)
        self.pev_ = pev.reshape(shape)
        self.x_mean_ = x_mean.reshape(shape)
        self.sxx_ = sxx.reshape(shape)
        self.n_eff_ = n_eff.reshape(shape)
        self.predictor_shape_ = gcm_mean.isel(year=0).shape
        self.lat_dim_ = olat
        self.lon_dim_ = olon
        self.predictand_coords_ = {olat: obs[olat], olon: obs[olon]}
        self.predictand_shape_ = shape
        self.n_train_ = n_years
        # Stash the (member-reduced) training predictor so the fitted-threshold
        # hindcast can be built lazily (see the fitted_hindcast_ property); it is
        # unused on the default threshold_source='obs' path.
        self._fit_gcm_mean_ = gcm_mean
        return self

    @property
    def fitted_hindcast_(self):
        """Calibrated hindcast over the training years, dims ``(year, lat, lon)``.

        Used only by ``predict_tercile(threshold_source="fitted")``. Computed
        lazily on first access and cached, so the default ``"obs"`` path doesn't
        pay for a full-grid broadcast on every fit/CV-fold. Computed WITHOUT
        ``clip_negative``: clipping is a deterministic-output floor, and applying
        it here would pile mass at zero and distort the climatological tercile
        boundaries.
        """
        cached = self.__dict__.get("_fitted_hindcast_cache")
        if cached is None:
            x = self._fit_gcm_mean_.values
            pred = self.slope_[None, ...] * x + self.intercept_[None, ...]
            cached = xr.DataArray(
                pred,
                dims=["year", self.lat_dim_, self.lon_dim_],
                coords={"year": self._fit_gcm_mean_["year"], **self.predictand_coords_},
            )
            self.__dict__["_fitted_hindcast_cache"] = cached
        return cached

    def predict(self, forecast, **kwargs):
        if "member" in forecast.dims:
            fc = forecast.mean("member")
        else:
            fc = forecast
        flat, flon = _spatial_dims(fc)
        has_year = "year" in fc.dims
        if has_year:
            fc = fc.transpose("year", flat, flon)
        else:
            fc = fc.transpose(flat, flon)

        grid = (fc.sizes[flat], fc.sizes[flon])
        if grid != self.predictor_shape_:
            raise ValueError(
                f"forecast grid shape {grid} does not match the training "
                f"predictor grid shape {self.predictor_shape_}."
            )

        x = fc.values
        pred = self.slope_ * x + self.intercept_  # broadcasts over leading year
        if self.clip_negative:
            pred = np.where(np.isfinite(pred) & (pred < 0), 0.0, pred)

        if has_year:
            return xr.DataArray(
                pred, dims=["year", self.lat_dim_, self.lon_dim_],
                coords={"year": fc.year, **self.predictand_coords_},
            )
        return xr.DataArray(
            pred, dims=[self.lat_dim_, self.lon_dim_], coords=self.predictand_coords_,
        )

    def predict_tercile(self, forecast, obs_climatology, threshold_source="obs"):
        """Single-year tercile probabilities from THIS model's calibrated Gaussian.

        The forecast distribution is ``N(calibrated_mean, sqrt(sigma2))`` per
        cell, where ``sigma2`` is the prediction-error variance (Wilks 2006
        eq 6.22), the ICPAC-compatible calibrated variance:

            sigma2 = residual_var * (1 + 1/n + (x_forecast - x_mean)^2 / Sxx)

        The ``(1 + leverage)`` factor inflates the residual variance for
        parameter-estimation uncertainty, widening the distribution when the
        forecast predictor extrapolates beyond the training range. This is the
        same convention africas2s uses for CCA terciles. (ICPAC's full
        ``sigma.f`` adds smaller ensemble-sampling terms on top; omitted here.)

        The spread is eReg's OWN calibrated error, not inter-model spread.
        Boundaries are the climatological terciles of ``obs_climatology`` by
        default. With ``threshold_source="fitted"``, boundaries are computed
        from the calibrated hindcast over the training years (the regression's
        natural, un-clipped scale — see ``fitted_hindcast_``), matching ICPAC's
        legacy EnsReg convention. ``clip_negative`` floors the deterministic
        forecast value but deliberately does not move the tercile boundaries.
        ``seasonal_mme`` calls this per model and averages the resulting maps,
        so the published forecast carries both eReg's within-model uncertainty
        (via ``sigma``) and between-model disagreement (via the average over
        models). Returns ``(tercile, lat, lon)``.
        """
        xf = forecast.mean("member") if "member" in forecast.dims else forecast
        if "year" in xf.dims:
            if xf.sizes["year"] != 1:
                raise ValueError(
                    "predict_tercile expects a single forecast year; got "
                    f"{xf.sizes['year']}."
                )
            xf = xf.isel(year=0, drop=True)
        flat, flon = _spatial_dims(xf)
        xf = xf.transpose(flat, flon)
        grid = (xf.sizes[flat], xf.sizes[flon])
        if grid != self.predictor_shape_:
            raise ValueError(
                f"forecast grid shape {grid} does not match the training "
                f"predictor grid shape {self.predictor_shape_}."
            )
        xf_v = xf.values

        mu = self.slope_ * xf_v + self.intercept_
        if self.clip_negative:
            mu = np.where(np.isfinite(mu) & (mu < 0), 0.0, mu)

        # Leverage-inflated prediction-error variance (Wilks 2006 eq 6.22).
        with np.errstate(invalid="ignore", divide="ignore"):
            leverage = 1.0 / self.n_eff_ + (xf_v - self.x_mean_) ** 2 / self.sxx_
            sigma2 = self.pev_ * (1.0 + leverage)
        sigma = np.sqrt(np.maximum(sigma2, 1e-12))

        if threshold_source == "obs":
            threshold_base = obs_climatology
        elif threshold_source == "fitted":
            threshold_base = self.fitted_hindcast_
        else:
            raise ValueError(
                "threshold_source must be 'obs' or 'fitted'; got "
                f"{threshold_source!r}."
            )
        t33 = threshold_base.quantile(1 / 3, dim="year").drop_vars("quantile")
        t67 = threshold_base.quantile(2 / 3, dim="year").drop_vars("quantile")

        p_bn = norm.cdf(t33.values, loc=mu, scale=sigma)
        p_an = 1.0 - norm.cdf(t67.values, loc=mu, scale=sigma)
        p_nn = 1.0 - p_bn - p_an

        out = xr.concat(
            [xr.DataArray(p_bn, dims=[self.lat_dim_, self.lon_dim_], coords=self.predictand_coords_),
             xr.DataArray(p_nn, dims=[self.lat_dim_, self.lon_dim_], coords=self.predictand_coords_),
             xr.DataArray(p_an, dims=[self.lat_dim_, self.lon_dim_], coords=self.predictand_coords_)],
            dim="tercile",
        )
        out["tercile"] = [0, 1, 2]
        return out
