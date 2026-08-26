import warnings
from dataclasses import dataclass, field

import xarray as xr

from .registry import get_metric, _METRICS


@dataclass
class SkillReport:
    scores: dict = field(default_factory=dict)
    spatial: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    diagrams: dict = field(default_factory=dict)

    def to_table(self):
        """Flat metric × scalar DataFrame. Spatial maps excluded by design.

        Returns: pandas.DataFrame with columns ['metric', 'value'], one row
        per scalar entry in self.scores (insertion order).
        """
        import pandas as pd
        return pd.DataFrame(
            [{"metric": k, "value": v} for k, v in self.scores.items()]
        )

    def to_dict(self):
        """JSON-friendly round-trippable representation.

        Spatial DataArrays serialize as {dims, coords, values} with nested
        lists. Diagram payloads (ROC, reliability) have their ndarrays
        converted to lists. Scalars pass through unchanged.
        """
        import numpy as np

        def _spatial_to_payload(da):
            if not isinstance(da, xr.DataArray):
                # Scalar metrics that returned a dict get stored as plain floats.
                return float(da) if not isinstance(da, (int, str)) else da
            return {
                "dims": list(da.dims),
                "coords": {k: da.coords[k].values.tolist() for k in da.coords},
                "values": da.values.tolist(),
            }

        def _ndarray_to_list(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _ndarray_to_list(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_ndarray_to_list(v) for v in obj]
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            return obj

        return {
            "scores": {k: (float(v) if not isinstance(v, (int, str)) else v)
                       for k, v in self.scores.items()},
            "spatial": {k: _spatial_to_payload(v) for k, v in self.spatial.items()},
            "diagrams": _ndarray_to_list(self.diagrams),
            "metadata": dict(self.metadata),
        }

    def to_geotiff(self, path, metric):
        """Write one metric's spatial map as a GeoTIFF (EPSG:4326).

        Caller loops if they want every metric on disk; one metric per call
        keeps the on-disk fan-out explicit.
        """
        from ._optional import require_optional
        require_optional("rioxarray", "pip install africas2s[plotting]")
        import rioxarray  # noqa: F401  (registers the .rio accessor on xarray)

        if metric not in self.spatial:
            if metric in self.scores:
                raise ValueError(
                    f"metric {metric!r} has no spatial map; "
                    f"computed scalar only (use to_table() or to_dict())"
                )
            raise KeyError(
                f"metric {metric!r} not in report. "
                f"Available with spatial maps: {sorted(self.spatial.keys())}"
            )

        da = self.spatial[metric]

        # rioxarray auto-detects 'x'/'y' and 'longitude'/'latitude' but not
        # 'lon'/'lat' (the africas2s convention).  Set spatial dims explicitly
        # when they aren't already detected, mapping common aliases.
        _X_ALIASES = {"lon", "longitude", "x"}
        _Y_ALIASES = {"lat", "latitude", "y"}
        try:
            da.rio.x_dim  # raises MissingSpatialDimensionError if not set
        except Exception:
            dims_lower = {str(d).lower(): d for d in da.dims}
            x_dim = next((dims_lower[a] for a in _X_ALIASES if a in dims_lower), None)
            y_dim = next((dims_lower[a] for a in _Y_ALIASES if a in dims_lower), None)
            if x_dim is not None and y_dim is not None:
                da = da.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)

        # Ensure CRS is set before writing
        da = da.rio.write_crs("EPSG:4326")
        da.rio.to_raster(str(path))

    def to_pdf(self, path, *, style="svslrf"):
        """Render the report to PDF in the requested style.

        Today only 'svslrf' is implemented. Future styles dispatch here.
        """
        if style == "svslrf":
            from .reporting.svslrf import render
            render(self, path)
            return
        raise ValueError(f"unknown style {style!r}; known: ['svslrf']")


PRESETS = {
    # WMO Standardized Verification System for Long-Range Forecasts.
    "svslrf": ["rpss", "roc", "reliability"],
    # "all" is resolved at call time so newly-registered metrics are picked up.
    # Sentinel value None means "expand dynamically from the registry".
    "all": None,
}


def _resolve_metrics(metrics):
    """Expand a `metrics=` argument into a concrete list of metric names."""
    if metrics is None:
        return ["rpss"]
    if isinstance(metrics, str):
        if metrics in PRESETS:
            preset = PRESETS[metrics]
            if preset is None:
                seen = set()
                names = []
                for name, cls in _METRICS.items():
                    if cls in seen:
                        continue
                    seen.add(cls)
                    names.append(name)
                return names
            return list(preset)
        return [metrics]
    return list(metrics)


def skill(forecast, obs, metrics=None, spatial=False, **kwargs):
    skip_incompatible = metrics == "all"
    metrics = _resolve_metrics(metrics)

    report = SkillReport()
    for name in metrics:
        metric = get_metric(name)()
        try:
            result = metric.compute(forecast, obs, spatial=spatial, **kwargs)
        except ValueError as exc:
            if not skip_incompatible:
                raise
            warnings.warn(
                f"skill(metrics='all'): skipping {name!r}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        if isinstance(result, dict):
            report.scores.update(result)
            if spatial:
                report.spatial.update(result)
        else:
            if spatial and isinstance(result, xr.DataArray):
                report.spatial[name] = result
                report.scores[name] = float(result.mean())
            else:
                report.scores[name] = result

        # Capture optional structured diagram data (ROC curves, reliability bins).
        # Metrics without compute_diagram() simply don't contribute anything.
        if hasattr(metric, "compute_diagram"):
            try:
                report.diagrams[name] = metric.compute_diagram(forecast, obs, **kwargs)
            except ValueError as exc:
                if not skip_incompatible:
                    raise
                warnings.warn(
                    f"skill(metrics='all'): skipping diagram for {name!r}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    return report
