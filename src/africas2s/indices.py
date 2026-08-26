"""Climate indices: reducing a gridded field to a scalar series.

An index is three choices — *which boxes*, *how each box is transformed*, and
*how the transformed boxes combine*. Name those three and almost every
teleconnection index in operational use falls out of the same machinery:

===========  =========================================  ===========  ==========
Index        Combine                                    Transform    Weighting
===========  =========================================  ===========  ==========
Niño3.4      ``z(nino34)``                              standardize  none
WVG          ``z(nino34) - mean(z(wnp), z(wep), z(wsp))`` standardize  none
DMI / IOD    ``a(wtio) - a(setio)``                     anomaly      cos-lat
RONI         ``a(nino34) - a(tropics)``                 anomaly      cos-lat
WIO          ``raw(wtio)``                              raw          cos-lat
===========  =========================================  ===========  ==========

The transform is the axis the module previously fixed. ``reduce()`` used to
z-score every box unconditionally, which makes RONI (a difference of *anomalies*,
in °C) and an absolute SST threshold (is the western Indian Ocean warmer than
29 °C?) inexpressible. Both are now ordinary configurations.

Weighting is the other fixed axis. A plain ``mean(["lat", "lon"])`` over-counts
high-latitude cells, which is harmless for a 10°-tall Niño box and materially
wrong for RONI's 40°-tall tropical mean. ``weights="cos_lat"`` applies the
area weighting; the default stays unweighted so the WVG family keeps matching
the operational ICPAC / WASS2S reference implementations, which are unweighted.

Nothing here is SST-specific. The boxes are just regions and the field is just
a field, so the same call extracts a standardized precipitation index over an
arbitrary West Pacific rectangle::

    Index.custom(name="wpac", regions={"wpac": [-9, 4, 103, 140]},
                 combine=lambda z: z["wpac"]).reduce(era5_precip)

Usage::

    idx_hcst = Index.named("wvg").reduce(sst_hcst)
    idx_fcst = Index.named("wvg").reduce(sst_fcst, climatology=sst_hcst)

Passing ``climatology`` puts the hindcast and forecast indices on a shared
scale. ``baseline`` narrows that reference to a fixed period (e.g. WMO's
1991-2020) without the caller having to slice it themselves.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field as _dc_field

import numpy as np
import xarray as xr

# Box definitions in degrees: (south, north, west, east). Longitudes are in the
# 0-360 convention (e.g. Nino3.4 spans 190E-240E). reduce() normalizes whatever
# convention the input field uses, so callers may pass -180..180 or 0..360 data.
REGIONS: dict[str, dict[str, float]] = {
    # --- Pacific ENSO boxes ---
    "nino12": dict(south=-10, north=0, west=270, east=280),
    "nino3": dict(south=-5, north=5, west=210, east=270),
    "nino34": dict(south=-5, north=5, west=190, east=240),
    "nino4": dict(south=-5, north=5, west=160, east=210),
    # --- Western-V Gradient boxes (Funk et al.) ---
    "wep": dict(south=-15, north=20, west=120, east=160),   # West Equatorial Pacific
    "wnp": dict(south=20, north=35, west=160, east=210),     # West North Pacific
    "wsp": dict(south=-30, north=-15, west=155, east=210),   # West South Pacific
    # --- Indian Ocean Dipole poles (Saji et al. 1999) ---
    "wtio": dict(south=-10, north=10, west=50, east=70),     # West Tropical Indian Ocean
    "setio": dict(south=-10, north=0, west=90, east=110),    # SE Tropical Indian Ocean
    # --- Tropical mean, the RONI reference band (L'Heureux et al. 2024) ---
    "tropics": dict(south=-20, north=20, west=0, east=360),
    # --- Equatorial West Pacific convection box ---
    # The region whose July rainfall discriminates wet from dry Ethiopian
    # Kiremt seasons via the Walker circulation (Funk, CHC 2026).
    "wpac": dict(south=-9, north=4, west=103, east=140),
}

# Kept as a private alias: the old name was module-internal, but pinning it here
# costs nothing and keeps any stray importer working.
_REGIONS = REGIONS

_TRANSFORMS = ("standardize", "anomaly", "raw")


# ---------------------------------------------------------------------------
# Combine functions
# ---------------------------------------------------------------------------


def _wvg3_combine(z: dict[str, xr.DataArray]) -> xr.DataArray:
    """3-box Western-V Gradient: z(Nino3.4) - mean(z(WNP), z(WEP), z(WSP))."""
    return z["nino34"] - (z["wnp"] + z["wep"] + z["wsp"]) / 3.0


def _wvg2_combine(z: dict[str, xr.DataArray]) -> xr.DataArray:
    """2-box Western-V Gradient variant: z(Nino3.4) - mean(z(WNP), z(WEP))."""
    return z["nino34"] - (z["wnp"] + z["wep"]) / 2.0


def _dmi_combine(a: dict[str, xr.DataArray]) -> xr.DataArray:
    """Dipole Mode Index: west-pole SST anomaly minus east-pole SST anomaly."""
    return a["wtio"] - a["setio"]


def _roni_combine(a: dict[str, xr.DataArray]) -> xr.DataArray:
    """Relative Oceanic Niño Index: Niño3.4 anomaly minus tropical-mean anomaly.

    Subtracting the 20°S-20°N mean removes the basin-wide warming trend, so a
    RONI of +1 °C means the same thing in 1982 and 2026 -- which a raw Niño3.4
    anomaly does not.
    """
    return a["nino34"] - a["tropics"]


@dataclass(frozen=True)
class _Spec:
    """A named index's definition, before it is bound to a field."""

    regions: tuple[str, ...]
    combine: Callable[[dict], xr.DataArray]
    transform: str = "standardize"
    weights: str | None = None
    description: str = ""


# The WVG family and the bare Niño indices keep transform="standardize" and
# weights=None so their values are unchanged from before this module grew those
# knobs. The new entries declare what they actually need.
_INDICES: dict[str, _Spec] = {
    "wvg": _Spec(
        ("nino34", "wnp", "wep", "wsp"), _wvg3_combine,
        description="Western-V Gradient, 3-box (ICPAC primary definition).",
    ),
    "wvg2": _Spec(
        ("nino34", "wnp", "wep"), _wvg2_combine,
        description="Western-V Gradient, 2-box variant.",
    ),
    "nino12": _Spec(("nino12",), lambda z: z["nino12"], description="Standardized Niño1+2."),
    "nino3": _Spec(("nino3",), lambda z: z["nino3"], description="Standardized Niño3."),
    "nino34": _Spec(("nino34",), lambda z: z["nino34"], description="Standardized Niño3.4."),
    "nino4": _Spec(("nino4",), lambda z: z["nino4"], description="Standardized Niño4."),
    "oni": _Spec(
        ("nino34",), lambda a: a["nino34"], transform="anomaly", weights="cos_lat",
        description=(
            "Niño3.4 SST anomaly, °C. The operational ONI additionally applies a "
            "3-month running mean; smooth the field or the series upstream for that."
        ),
    ),
    "roni": _Spec(
        ("nino34", "tropics"), _roni_combine, transform="anomaly", weights="cos_lat",
        description="Relative Oceanic Niño Index: Niño3.4 anomaly minus 20°S-20°N mean anomaly, °C.",
    ),
    "dmi": _Spec(
        ("wtio", "setio"), _dmi_combine, transform="anomaly", weights="cos_lat",
        description="Dipole Mode Index (Indian Ocean Dipole), °C.",
    ),
    "wtio": _Spec(
        ("wtio",), lambda a: a["wtio"], transform="anomaly", weights="cos_lat",
        description="West Tropical Indian Ocean SST anomaly, °C.",
    ),
    "setio": _Spec(
        ("setio",), lambda a: a["setio"], transform="anomaly", weights="cos_lat",
        description="Southeast Tropical Indian Ocean SST anomaly, °C.",
    ),
    "wio": _Spec(
        ("wtio",), lambda r: r["wtio"], transform="raw", weights="cos_lat",
        description=(
            "Western Indian Ocean SST in absolute units (the IOD western pole box), "
            "for thresholds such as the >29 °C criterion for extreme East African "
            "short rains."
        ),
    ),
    "wpac": _Spec(
        ("wpac",), lambda z: z["wpac"],
        description=(
            "Standardized equatorial West Pacific field mean (9°S-4°N, 103°E-140°E). "
            "Applied to precipitation, this is the Walker-circulation indicator; the "
            "module is not SST-specific."
        ),
    ),
}

# `iod` is the common name for the Dipole Mode Index.
_ALIASES = {"iod": "dmi"}


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Index:
    """A climate index that reduces a gridded field to a 1-D series.

    Construct via :meth:`named` or :meth:`custom`. Call :meth:`reduce` to turn
    a field into a per-time index series.

    Attributes
    ----------
    transform : str or mapping
        How each box series is transformed before ``combine`` sees it. One of
        ``"standardize"`` (z-score against the reference), ``"anomaly"``
        (subtract the reference mean, keeping physical units) or ``"raw"``
        (pass through untouched). A mapping keyed by region name sets it per
        box; a bare string applies to all of them.
    weights : str, xr.DataArray or None
        ``"cos_lat"`` area-weights the box mean by the cosine of latitude.
        ``None`` (the default) takes a plain mean. A DataArray is used as
        explicit weights.
    baseline : tuple or slice, optional
        Restricts the reference used by ``standardize`` and ``anomaly`` to a
        period, e.g. ``(1991, 2020)``. Ignored by ``transform="raw"``.
    """

    name: str
    regions: Mapping[str, object]
    _combine: Callable[[dict], xr.DataArray]
    transform: str | Mapping[str, str] = "standardize"
    weights: object = None
    baseline: object = None
    description: str = _dc_field(default="", compare=False)

    def __post_init__(self):
        self._validate_transform(self.transform, self.regions)

    # -- construction ------------------------------------------------------
    @staticmethod
    def _validate_transform(transform, regions) -> None:
        if isinstance(transform, str):
            if transform not in _TRANSFORMS:
                raise ValueError(
                    f"transform must be one of {_TRANSFORMS}, got {transform!r}."
                )
            return
        if not isinstance(transform, Mapping):
            raise TypeError(
                "transform must be a string or a mapping of region name -> string."
            )
        unknown = set(transform) - set(regions)
        if unknown:
            raise ValueError(
                f"transform names regions that this index does not use: "
                f"{sorted(unknown)}. Known regions: {sorted(regions)}."
            )
        bad = {k: v for k, v in transform.items() if v not in _TRANSFORMS}
        if bad:
            raise ValueError(f"transform must be one of {_TRANSFORMS}; got {bad}.")

    @classmethod
    def named(cls, name: str, **overrides) -> "Index":
        """Return a predefined index by name (e.g. ``"wvg"``, ``"roni"``, ``"dmi"``).

        ``overrides`` may set ``transform``, ``weights`` or ``baseline`` on top
        of the definition — e.g. ``Index.named("roni", baseline=(1991, 2020))``.
        """
        key = _ALIASES.get(name.lower(), name.lower())
        if key not in _INDICES:
            raise KeyError(
                f"Unknown index {name!r}. Known indices: {sorted(_INDICES)} "
                f"(aliases: {sorted(_ALIASES)})."
            )
        spec = _INDICES[key]
        return cls(
            name=key,
            regions={region: REGIONS[region] for region in spec.regions},
            _combine=spec.combine,
            transform=overrides.pop("transform", spec.transform),
            weights=overrides.pop("weights", spec.weights),
            baseline=overrides.pop("baseline", None),
            description=spec.description,
            **overrides,
        )

    @classmethod
    def custom(
        cls,
        *,
        name: str,
        regions: Mapping[str, object],
        combine: Callable[[dict[str, xr.DataArray]], xr.DataArray],
        transform: str | Mapping[str, str] = "standardize",
        weights: object = None,
        baseline: object = None,
    ) -> "Index":
        """Build a custom index from named regions and a combine function.

        ``regions`` accepts the same simple bbox convention used by Rosetta:
        ``[lat_s, lat_n, lon_w, lon_e]``, or a ``{"south": ..., "north": ...}``
        mapping, or the name of any box in :data:`REGIONS`. If Rosetta is
        importable, shapefile paths and shapely/geopandas geometries work too.

        ``combine`` receives the transformed regional series keyed by region
        name. For example, WVG is::

            lambda z: z["nino34"] - (z["wnp"] + z["wep"] + z["wsp"]) / 3

        See :class:`Index` for ``transform``, ``weights`` and ``baseline``.
        """
        if not regions:
            raise ValueError("Index.custom requires at least one region.")
        bad_names = [key for key in regions if not isinstance(key, str) or not key]
        if bad_names:
            raise ValueError("Index.custom region names must be non-empty strings.")
        resolved = {
            key: REGIONS[value] if isinstance(value, str) and value in REGIONS else value
            for key, value in regions.items()
        }
        return cls(
            name=name,
            regions=resolved,
            _combine=combine,
            transform=transform,
            weights=weights,
            baseline=baseline,
        )

    @staticmethod
    def list_named() -> dict[str, str]:
        """Map every predefined index name to its one-line description."""
        return {key: spec.description for key, spec in sorted(_INDICES.items())}

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _spatial_dims(field: xr.DataArray) -> tuple[str, str]:
        from ._spatial import spatial_dims

        return spatial_dims(field, context="Index.reduce")

    @staticmethod
    def _resolve_region(region) -> tuple[list[float], object | None]:
        if isinstance(region, Mapping):
            return [
                float(region["south"]),
                float(region["north"]),
                float(region["west"]),
                float(region["east"]),
            ], None
        try:
            seq = list(region)
        except TypeError:
            seq = None
        if seq is not None and len(seq) == 4:
            return [float(v) for v in seq], None

        try:
            from rosetta.region import resolve_region
        except ImportError as e:
            raise TypeError(
                "Index regions must be bbox-like [lat_s, lat_n, lon_w, lon_e] "
                "unless Rosetta is installed for shapefile/geometry support."
            ) from e
        return resolve_region(region)

    @staticmethod
    def _mask_geometry(sub: xr.DataArray, geometry, lat: str, lon: str) -> xr.DataArray:
        try:
            from rosetta.normalize import clip_to_geometry
        except ImportError as e:
            raise TypeError(
                "Index geometry regions require Rosetta's geometry clipping "
                "support. Use a bbox region or install Rosetta."
            ) from e
        ds = sub.to_dataset(name="_index_source")
        clipped = clip_to_geometry(ds, geometry)
        return clipped["_index_source"]

    def _resolve_weights(self, sub: xr.DataArray, lat: str) -> xr.DataArray | None:
        if self.weights is None:
            return None
        if isinstance(self.weights, xr.DataArray):
            return self.weights
        if self.weights == "cos_lat":
            return np.cos(np.deg2rad(sub[lat])).clip(min=0.0)
        raise ValueError(
            f"weights must be None, 'cos_lat' or a DataArray, got {self.weights!r}."
        )

    def _transform_for(self, region: str) -> str:
        if isinstance(self.transform, str):
            return self.transform
        return self.transform.get(region, "standardize")

    def _box_series(self, field: xr.DataArray) -> dict[str, xr.DataArray]:
        """Regional-mean of ``field`` per region, reduced over space (and member)."""
        lat, lon = self._spatial_dims(field)
        # Normalize longitude to 0-360 so the box bounds (0-360) always apply.
        field = field.assign_coords({lon: field[lon] % 360}).sortby(lon)
        reduce_dims = [lat, lon]
        if "member" in field.dims:
            reduce_dims.append("member")

        out = {}
        for name, region in self.regions.items():
            bbox, geometry = self._resolve_region(region)
            lat_s, lat_n, lon_w, lon_e = bbox
            lon_w, lon_e = lon_w % 360, lon_e % 360
            if lon_w < lon_e:
                lon_mask = (field[lon] >= lon_w) & (field[lon] <= lon_e)
            elif lon_w > lon_e:
                # Box wraps the prime meridian (e.g. 340E -> 20E).
                lon_mask = (field[lon] >= lon_w) | (field[lon] <= lon_e)
            else:
                # Equal bounds after the mod: a full 0-360 sweep, e.g. the
                # RONI tropical band. Selecting a single meridian instead would
                # silently reduce a basin mean to one column of cells.
                lon_mask = xr.ones_like(field[lon], dtype=bool)
            sub = field.where(
                (field[lat] >= lat_s) & (field[lat] <= lat_n) & lon_mask
            )
            if geometry is not None:
                sub = self._mask_geometry(sub, geometry, lat, lon)

            weights = self._resolve_weights(sub, lat)
            if weights is None:
                out[name] = sub.mean(reduce_dims, skipna=True)
            else:
                out[name] = sub.weighted(weights).mean(reduce_dims, skipna=True)
        return out

    @staticmethod
    def _select_baseline(field: xr.DataArray, baseline) -> xr.DataArray:
        """Restrict ``field`` to the baseline period along its time-like dim."""
        if baseline is None:
            return field
        for dim in ("year", "time", "init_time"):
            if dim not in field.dims:
                continue
            if isinstance(baseline, slice):
                selector = baseline
            elif dim == "year":
                selector = slice(int(baseline[0]), int(baseline[1]))
            else:
                selector = slice(str(baseline[0]), str(baseline[1]))
            selected = field.sel({dim: selector})
            if selected.sizes[dim] == 0:
                raise ValueError(
                    f"baseline {baseline!r} selects no points along {dim!r} "
                    f"(which spans {field[dim].values.min()} to "
                    f"{field[dim].values.max()})."
                )
            return selected
        raise ValueError(
            "baseline was given but the reference field has no 'year', 'time' or "
            f"'init_time' dim to slice (dims: {tuple(field.dims)})."
        )

    def _apply_transform(self, series, reference, kind):
        if kind == "raw":
            return series
        mean = reference.mean(skipna=True)
        if kind == "anomaly":
            return series - mean
        std = reference.std(skipna=True)
        std = xr.where(std < 1e-12, 1e-12, std)
        return (series - mean) / std

    # -- public ------------------------------------------------------------
    def reduce(
        self,
        field: xr.DataArray | None = None,
        climatology: xr.DataArray | None = None,
        *,
        baseline=None,
        sst: xr.DataArray | None = None,
    ) -> xr.DataArray:
        """Reduce a gridded field to the index series.

        Parameters
        ----------
        field : xr.DataArray
            A field with lat/lon dims (named lat/latitude/Y or lon/longitude/X)
            and usually a time/year dim. An optional ``member`` dim is averaged
            out. Despite the historical naming this need not be SST.
        climatology : xr.DataArray, optional
            Reference field supplying the transform's mean/std. Pass the
            hindcast when reducing a forecast year so both indices share a
            scale. Defaults to ``field`` itself. Unused when every box's
            transform is ``"raw"``.
        baseline : tuple or slice, optional
            Restricts the reference to a period, e.g. ``(1991, 2020)``.
            Overrides the index's own ``baseline``.
        sst : xr.DataArray, optional
            Deprecated alias for ``field``.

        Returns
        -------
        xr.DataArray
            The 1-D index series over the time/year dim of ``field``, or a
            scalar if it has none.
        """
        if sst is not None:
            if field is not None:
                raise TypeError("pass either `field` or the deprecated `sst`, not both")
            warnings.warn(
                "Index.reduce(sst=...) is deprecated; the argument is now `field` "
                "because the reduction is not SST-specific.",
                DeprecationWarning,
                stacklevel=2,
            )
            field = sst
        if field is None:
            raise TypeError("Index.reduce() requires a `field`")

        boxes = self._box_series(field)

        # "raw" never consults the reference, so an all-raw index (e.g. an
        # absolute SST threshold) works without a climatology even when the
        # field is a single forecast map with no time axis to average over.
        kinds = {region: self._transform_for(region) for region in boxes}
        if all(kind == "raw" for kind in kinds.values()):
            reference_boxes = boxes
        else:
            reference = field if climatology is None else climatology
            period = self.baseline if baseline is None else baseline
            reference_boxes = self._box_series(self._select_baseline(reference, period))

        transformed = {
            region: self._apply_transform(series, reference_boxes[region], kinds[region])
            for region, series in boxes.items()
        }

        index = self._combine(transformed)
        index.name = self.name
        return index
