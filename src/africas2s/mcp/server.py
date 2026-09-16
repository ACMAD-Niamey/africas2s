"""africas2s MCP server: downscaling, calibration, ensembles, verification.

Design rules (keep these when adding tools):

* Tools are thin wrappers over the public verbs. Behaviour lives in the
  library; the server only loads NetCDF inputs, calls the verb, and writes
  the result back out.
* Arrays never cross the wire. Every gridded input is a path to a NetCDF file
  (one data variable, or ``variable=`` to pick one) and every gridded output
  is written under ``AFRICAS2S_MCP_WORKDIR`` (default ``~/.africas2s/mcp``)
  and returned as a path plus a compact summary. Scalars (skill scores,
  weights, chosen method) come back inline.
* Every parameter carries a schema-level description and closed vocabularies
  (methods, calibrators, strategies, metrics, CV schemes) are ``Literal``
  enums built from the registries at import time, so the schema is always in
  sync with what the library accepts. Every tool documents what it returns
  and shows one example.
* Library exceptions are re-raised as ``ToolError`` so the agent sees the
  message. The MCP SDK hides the text of any other exception.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field
from typing_extensions import NotRequired, TypedDict

import africas2s
from africas2s import cv as _cv
from africas2s import registry
from africas2s.tercile import to_tercile as _to_tercile

_HERE = Path(__file__).resolve().parent
# Source checkouts carry the Agent Skill at <repo>/skills/africas2s; wheels do
# not, so the resources degrade gracefully when the directory is absent.
_SKILL_DIR = _HERE.parents[2] / "skills" / "africas2s"

_INSTRUCTIONS = """\
africas2s turns coarse GCM seasonal hindcasts plus fine-resolution observations
into calibrated forecasts (continuous fields or below/normal/above tercile
probabilities) and scores them with cross-validated skill metrics.

Data conventions (get these right first):
  GCM hindcast      (year, member, lat, lon)   year = consecutive integers
  GCM forecast      (member, lat, lon)
  observations      (year, lat, lon)           fine grid, the predictand
  tercile forecast  (tercile, lat, lon)        tercile=[0,1,2] = below/normal/above
  CV hindcast       (year, tercile, lat, lon)

Every gridded input is a path to a NetCDF file (acmaddl-mcp fetch with
year_index=true produces the hindcast shape directly). Every gridded output
is written to a NetCDF file and its path returned; use describe_dataset to
inspect any file. Methods, calibrators, ensemble strategies, and metrics are
selected by name and the tool schemas list the valid names.

Typical flow: optimize (pick a method by CV skill) or downscale -> skill ->
plot_terciles. Rules: tercile metrics such as rpss need tercile probabilities
(downscale output_type="tercile", or to_tercile); calibrate does not regrid,
so put the GCM on the obs grid first; downscale methods regrid themselves.

Read the africas2s://skill resource for the full API and the statistical
discipline rules (tercile leakage, grid rule, CV requirements).
"""

mcp = MCPServer(
    name="africas2s",
    title="AfricaS2S seasonal forecast downscaling and verification",
    instructions=_INSTRUCTIONS,
    version=getattr(africas2s, "__version__", ""),
)


# --------------------------------------------------------------------------
# vocabularies from the registries (populated by ``import africas2s``)
# --------------------------------------------------------------------------

_PRESETS: dict = importlib.import_module("africas2s.skill").PRESETS

Method = Literal[tuple(sorted(registry._METHODS))]
Calibrator = Literal[tuple(sorted(registry._CALIBRATORS))]
Strategy = Literal[tuple(sorted(registry._STRATEGIES))]
MetricName = Literal[tuple(sorted(registry._METRICS))]
MetricPreset = Literal[tuple(sorted(_PRESETS))]
CVScheme = Literal[tuple(sorted(_cv._REGISTRY))]
OutputType = Literal["continuous", "tercile"]
TercileMethod = Literal["counting", "gaussian"]
VariableKind = Literal["precip", "temp"]


# --------------------------------------------------------------------------
# typed results (become the tools' output schemas)
# --------------------------------------------------------------------------

class CoordSummary(TypedDict, total=False):
    size: int
    dtype: str
    min: float
    max: float
    step: float
    first: str
    last: str


class VariableSummary(TypedDict, total=False):
    dims: list[str]
    shape: list[int]
    dtype: str
    units: str | None
    nan_fraction: float
    min: float
    max: float


class DatasetSummary(TypedDict):
    """What every gridded output returns: where the file is and what is in it."""
    dims: dict[str, int]
    coords: dict[str, CoordSummary]
    variables: dict[str, VariableSummary]
    attrs: dict[str, Any]
    path: NotRequired[str]
    size_bytes: NotRequired[int]
    request: NotRequired[dict[str, Any]]


class OptimizeOut(DatasetSummary):
    method: str
    score: float | None
    primary_metric: str


class EnsembleOut(DatasetSummary):
    weights: list[float]
    member_names: list[str]
    member_cv_skill: dict[str, Any]
    effective_n: float
    gate_passed: bool
    shrinkage_lambda: float
    safeguards_applied: dict[str, Any]
    pev: NotRequired[DatasetSummary]


class SkillOut(TypedDict):
    scores: dict[str, Any]
    metadata: dict[str, Any]
    diagrams: NotRequired[dict[str, Any]]
    spatial_path: NotRequired[str]
    spatial: NotRequired[DatasetSummary]


class PlotOut(TypedDict):
    path: str
    format: Literal["png"]
    request: dict[str, Any]


class RegistryOut(TypedDict):
    methods: list[str]
    calibrators: list[str]
    strategies: list[str]
    metrics: list[str]
    metric_aliases: dict[str, list[str]]
    metric_presets: dict[str, list[str] | None]
    cv_schemes: list[str]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def tool(*, read_only: bool = False, idempotent: bool = True, open_world: bool = False,
         destructive: bool = False, **kwargs):
    """``mcp.tool`` with client-facing annotations and a dedented docstring."""
    annotations = ToolAnnotations(read_only_hint=read_only, destructive_hint=destructive,
                                  idempotent_hint=idempotent, open_world_hint=open_world)

    def decorate(fn):
        description = kwargs.pop("description", None) or inspect.cleandoc(fn.__doc__ or "")
        return mcp.tool(annotations=annotations, description=description, **kwargs)(fn)

    return decorate


def workdir() -> Path:
    root = os.environ.get("AFRICAS2S_MCP_WORKDIR") or (Path.home() / ".africas2s" / "mcp")
    path = Path(root).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_safe(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, float) and obj != obj:  # NaN
        return None
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _coord_summary(coord) -> CoordSummary:
    import numpy as np

    vals = coord.values
    out: CoordSummary = {"size": int(vals.size), "dtype": str(vals.dtype)}
    if vals.size == 0:
        return out
    flat = vals.ravel()
    if np.issubdtype(vals.dtype, np.datetime64):
        out["first"] = str(np.datetime_as_string(flat[0], unit="D"))
        out["last"] = str(np.datetime_as_string(flat[-1], unit="D"))
    elif np.issubdtype(vals.dtype, np.number):
        out["min"] = float(np.nanmin(flat))
        out["max"] = float(np.nanmax(flat))
        if vals.size > 1:
            step = np.diff(flat[: min(flat.size, 3)])
            if step.size and np.all(step == step[0]):
                out["step"] = float(step[0])
    else:
        out["first"] = str(flat[0])
        out["last"] = str(flat[-1])
    return out


def summarize(ds, *, path: str | Path | None = None, max_stats_elements: int = 20_000_000) -> DatasetSummary:
    """Compact, JSON-safe description of a Dataset / DataArray."""
    import numpy as np
    import xarray as xr

    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset(name=ds.name or "data")
    out: DatasetSummary = {
        "dims": {k: int(v) for k, v in ds.sizes.items()},
        "coords": {k: _coord_summary(ds.coords[k]) for k in ds.coords},
        "variables": {},
        "attrs": _json_safe(dict(ds.attrs)),
    }
    if path is not None:
        out["path"] = str(path)
        try:
            out["size_bytes"] = os.path.getsize(path)
        except OSError:
            pass
    for name, da in ds.data_vars.items():
        units = da.attrs.get("units")
        entry: VariableSummary = {
            "dims": list(da.dims),
            "shape": [int(s) for s in da.shape],
            "dtype": str(da.dtype),
            "units": None if units is None else str(units),
        }
        if da.size and da.size <= max_stats_elements and np.issubdtype(da.dtype, np.number):
            vals = da.values
            finite = np.isfinite(vals)
            entry["nan_fraction"] = float(1 - finite.mean())
            if finite.any():
                entry["min"] = float(vals[finite].min())
                entry["max"] = float(vals[finite].max())
        out["variables"][str(name)] = entry
    return out


def _open(path: str):
    import xarray as xr

    p = Path(path).expanduser()
    if not p.exists():
        raise ToolError(f"No such file: {p}")
    try:
        return xr.open_dataset(p)
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"Could not open {p} as NetCDF: {exc}") from exc


def load_array(path: str, variable: str | None = None):
    """Load one DataArray from a NetCDF file into memory.

    Picks ``variable`` if given, else the file's only data variable; a file
    with several variables and no ``variable`` is an error that lists them.
    """
    with _open(path) as ds:
        names = list(ds.data_vars)
        if variable is None:
            if len(names) != 1:
                raise ToolError(
                    f"{path} has {len(names)} data variables {names}; pass variable=."
                )
            variable = names[0]
        if variable not in ds.data_vars:
            raise ToolError(f"{path} has no variable {variable!r}; available: {names}")
        da = ds[variable].load()
    da.name = variable
    return da


def _netcdf_safe(obj):
    """Cast object-dtype coords/variables to unicode so netCDF4 can write them."""
    import xarray as xr

    if isinstance(obj, xr.DataArray):
        name = obj.name or "data"
        obj = obj.to_dataset(name=name)
    for name in list(obj.coords) + list(obj.data_vars):
        if obj[name].dtype == object:
            obj[name] = obj[name].astype(str)
    return obj


def _slug(*parts: Any) -> str:
    text = "_".join(str(p) for p in parts if p not in (None, ""))
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in text).strip("-")


def _output_path(destination: str | None, *parts: Any, params: dict, suffix: str = ".nc") -> str:
    if destination:
        p = Path(destination).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)
    digest = hashlib.sha1(json.dumps(_json_safe(params), sort_keys=True).encode()).hexdigest()[:8]
    return str(workdir() / f"{_slug(*parts)}_{digest}{suffix}")


def _write(da, name: str, destination: str | None, *parts: Any, params: dict) -> DatasetSummary:
    """Write a DataArray as NetCDF and return its path + summary."""
    out_path = _output_path(destination, *parts, params=params)
    da = da.rename(name) if da.name != name else da
    try:
        ds = _netcdf_safe(da)
        ds.to_netcdf(out_path)
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc) from exc
    summary = summarize(ds, path=out_path)
    summary["request"] = _json_safe(params)
    return summary


def _wrap(exc: Exception) -> ToolError:
    if isinstance(exc, ToolError):
        return exc
    return ToolError(f"{type(exc).__name__}: {exc}")


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc) from exc


# --------------------------------------------------------------------------
# shared parameter annotations
# --------------------------------------------------------------------------

ObsPathArg = Annotated[str, Field(
    description="NetCDF path of the observations (predictand): (year, lat, lon) on the fine "
                "target grid, year = consecutive integers.")]
HindcastPathArg = Annotated[str, Field(
    description="NetCDF path of the GCM hindcast: (year, member, lat, lon), same years as obs.")]
VariableArg = Annotated[str | None, Field(
    description="Data variable to read from the forecast/hindcast files when they hold more than one.")]
ObsVariableArg = Annotated[str | None, Field(
    description="Data variable to read from the obs file when it holds more than one.")]
OptionsArg = Annotated[dict[str, Any] | None, Field(
    description="Extra keyword arguments forwarded to the library verb (method-specific, e.g. "
                "{\"n_modes\": 3} for cca).")]
DestinationArg = Annotated[str | None, Field(
    description="Explicit output path. Default: a stable name derived from the request under "
                "AFRICAS2S_MCP_WORKDIR, so repeating a request reuses the file.")]


# --------------------------------------------------------------------------
# introspection
# --------------------------------------------------------------------------

@tool(read_only=True)
def list_registry() -> RegistryOut:
    """Names accepted by the other tools: downscaling methods, calibrators,
    ensemble strategies, skill metrics (with aliases and presets), and
    cross-validation schemes. The tool schemas already enumerate these; call
    this when you want them with their aliases in one place.

    Returns: {methods, calibrators, strategies, metrics, metric_aliases,
    metric_presets, cv_schemes}.
    """
    by_class: dict[str, list[str]] = {}
    for name, cls in registry._METRICS.items():
        by_class.setdefault(cls.__name__, []).append(name)
    return {
        "methods": sorted(registry._METHODS),
        "calibrators": sorted(registry._CALIBRATORS),
        "strategies": sorted(registry._STRATEGIES),
        "metrics": sorted(registry._METRICS),
        "metric_aliases": {v[0]: v[1:] for v in by_class.values() if len(v) > 1},
        "metric_presets": dict(_PRESETS),
        "cv_schemes": sorted(_cv._REGISTRY),
    }


@tool(read_only=True)
def describe_dataset(
    path: Annotated[str, Field(description="Path to a NetCDF file, e.g. one returned by another tool.")],
) -> DatasetSummary:
    """Summarize a NetCDF file without loading it into your context: dims,
    coordinate ranges, variables with units and NaN fraction, attributes.

    Use it to confirm a file has the shape a tool expects (see the data
    conventions in the server instructions) before calling that tool.
    Returns: {path, dims, coords, variables, attrs, size_bytes}.
    Example: describe_dataset(path="hindcast.nc")
    """
    with _open(path) as ds:
        return summarize(ds, path=Path(path).expanduser())


# --------------------------------------------------------------------------
# downscale / optimize / calibrate
# --------------------------------------------------------------------------

@tool()
def downscale(
    predictor_hindcast_path: HindcastPathArg,
    obs_path: ObsPathArg,
    method: Annotated[Method, Field(
        description="Downscaling / bias-correction method. bcsd and cca are the usual choices; "
                    "climatology is the no-skill baseline; corrdiff needs a GPU.")] = "bcsd",
    output_type: Annotated[OutputType, Field(
        description='"continuous" (a field in obs units) or "tercile" (below/normal/above '
                    'probabilities, needed for rpss / roc).')] = "continuous",
    forecast_path: Annotated[str | None, Field(
        description="NetCDF path of the (member, lat, lon) forecast to predict. When omitted the "
                    "last hindcast year is held out and predicted (a quick sanity check).")] = None,
    predictor_variable: VariableArg = None,
    obs_variable: ObsVariableArg = None,
    options: OptionsArg = None,
    destination: DestinationArg = None,
) -> DatasetSummary:
    """Bias-correct and downscale a GCM hindcast/forecast onto the observation
    grid. Downscale methods regrid coarse to fine themselves; the output is on
    the obs grid.

    Returns: {path, dims, coords, variables, attrs, size_bytes, request}; the
    file holds "probability" (tercile, lat, lon) for tercile output or the obs
    variable name for continuous output.
    Example: downscale(predictor_hindcast_path="cfsv2_mam.nc", obs_path="chirps_mam.nc",
    method="cca", output_type="tercile", forecast_path="cfsv2_2025.nc",
    options={"n_modes": 3})
    """
    hind = load_array(predictor_hindcast_path, predictor_variable)
    obs = load_array(obs_path, obs_variable)
    kwargs = dict(options or {})
    kwargs.setdefault("verbose", False)
    if forecast_path:
        kwargs["forecast"] = load_array(forecast_path, predictor_variable)
    result = _call(africas2s.downscale, hind, obs, method=method, output_type=output_type, **kwargs)
    params = dict(predictor_hindcast_path=predictor_hindcast_path, obs_path=obs_path,
                  method=method, output_type=output_type, forecast_path=forecast_path,
                  options=options)
    name = "probability" if output_type == "tercile" else (obs.name or "forecast")
    return _write(result, name, destination, "downscale", method, output_type, params=params)


@tool()
def optimize(
    predictor_hindcast_path: HindcastPathArg,
    obs_path: ObsPathArg,
    methods: Annotated[list[Method] | None, Field(
        description='Candidate methods to compare. Default ["bcsd", "cca"].')] = None,
    cv: Annotated[CVScheme, Field(
        description='Cross-validation scheme: "loyo" leave-one-year-out (default), "lko", '
                    '"blocked", "expanding".')] = "loyo",
    primary_metric: Annotated[MetricName, Field(
        description="Metric that decides the winner (computed on CV tercile hindcasts).")] = "rpss",
    predictor_variable: VariableArg = None,
    obs_variable: ObsVariableArg = None,
    options: OptionsArg = None,
    destination: DestinationArg = None,
) -> OptimizeOut:
    """Try several downscaling methods under cross-validation and keep the most
    skillful. This is the honest way to choose a method: every score is on
    held-out years. Cost is roughly n_methods x n_years downscale fits.

    Returns: the winner's forecast file summary plus {method, score,
    primary_metric}.
    Example: optimize(predictor_hindcast_path="cfsv2_mam.nc", obs_path="chirps_mam.nc",
    methods=["bcsd", "cca", "qm"], cv="loyo", primary_metric="rpss")
    """
    hind = load_array(predictor_hindcast_path, predictor_variable)
    obs = load_array(obs_path, obs_variable)
    kwargs = dict(options or {})
    kwargs.setdefault("verbose", False)
    kwargs.setdefault("progress", False)
    res = _call(africas2s.optimize, hind, obs, methods=methods, cv=cv,
                primary_metric=primary_metric, **kwargs)
    params = dict(predictor_hindcast_path=predictor_hindcast_path, obs_path=obs_path,
                  methods=methods, cv=cv, primary_metric=primary_metric, options=options)
    out = _write(res.forecast, "forecast", destination, "optimize", res.method, params=params)
    out.update(method=res.method, score=_json_safe(res.score), primary_metric=primary_metric)
    return out


@tool()
def calibrate(
    obs_path: ObsPathArg,
    method: Annotated[Calibrator, Field(
        description='Calibration method: "ereg" ensemble regression (gridded), "logit" logistic '
                    'index calibration, "smoothed_regression" season-aware Kharin et al. (2017).')] = "ereg",
    hindcast_path: Annotated[str | None, Field(
        description="Single model: NetCDF (year, member, lat, lon) hindcast already on the obs grid.")] = None,
    forecast_path: Annotated[str | None, Field(
        description="Single model: NetCDF (member, lat, lon) forecast already on the obs grid.")] = None,
    models: Annotated[dict[str, list[str]] | None, Field(
        description='Several models: {"name": [hindcast_path, forecast_path], ...}. Their calibrated '
                    'maps are averaged. Use instead of hindcast_path/forecast_path.')] = None,
    output_type: Annotated[Literal["tercile", "deterministic"], Field(
        description='"tercile" probabilities (default) or "deterministic" calibrated field '
                    '(ereg and smoothed_regression only).')] = "tercile",
    forecast_year: Annotated[int | None, Field(
        description="Year the forecast is for. ereg needs it to place the forecast on the fitted "
                    "trend when the forecast file has no year coordinate.")] = None,
    cv: Annotated[Literal["loyo"] | None, Field(
        description='"loyo" returns the leave-year-out cross-validated hindcast (year, tercile, '
                    'lat, lon) for skill scoring instead of the forecast. ereg/logit, tercile only.')] = None,
    cv_window: Annotated[int, Field(
        description="Years left out per CV fold (1 = strict leave-one-out; 5 matches PyCPT).")] = 1,
    variable: VariableArg = None,
    obs_variable: ObsVariableArg = None,
    options: OptionsArg = None,
    destination: DestinationArg = None,
) -> DatasetSummary:
    """Calibrate gridded predictors into tercile probabilities without changing
    resolution. calibrate does NOT regrid: the hindcast and forecast must
    already be on the obs grid (interpolate the GCM first, or use downscale).

    Returns: {path, dims, coords, variables, attrs, size_bytes, request}; the
    file holds "probability" (tercile, lat, lon), or (year, tercile, lat, lon)
    under cv="loyo".
    Example: calibrate(obs_path="chirps_mam.nc", method="ereg",
    hindcast_path="cfsv2_on_obs_grid.nc", forecast_path="cfsv2_2025_on_obs_grid.nc",
    forecast_year=2025)
    """
    obs = load_array(obs_path, obs_variable)
    if models:
        predictor = {}
        for name, paths in models.items():
            if not isinstance(paths, (list, tuple)) or len(paths) != 2:
                raise ToolError(f"models[{name!r}] must be [hindcast_path, forecast_path].")
            predictor[name] = (load_array(paths[0], variable), load_array(paths[1], variable))
    elif hindcast_path and forecast_path:
        predictor = (load_array(hindcast_path, variable), load_array(forecast_path, variable))
    elif hindcast_path and cv:
        # CV mode never uses the forecast; reuse the hindcast's last year as a placeholder.
        hind = load_array(hindcast_path, variable)
        predictor = (hind, hind.isel(year=-1, drop=True))
    else:
        raise ToolError("Pass hindcast_path + forecast_path, or models={...}, "
                        "or hindcast_path with cv='loyo'.")
    kwargs = dict(options or {})
    result = _call(africas2s.calibrate, predictor, obs, method=method, output_type=output_type,
                   forecast_year=forecast_year, cv=cv, cv_window=cv_window, **kwargs)
    params = dict(obs_path=obs_path, method=method, hindcast_path=hindcast_path,
                  forecast_path=forecast_path, models=models, output_type=output_type,
                  forecast_year=forecast_year, cv=cv, cv_window=cv_window, options=options)
    name = "probability" if output_type == "tercile" else "forecast"
    return _write(result, name, destination, "calibrate", method, cv or "forecast", params=params)


# --------------------------------------------------------------------------
# ensemble / verification
# --------------------------------------------------------------------------

@tool()
def ensemble(
    forecast_paths: Annotated[list[str], Field(
        description="NetCDF paths of the member forecasts, all on the same grid with the same "
                    "dims. File stems become the member names.", min_length=1)],
    obs_path: Annotated[str | None, Field(
        description="Observations (year, lat, lon). Required for skill-based weighting and for "
                    "the prediction-error-variance diagnostic; optional for uniform.")] = None,
    strategy: Annotated[Strategy, Field(
        description='Combination strategy: "uniform" equal weights, "skill_weighted", "bma", '
                    '"drop_worst".')] = "uniform",
    optimize_ensemble: Annotated[bool, Field(
        description="Fit the weights by cross-validated skill (needs obs). Falls back to uniform "
                    "with a warning if the skill gate fails.")] = False,
    primary_metric: Annotated[MetricName, Field(
        description="Metric used to weight members when optimizing.")] = "rpss",
    variable: VariableArg = None,
    obs_variable: ObsVariableArg = None,
    options: OptionsArg = None,
    destination: DestinationArg = None,
) -> EnsembleOut:
    """Combine several forecasts into one multi-model ensemble.

    Returns: the combined forecast file summary plus {weights, member_names,
    member_cv_skill, effective_n, gate_passed, shrinkage_lambda,
    safeguards_applied} and, when obs are given, "pev" (a file of per-cell
    prediction error variance).
    Example: ensemble(forecast_paths=["cfsv2_bcsd.nc", "ecmwf_bcsd.nc"],
    obs_path="chirps_mam.nc", strategy="skill_weighted", optimize_ensemble=true)
    """
    if not forecast_paths:
        raise ToolError("forecast_paths must not be empty.")
    forecasts = [load_array(p, variable) for p in forecast_paths]
    for da, p in zip(forecasts, forecast_paths):
        da.name = Path(p).stem
    obs = load_array(obs_path, obs_variable) if obs_path else None
    kwargs = dict(options or {})
    res = _call(africas2s.ensemble, forecasts, obs, strategy=strategy,
                optimize_ensemble=optimize_ensemble, primary_metric=primary_metric, **kwargs)
    params = dict(forecast_paths=forecast_paths, obs_path=obs_path, strategy=strategy,
                  optimize_ensemble=optimize_ensemble, primary_metric=primary_metric,
                  options=options)
    out = _write(res.forecast, "forecast", destination, "ensemble", strategy, params=params)
    out.update(
        weights=_json_safe(res.weights),
        member_names=_json_safe(res.member_names),
        member_cv_skill=_json_safe(res.member_cv_skill),
        effective_n=_json_safe(res.effective_n),
        gate_passed=bool(res.gate_passed),
        shrinkage_lambda=_json_safe(res.shrinkage_lambda),
        safeguards_applied=_json_safe(res.safeguards_applied),
    )
    if res.pev is not None:
        out["pev"] = _write(res.pev, "pev", None, "ensemble", strategy, "pev", params=params)
    return out


@tool()
def skill(
    forecast_path: Annotated[str, Field(
        description="NetCDF path of the forecast to score. Tercile metrics (rpss, roc, reliability, "
                    "...) need (year, tercile, lat, lon) CV probabilities; deterministic metrics "
                    "(pearson_r, rmse, ...) take (year, lat, lon) fields on the obs grid.")],
    obs_path: ObsPathArg,
    metrics: Annotated[list[MetricName] | MetricPreset | None, Field(
        description='Metric names, or a preset: "svslrf" (rpss + roc + reliability, the WMO '
                    'standard) or "all". Default ["rpss"].')] = None,
    spatial: Annotated[bool, Field(
        description="Also compute per-cell skill maps and write them to NetCDF.")] = False,
    include_diagrams: Annotated[bool, Field(
        description="Return ROC / reliability curve data inline (can be large).")] = False,
    forecast_variable: VariableArg = None,
    obs_variable: ObsVariableArg = None,
    options: OptionsArg = None,
    destination: DestinationArg = None,
) -> SkillOut:
    """Score a forecast against observations with the named metrics. Scores are
    honest only if the forecast is a cross-validated hindcast (from
    downscale/optimize/calibrate with cv), never a fit on the same years.

    Returns: {scores: {metric: value}, metadata} plus "spatial_path" and
    "spatial" (a file summary) when spatial=true, and "diagrams" on request.
    Example: skill(forecast_path="cv_terciles.nc", obs_path="chirps_mam.nc",
    metrics="svslrf", spatial=true)
    """
    fc = load_array(forecast_path, forecast_variable)
    obs = load_array(obs_path, obs_variable)
    kwargs = dict(options or {})
    report = _call(africas2s.skill, fc, obs, metrics=metrics, spatial=spatial, **kwargs)
    payload = report.to_dict()
    out: SkillOut = {
        "scores": _json_safe(payload.get("scores", {})),
        "metadata": _json_safe(payload.get("metadata", {})),
    }
    if include_diagrams:
        out["diagrams"] = _json_safe(payload.get("diagrams", {}))
    if spatial and report.spatial:
        import xarray as xr

        maps = {k: v for k, v in report.spatial.items() if isinstance(v, xr.DataArray)}
        if maps:
            params = dict(forecast_path=forecast_path, obs_path=obs_path, metrics=metrics)
            out_path = _output_path(destination, "skill", Path(forecast_path).stem, params=params)
            try:
                xr.Dataset(maps).to_netcdf(out_path)
            except Exception as exc:  # noqa: BLE001
                raise _wrap(exc) from exc
            out["spatial_path"] = out_path
            out["spatial"] = summarize(xr.Dataset(maps), path=out_path)
    return out


@tool()
def to_tercile(
    forecast_path: Annotated[str, Field(
        description="NetCDF path of a single-year ensemble forecast (member, lat, lon) on the obs grid.")],
    obs_path: Annotated[str, Field(
        description="Observation climatology (year, lat, lon) that sets the tercile boundaries.")],
    method: Annotated[TercileMethod, Field(
        description='"counting" (fraction of members per tercile) or "gaussian" (parametric fit).')] = "counting",
    forecast_variable: VariableArg = None,
    obs_variable: ObsVariableArg = None,
    destination: DestinationArg = None,
) -> DatasetSummary:
    """Convert a single-year ensemble forecast into below/normal/above tercile
    probabilities. Do not use on cross-validated hindcasts: it would leak the
    held-out year into the boundaries (use downscale output_type="tercile" or
    calibrate cv="loyo" for those).

    Returns: {path, dims, coords, variables, attrs, size_bytes, request}; the
    file holds "probability" (tercile, lat, lon).
    Example: to_tercile(forecast_path="cfsv2_2025_bcsd.nc", obs_path="chirps_mam.nc")
    """
    fc = load_array(forecast_path, forecast_variable)
    obs = load_array(obs_path, obs_variable)
    probs = _call(_to_tercile, fc, obs, method=method)
    params = dict(forecast_path=forecast_path, obs_path=obs_path, method=method)
    return _write(probs, "probability", destination, "tercile", Path(forecast_path).stem, params=params)


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------

def _savefig(destination: str | None, *parts: Any, params: dict, dpi: int = 150) -> str:
    import matplotlib.pyplot as plt

    out_path = _output_path(destination, *parts, params=params, suffix=".png")
    fig = plt.gcf()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


@tool()
def plot_terciles(
    probs_path: Annotated[str, Field(
        description="NetCDF path of a (tercile, lat, lon) probability forecast.")],
    title: Annotated[str | None, Field(description="Map title.")] = None,
    variable_kind: Annotated[VariableKind, Field(
        description='Colour convention: "precip" (below=red, above=green) or "temp" '
                    '(below=blue, above=red).')] = "precip",
    smooth: Annotated[bool, Field(description="Smooth the field before drawing.")] = False,
    variable: VariableArg = None,
    destination: Annotated[str | None, Field(
        description="Output .png path. Default: a stable name under AFRICAS2S_MCP_WORKDIR.")] = None,
) -> PlotOut:
    """Render a tercile probability forecast as a dominant-tercile map (IRI /
    PyCPT convention) and save it as PNG.

    Returns: {path, format: "png", request}.
    Example: plot_terciles(probs_path="cfsv2_2025_terciles.nc", title="MAM 2025 precipitation")
    """
    import matplotlib

    matplotlib.use("Agg")
    probs = load_array(probs_path, variable)
    _call(africas2s.plot_terciles, probs, title=title, variable_kind=variable_kind, smooth=smooth)
    params = dict(probs_path=probs_path, title=title, variable_kind=variable_kind, smooth=smooth)
    path = _savefig(destination, "terciles", Path(probs_path).stem, params=params)
    return {"path": path, "format": "png", "request": params}


@tool()
def plot_field(
    path: Annotated[str, Field(
        description="NetCDF path of a 2-D (lat, lon) field: a skill map, anomaly, or deterministic "
                    "forecast. Reduce or select any other dims first.")],
    title: Annotated[str | None, Field(description="Map title.")] = None,
    cmap: Annotated[str | None, Field(description="Matplotlib colormap name, e.g. \"RdBu_r\".")] = None,
    vmin: Annotated[float | None, Field(description="Colour scale minimum.")] = None,
    vmax: Annotated[float | None, Field(description="Colour scale maximum.")] = None,
    cbar_label: Annotated[str | None, Field(description="Colour bar label.")] = None,
    variable: VariableArg = None,
    destination: Annotated[str | None, Field(
        description="Output .png path. Default: a stable name under AFRICAS2S_MCP_WORKDIR.")] = None,
) -> PlotOut:
    """Render a 2-D (lat, lon) field as a map and save it as PNG.

    Returns: {path, format: "png", request}.
    Example: plot_field(path="skill_rpss.nc", title="RPSS", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    """
    import matplotlib

    matplotlib.use("Agg")
    da = load_array(path, variable)
    if da.ndim != 2:
        raise ToolError(f"plot_field needs a 2-D (lat, lon) field; got dims {list(da.dims)}. "
                        "Select or reduce the other dims first.")
    _call(africas2s.plot_field_map, da, title=title, cmap=cmap, vmin=vmin, vmax=vmax,
          cbar_label=cbar_label)
    params = dict(path=path, title=title, cmap=cmap, vmin=vmin, vmax=vmax, cbar_label=cbar_label)
    out = _savefig(destination, "field", Path(path).stem, params=params)
    return {"path": out, "format": "png", "request": params}


# --------------------------------------------------------------------------
# resources: the Agent Skill, for harnesses that can't load skills natively
# --------------------------------------------------------------------------

def _read_text(path: Path, what: str) -> str:
    if not path.exists():
        return (f"{what} is not available in this install (looked for {path}). "
                "It ships with the source checkout: https://github.com/ACMAD-Niamey/africas2s")
    return path.read_text(encoding="utf-8")


@mcp.resource("africas2s://skill", mime_type="text/markdown",
              description="The africas2s Agent Skill: API quick reference, data conventions, discipline rules.")
def skill_resource() -> str:
    return _read_text(_SKILL_DIR / "SKILL.md", "SKILL.md")


@mcp.resource("africas2s://skill/references/{name}", mime_type="text/markdown",
              description="One skill reference doc: api, methods, metrics-and-terciles, "
                          "plotting-reporting, analog-completion, aggregations, troubleshooting.")
def skill_reference(name: str) -> str:
    name = name.removesuffix(".md")
    return _read_text(_SKILL_DIR / "references" / f"{name}.md", f"reference {name!r}")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="africas2s-mcp", description="Run the africas2s MCP server.")
    parser.add_argument("--transport", choices=["stdio", "streamable-http", "sse"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args(argv)
    if args.transport == "stdio":
        mcp.run("stdio")
    else:
        mcp.run(args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
