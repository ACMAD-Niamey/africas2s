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
* Library exceptions are re-raised as ``ToolError`` so the agent sees the
  message. The MCP SDK hides the text of any other exception.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import africas2s
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
selected by name; list_registry shows what is available.

Typical flow: optimize (pick a method by CV skill) or downscale -> skill ->
plot_terciles. For tercile metrics such as rpss the forecast must be tercile
probabilities: use downscale(output_type="tercile") or to_tercile.

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
# helpers
# --------------------------------------------------------------------------

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


def _coord_summary(coord) -> dict:
    import numpy as np

    vals = coord.values
    out: dict[str, Any] = {"size": int(vals.size), "dtype": str(vals.dtype)}
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


def summarize(ds, *, path: str | Path | None = None, max_stats_elements: int = 20_000_000) -> dict:
    """Compact, JSON-safe description of a Dataset / DataArray."""
    import numpy as np
    import xarray as xr

    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset(name=ds.name or "data")
    out: dict[str, Any] = {
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
        entry: dict[str, Any] = {
            "dims": list(da.dims),
            "shape": [int(s) for s in da.shape],
            "dtype": str(da.dtype),
            "units": da.attrs.get("units"),
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


def _write(da, name: str, destination: str | None, *parts: Any, params: dict) -> dict:
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
# introspection
# --------------------------------------------------------------------------

@mcp.tool()
def list_registry() -> dict:
    """Names accepted by the other tools: downscaling methods, calibrators,
    ensemble strategies, and skill metrics (with aliases)."""
    metrics: dict[str, list[str]] = {}
    for name, cls in registry._METRICS.items():
        metrics.setdefault(cls.__name__, []).append(name)
    return {
        "methods": sorted(registry._METHODS),
        "calibrators": sorted(registry._CALIBRATORS),
        "strategies": sorted(registry._STRATEGIES),
        "metrics": sorted(registry._METRICS),
        "metric_aliases": {v[0]: v[1:] for v in metrics.values() if len(v) > 1},
        # africas2s.skill is the verb; the module (and its PRESETS) sits behind it.
        "metric_presets": dict(importlib.import_module("africas2s.skill").PRESETS),
    }


@mcp.tool()
def describe_dataset(path: str) -> dict:
    """Summarize a NetCDF file: dims, coordinate ranges, variables, units, NaN
    fraction, attributes. Use it on any path returned by another tool."""
    with _open(path) as ds:
        return summarize(ds, path=Path(path).expanduser())


# --------------------------------------------------------------------------
# downscale / optimize / calibrate
# --------------------------------------------------------------------------

@mcp.tool()
def downscale(
    predictor_hindcast_path: str,
    obs_path: str,
    method: str = "bcsd",
    output_type: str = "continuous",
    forecast_path: str | None = None,
    predictor_variable: str | None = None,
    obs_variable: str | None = None,
    options: dict[str, Any] | None = None,
    destination: str | None = None,
) -> dict:
    """Bias-correct and downscale a GCM hindcast/forecast onto the observation grid.

    predictor_hindcast_path: NetCDF (year, member, lat, lon). obs_path: NetCDF
    (year, lat, lon) on the fine target grid. method: a name from list_registry
    (bcsd, cca, qm, dqm, delta, climatology, rank-analog, chelsa, corrdiff).
    output_type: "continuous" or "tercile" (below/normal/above probabilities).
    forecast_path: optional (member, lat, lon) forecast to predict; when omitted
    the last hindcast year is held out and predicted. options: method keyword
    arguments (e.g. n_modes for cca). Writes NetCDF and returns path + summary."""
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


@mcp.tool()
def optimize(
    predictor_hindcast_path: str,
    obs_path: str,
    methods: list[str] | None = None,
    cv: str = "loyo",
    primary_metric: str = "rpss",
    predictor_variable: str | None = None,
    obs_variable: str | None = None,
    options: dict[str, Any] | None = None,
    destination: str | None = None,
) -> dict:
    """Try several downscaling methods under cross-validation and keep the most
    skillful. methods default to ["bcsd", "cca"]; cv is a CV scheme name
    ("loyo" leave-one-year-out); primary_metric a metric name. Returns the
    winning method, its CV score, and the path of its forecast."""
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


@mcp.tool()
def calibrate(
    obs_path: str,
    method: str = "ereg",
    hindcast_path: str | None = None,
    forecast_path: str | None = None,
    models: dict[str, list[str]] | None = None,
    output_type: str = "tercile",
    forecast_year: int | None = None,
    cv: str | None = None,
    cv_window: int = 1,
    variable: str | None = None,
    obs_variable: str | None = None,
    options: dict[str, Any] | None = None,
    destination: str | None = None,
) -> dict:
    """Calibrate gridded predictors already on the obs grid into tercile
    probabilities (no regridding: interpolate the GCM to the obs grid first).

    Single model: hindcast_path (year, member, lat, lon) + forecast_path
    (member, lat, lon). Several models: models={"name": [hindcast_path,
    forecast_path], ...}; their calibrated maps are averaged. method: ereg,
    logit, or smoothed_regression. forecast_year: the year the forecast is
    for (ereg needs it to place the forecast on the fitted trend).
    cv="loyo" returns the leave-year-out
    cross-validated hindcast (year, tercile, lat, lon) for skill scoring
    instead of the forecast. options: method keywords."""
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

@mcp.tool()
def ensemble(
    forecast_paths: list[str],
    obs_path: str | None = None,
    strategy: str = "uniform",
    optimize_ensemble: bool = False,
    primary_metric: str = "rpss",
    variable: str | None = None,
    obs_variable: str | None = None,
    options: dict[str, Any] | None = None,
    destination: str | None = None,
) -> dict:
    """Combine several forecasts (same grid and shape) into one multi-model
    ensemble. strategy: uniform, skill_weighted, bma, drop_worst. obs_path is
    required for skill-based weighting and for the prediction-error-variance
    diagnostic. Returns the combined forecast path plus weights, member names,
    effective N, and whether the skill gate passed."""
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


@mcp.tool()
def skill(
    forecast_path: str,
    obs_path: str,
    metrics: list[str] | str | None = None,
    spatial: bool = False,
    include_diagrams: bool = False,
    forecast_variable: str | None = None,
    obs_variable: str | None = None,
    options: dict[str, Any] | None = None,
    destination: str | None = None,
) -> dict:
    """Score a forecast against observations. metrics: a list of names, a
    preset ("svslrf" = rpss+roc+reliability, "all"), or omitted for rpss.
    Tercile metrics need (year, tercile, lat, lon) CV probabilities;
    deterministic metrics take continuous fields. spatial=true also writes
    per-cell skill maps to NetCDF and returns the path. include_diagrams adds
    ROC / reliability curve data inline."""
    fc = load_array(forecast_path, forecast_variable)
    obs = load_array(obs_path, obs_variable)
    kwargs = dict(options or {})
    report = _call(africas2s.skill, fc, obs, metrics=metrics, spatial=spatial, **kwargs)
    payload = report.to_dict()
    out: dict[str, Any] = {
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


@mcp.tool()
def to_tercile(
    forecast_path: str,
    obs_path: str,
    method: str = "counting",
    forecast_variable: str | None = None,
    obs_variable: str | None = None,
    destination: str | None = None,
) -> dict:
    """Convert a single-year ensemble forecast (member, lat, lon) into tercile
    probabilities using the observation climatology (year, lat, lon) for the
    boundaries. method: "counting" (member counts) or "gaussian". Do not use on
    cross-validated hindcasts: that leaks the held-out year."""
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


@mcp.tool()
def plot_terciles(
    probs_path: str,
    title: str | None = None,
    variable_kind: str = "precip",
    smooth: bool = False,
    variable: str | None = None,
    destination: str | None = None,
) -> dict:
    """Render a (tercile, lat, lon) probability forecast as a dominant-tercile
    map (IRI / PyCPT convention) and save it as PNG. variable_kind: "precip"
    (below=red, above=green) or "temp" (below=blue, above=red)."""
    import matplotlib

    matplotlib.use("Agg")
    probs = load_array(probs_path, variable)
    _call(africas2s.plot_terciles, probs, title=title, variable_kind=variable_kind, smooth=smooth)
    params = dict(probs_path=probs_path, title=title, variable_kind=variable_kind, smooth=smooth)
    path = _savefig(destination, "terciles", Path(probs_path).stem, params=params)
    return {"path": path, "format": "png", "request": params}


@mcp.tool()
def plot_field(
    path: str,
    title: str | None = None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str | None = None,
    variable: str | None = None,
    destination: str | None = None,
) -> dict:
    """Render a 2-D (lat, lon) field (a skill map, anomaly, or deterministic
    forecast) as a map and save it as PNG. Extra dims must already be reduced."""
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
