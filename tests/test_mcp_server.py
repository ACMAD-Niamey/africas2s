"""Unit tests for the africas2s MCP server (fast, synthetic data).

The server is a thin wrapper, so these tests pin the contract: tools are
registered under the documented names, gridded inputs are read from NetCDF
by path, gridded outputs are written to NetCDF under the workdir and returned
as path + summary, scalars come back inline and JSON-safe, and library errors
reach the caller with their message.
"""

import asyncio
import json

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("mcp")

from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

from africas2s.mcp import server  # noqa: E402


EXPECTED_TOOLS = {
    "list_registry", "describe_dataset", "downscale", "optimize", "calibrate",
    "ensemble", "skill", "to_tercile", "plot_terciles", "plot_field",
}


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _workdir(tmp_path, monkeypatch):
    monkeypatch.setenv("AFRICAS2S_MCP_WORKDIR", str(tmp_path / "work"))
    return tmp_path / "work"


@pytest.fixture
def gcm_hindcast(synthetic_gcm_hindcast):
    return synthetic_gcm_hindcast


@pytest.fixture
def obs(synthetic_obs):
    return synthetic_obs


@pytest.fixture
def files(tmp_path, gcm_hindcast, obs):
    """Write the conftest synthetic hindcast / obs to NetCDF and return paths."""
    hind = tmp_path / "hind.nc"
    ob = tmp_path / "obs.nc"
    gcm_hindcast.rename("precip").to_netcdf(hind)
    obs.rename("precip").to_netcdf(ob)
    return {"hind": str(hind), "obs": str(ob)}


def test_tools_registered():
    names = {t.name for t in _run(server.mcp.list_tools())}
    assert EXPECTED_TOOLS <= names


def test_tool_schemas_have_descriptions():
    for tool in _run(server.mcp.list_tools()):
        assert tool.description, f"{tool.name} has no description"
        assert tool.input_schema.get("type") == "object"


def test_list_registry_matches_registries():
    reg = server.list_registry()
    json.dumps(reg)
    assert {"bcsd", "cca", "qm"} <= set(reg["methods"])
    assert {"ereg", "logit"} <= set(reg["calibrators"])
    assert {"uniform", "bma"} <= set(reg["strategies"])
    assert {"rpss", "roc", "pearson_r"} <= set(reg["metrics"])
    assert reg["metric_presets"]["svslrf"] == ["rpss", "roc", "reliability"]


def test_load_array_picks_single_variable_or_demands_one(tmp_path, obs):
    single = tmp_path / "one.nc"
    obs.rename("precip").to_netcdf(single)
    da = server.load_array(str(single))
    assert da.name == "precip" and da.dims == ("year", "lat", "lon")

    multi = tmp_path / "two.nc"
    xr.Dataset({"a": obs, "b": obs}).to_netcdf(multi)
    with pytest.raises(ToolError, match="2 data variables"):
        server.load_array(str(multi))
    assert server.load_array(str(multi), "b").name == "b"
    with pytest.raises(ToolError, match="no variable 'zzz'"):
        server.load_array(str(multi), "zzz")
    with pytest.raises(ToolError, match="No such file"):
        server.load_array(str(tmp_path / "missing.nc"))


def test_describe_dataset_via_protocol(files):
    result = _run(server.mcp.call_tool("describe_dataset", {"path": files["hind"]}))
    assert not result.is_error
    payload = json.loads(result.content[0].text)
    assert payload["variables"]["precip"]["dims"] == ["year", "member", "lat", "lon"]
    assert payload["coords"]["year"]["min"] == 2000


def test_downscale_continuous_writes_fine_grid(files, _workdir, obs):
    out = server.downscale(files["hind"], files["obs"], method="bcsd")
    assert out["path"].startswith(str(_workdir))
    da = xr.open_dataarray(out["path"])
    assert da.dims[-2:] == ("lat", "lon")
    assert da.sizes["lat"] == obs.sizes["lat"] and da.sizes["lon"] == obs.sizes["lon"]
    assert out["request"]["method"] == "bcsd"


def test_downscale_tercile_sums_to_one(files):
    out = server.downscale(files["hind"], files["obs"], method="bcsd", output_type="tercile")
    da = xr.open_dataarray(out["path"])
    assert "tercile" in da.dims and list(da.tercile.values) == [0, 1, 2]
    total = da.sum("tercile")
    assert np.allclose(total.values[np.isfinite(total.values)], 1.0)
    assert out["variables"]["probability"]["max"] <= 1.0


def test_downscale_unknown_method_surfaces_message(files):
    with pytest.raises(ToolError, match="nope"):
        server.downscale(files["hind"], files["obs"], method="nope")


def test_downscale_error_via_protocol_reaches_client(files):
    with pytest.raises(ToolError, match="nope"):
        _run(server.mcp.call_tool("downscale", {
            "predictor_hindcast_path": files["hind"], "obs_path": files["obs"], "method": "nope",
        }))


def test_optimize_reports_winner(files):
    out = server.optimize(files["hind"], files["obs"], methods=["bcsd", "climatology"])
    assert out["method"] in ("bcsd", "climatology")
    assert isinstance(out["score"], float)
    assert out["path"].endswith(".nc")
    json.dumps(out)


def test_to_tercile_then_skill(files, tmp_path, gcm_hindcast, perfect_tercile_forecast):
    fc_path = tmp_path / "fc.nc"
    gcm_hindcast.isel(year=-1, drop=True).rename("precip").to_netcdf(fc_path)
    probs = server.to_tercile(str(fc_path), files["obs"])
    da = xr.open_dataarray(probs["path"])
    assert da.dims == ("tercile", "lat", "lon")

    # A (year, tercile, lat, lon) hindcast of probabilities scores with rpss.
    cv_path = tmp_path / "cv.nc"
    perfect_tercile_forecast.rename("probability").to_netcdf(cv_path)
    report = server.skill(str(cv_path), files["obs"], metrics=["rpss"], spatial=True)
    assert report["scores"]["rpss"] > 0.9
    json.dumps(report)
    assert report["spatial_path"].endswith(".nc")
    assert "rpss" in report["spatial"]["variables"]


def test_skill_deterministic_metric(files, tmp_path, gcm_hindcast, obs):
    fc = tmp_path / "det.nc"
    gcm_hindcast.mean("member").interp(lat=obs.lat, lon=obs.lon).rename("precip").to_netcdf(fc)
    report = server.skill(str(fc), files["obs"], metrics=["pearson_r"])
    assert set(report["scores"]) == {"pearson_r"}
    assert -1.0 <= report["scores"]["pearson_r"] <= 1.0


def test_ensemble_uniform(files, tmp_path, gcm_hindcast, obs):
    members = []
    for i in range(2):
        p = tmp_path / f"m{i}.nc"
        (gcm_hindcast.mean("member").interp(lat=obs.lat, lon=obs.lon) + i).rename("precip").to_netcdf(p)
        members.append(str(p))
    out = server.ensemble(members, files["obs"], strategy="uniform")
    assert out["member_names"] == ["m0", "m1"]
    assert np.allclose(out["weights"], [0.5, 0.5])
    assert out["gate_passed"] is True
    da = xr.open_dataarray(out["path"])
    assert da.dims == ("year", "lat", "lon")
    json.dumps(out)


def test_ensemble_empty_is_error():
    with pytest.raises(ToolError, match="must not be empty"):
        server.ensemble([])


def test_calibrate_ereg_single_model(files, tmp_path, gcm_hindcast, obs):
    fine = gcm_hindcast.interp(lat=obs.lat, lon=obs.lon)
    hind = tmp_path / "hind_fine.nc"
    fc = tmp_path / "fc_fine.nc"
    fine.rename("precip").to_netcdf(hind)
    fine.isel(year=-1, drop=True).rename("precip").to_netcdf(fc)
    out = server.calibrate(files["obs"], method="ereg", hindcast_path=str(hind),
                           forecast_path=str(fc), forecast_year=2010)
    da = xr.open_dataarray(out["path"])
    assert da.dims == ("tercile", "lat", "lon")
    total = da.sum("tercile")
    assert np.allclose(total.values[np.isfinite(total.values)], 1.0)


def test_calibrate_requires_inputs(files):
    with pytest.raises(ToolError, match="Pass hindcast_path"):
        server.calibrate(files["obs"])
    with pytest.raises(ToolError, match="must be \\[hindcast_path, forecast_path\\]"):
        server.calibrate(files["obs"], models={"a": ["only-one"]})


def test_plot_terciles_and_field_write_png(files, tmp_path, gcm_hindcast, obs):
    pytest.importorskip("cartopy")
    probs = server.downscale(files["hind"], files["obs"], method="bcsd", output_type="tercile")
    png = server.plot_terciles(probs["path"], title="test")
    assert png["path"].endswith(".png")
    assert open(png["path"], "rb").read(8).startswith(b"\x89PNG")

    field = tmp_path / "field.nc"
    obs.isel(year=0, drop=True).rename("precip").to_netcdf(field)
    out = server.plot_field(str(field), title="obs", cmap="viridis")
    assert open(out["path"], "rb").read(8).startswith(b"\x89PNG")

    with pytest.raises(ToolError, match="2-D"):
        server.plot_field(files["obs"])


def test_resources_listed_and_readable():
    uris = {str(r.uri) for r in _run(server.mcp.list_resources())}
    assert "africas2s://skill" in uris
    templates = {t.uri_template for t in _run(server.mcp.list_resource_templates())}
    assert "africas2s://skill/references/{name}" in templates
    contents = list(_run(server.mcp.read_resource("africas2s://skill")))
    assert contents and "africas2s" in contents[0].content
    api = list(_run(server.mcp.read_resource("africas2s://skill/references/methods")))
    assert api and "bcsd" in api[0].content


def test_main_parses_transport(monkeypatch):
    seen = {}
    monkeypatch.setattr(server.mcp, "run", lambda transport, **kw: seen.update(t=transport, **kw))
    server.main(["--transport", "streamable-http", "--port", "9002"])
    assert seen == {"t": "streamable-http", "host": "127.0.0.1", "port": 9002}
    server.main([])
    assert seen["t"] == "stdio"


def test_schemas_carry_field_descriptions_enums_and_outputs():
    from africas2s import registry

    tools = {t.name: t for t in _run(server.mcp.list_tools())}
    ds = tools["downscale"]
    props = ds.input_schema["properties"]
    assert all(p.get("description") for p in props.values()), [k for k, p in props.items() if not p.get("description")]
    assert set(props["method"]["enum"]) == set(registry._METHODS)
    assert props["output_type"]["enum"] == ["continuous", "tercile"]
    assert "path" in ds.output_schema["properties"]
    ens = tools["ensemble"].input_schema["properties"]
    assert set(ens["strategy"]["enum"]) == set(registry._STRATEGIES)
    assert "weights" in tools["ensemble"].output_schema["properties"]
    assert tools["list_registry"].annotations.read_only_hint is True
    assert tools["downscale"].annotations.read_only_hint is False
    for tool in tools.values():
        assert "Returns:" in tool.description, tool.name
        assert not tool.description.startswith(" "), tool.name


def test_enum_violation_is_rejected_at_the_protocol(files):
    with pytest.raises(ToolError, match="method"):
        _run(server.mcp.call_tool("downscale", {
            "predictor_hindcast_path": files["hind"], "obs_path": files["obs"], "method": "nope",
        }))


def test_list_registry_matches_schema_enums():
    reg = server.list_registry()
    tools = {t.name: t for t in _run(server.mcp.list_tools())}
    assert tools["optimize"].input_schema["properties"]["cv"]["enum"] == reg["cv_schemes"]
    assert set(tools["skill"].input_schema["properties"]["metrics"]["anyOf"][0]["items"]["enum"]) == set(reg["metrics"])
