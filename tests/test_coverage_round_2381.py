"""Focused coverage for issue #2381 ranked targets (remainder after PR #2384).

Evidence: docs/evidence/issue-2381-closer-repair/
"""

from __future__ import annotations

import builtins
import json
import runpy
import sys
import types
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import plotly.graph_objects as go
import pytest

import pa_core.pa as pa_entry
from pa_core.config import ModelConfig, load_config
from pa_core.facade import ExportOptions, RunOptions, export, run_single, run_sweep


class _Container:
    def __init__(self) -> None:
        self.metrics: list[tuple[str, str]] = []
        self.downloads: list[tuple[str, str]] = []

    def __enter__(self) -> _Container:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def metric(self, label: str, value: str, **_: object) -> None:
        self.metrics.append((label, value))

    def download_button(self, label: str, *_: object, **kwargs: object) -> None:
        self.downloads.append((label, str(kwargs.get("file_name"))))


class _ResultsStreamlit:
    def __init__(self) -> None:
        self.sidebar = self
        self.session_state: dict[str, object] = {}
        self.titles: list[str] = []
        self.subheaders: list[str] = []
        self.captions: list[str] = []
        self.frames: list[pd.DataFrame] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.infos: list[str] = []
        self.charts: list[object] = []
        self.downloads: list[tuple[str, str]] = []
        self.column_containers: list[list[_Container]] = []
        self.tab_containers: list[list[_Container]] = []

    def title(self, value: str) -> None:
        self.titles.append(value)

    def subheader(self, value: str) -> None:
        self.subheaders.append(value)

    def caption(self, value: str) -> None:
        self.captions.append(value)

    def columns(self, count: int) -> list[_Container]:
        containers = [_Container() for _ in range(count)]
        self.column_containers.append(containers)
        return containers

    def tabs(self, names: list[str]) -> list[_Container]:
        containers = [_Container() for _ in names]
        self.tab_containers.append(containers)
        return containers

    def slider(self, *_: object, **kwargs: object) -> int:
        return int(kwargs.get("value", 1))

    def multiselect(self, _: str, options: list[str], default: list[str]) -> list[str]:
        return default if default else options

    def number_input(self, *_: object, **__: object) -> float:
        return 0.0

    def checkbox(self, _: str) -> bool:
        return False

    def plotly_chart(self, figure: object, **_: object) -> None:
        self.charts.append(figure)

    def dataframe(self, value: pd.DataFrame) -> None:
        self.frames.append(value)

    def download_button(self, label: str, *_: object, **kwargs: object) -> None:
        self.downloads.append((label, str(kwargs.get("file_name"))))

    def info(self, value: str) -> None:
        self.infos.append(value)

    def success(self, value: str) -> None:
        self.infos.append(value)

    def warning(self, value: str) -> None:
        self.warnings.append(value)

    def error(self, value: str) -> None:
        self.errors.append(value)

    def stop(self) -> None:
        raise RuntimeError("streamlit-stop")


def _load_results_module() -> dict[str, Any]:
    module_name = "dashboard.pages.page_4_Results"
    module_path = str(Path("dashboard/pages/4_Results.py"))
    module_dict = runpy.run_path(module_path, run_name=module_name)
    if module_name not in sys.modules:
        module = types.ModuleType(module_name)
        module.__dict__.update(module_dict)
        sys.modules[module_name] = module
    return module_dict


class _CallRecorder:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __call__(self, *args: object, **kwargs: object) -> None:
        self.calls.append((args, kwargs))


def _patch_results_globals(
    module: dict[str, Any],
    *,
    fake_st: _ResultsStreamlit,
    workbook: Path,
    summary: pd.DataFrame,
    paths: pd.DataFrame | None,
    render_explain: _CallRecorder,
    render_comparison: _CallRecorder,
    render_png: _CallRecorder,
    render_async: _CallRecorder,
    manifest_path: Path | None = None,
) -> None:
    globals_ = module["main"].__globals__
    globals_.update(
        {
            "st": fake_st,
            "PLOTS": {"Headline": "headline", "Paths": "paths"},
            "render_settings_sidebar": lambda _: (str(workbook), "theme.yaml"),
            "apply_theme": lambda _: None,
            "load_data": lambda _: (summary, paths),
            "_get_plot_fn": lambda _: lambda frame: frame,
            "_render_explain_results": render_explain,
            "_render_comparison_panel": render_comparison,
            "render_png_download": render_png,
            "render_async_export_button": render_async,
            "manifest_path_for_output": lambda _xlsx: manifest_path
            or workbook.with_suffix(".json"),
        }
    )


def test_results_main_renders_metrics_plots_and_download_for_available_workbook(tmp_path) -> None:
    module = _load_results_module()
    fake_st = _ResultsStreamlit()
    workbook = tmp_path / "results.xlsx"
    workbook.write_bytes(b"workbook")
    summary = pd.DataFrame(
        {
            "Config": ["Base"],
            "monthly_TE": [0.02],
            "monthly_CVaR": [-0.01],
            "monthly_BreachProb": [0.03],
            "Agent": ["Total"],
            "terminal_AnnReturn": [0.04],
        }
    )
    paths = pd.DataFrame([[0.01]], columns=["Base"])
    render_explain = _CallRecorder("explain")
    render_comparison = _CallRecorder("comparison")
    render_png = _CallRecorder("png")
    render_async = _CallRecorder("async")
    _patch_results_globals(
        module,
        fake_st=fake_st,
        workbook=workbook,
        summary=summary,
        paths=paths,
        render_explain=render_explain,
        render_comparison=render_comparison,
        render_png=render_png,
        render_async=render_async,
    )

    module["main"]()

    assert fake_st.titles == ["Results"]
    assert fake_st.subheaders == ["Key Metrics"]
    assert fake_st.frames == [summary]
    assert fake_st.errors == []
    assert len(fake_st.charts) == 2
    assert fake_st.charts[0] is summary
    assert list(fake_st.charts[1].columns) == ["Base"]
    assert ("Download XLSX", "results.xlsx") in fake_st.downloads
    assert len(fake_st.column_containers) == 2
    metric_cols = fake_st.column_containers[0]
    assert len(metric_cols) == 4
    metrics = [pair for col in metric_cols for pair in col.metrics]
    assert metrics == [
        ("Tracking Error", "2.00%"),
        ("monthly_CVaR", "-1.00%"),
        ("Breach Prob", "3.00%"),
        ("Overlay Total", "4.00%"),
    ]
    assert len(render_explain.calls) == 1
    assert render_explain.calls[0][1]["summary"] is summary
    assert len(render_comparison.calls) == 1
    assert render_comparison.calls[0][1]["xlsx"] == str(workbook)
    assert len(render_png.calls) >= 1
    assert len(render_async.calls) >= 1


def test_overlay_total_return_computes_from_summary_with_agent_total() -> None:
    module = _load_results_module()
    overlay_fn = module["_overlay_total_return"]

    summary_with_total = pd.DataFrame(
        {"Agent": ["Base", "Total"], "terminal_AnnReturn": [0.05, 0.08]}
    )
    assert overlay_fn(summary_with_total) == pytest.approx(0.08)
    assert overlay_fn(
        pd.DataFrame({"Agent": ["Total"], "terminal_AnnReturn": [0.10]})
    ) == pytest.approx(0.10)
    assert overlay_fn(pd.DataFrame({"Agent": ["Base"], "terminal_AnnReturn": [0.05]})) is None
    assert overlay_fn(pd.DataFrame({"Config": ["Base"]})) is None


def test_prior_output_path_resolves_relative_cli_output(tmp_path) -> None:
    module = _load_results_module()
    prior_path = tmp_path / "prior" / "manifest.json"
    prior_path.parent.mkdir()
    assert (
        module["_prior_output_path"]({"cli_args": {"output": "workbook.xlsx"}}, prior_path)
        == prior_path.parent / "workbook.xlsx"
    )
    absolute = tmp_path / "abs.xlsx"
    assert (
        module["_prior_output_path"]({"cli_args": {"output": str(absolute)}}, prior_path)
        == absolute
    )
    assert module["_prior_output_path"]({"cli_args": {"output": "  "}}, prior_path) is None
    assert module["_prior_output_path"](None, prior_path) is None


def test_results_main_stops_when_workbook_missing(tmp_path) -> None:
    module = _load_results_module()
    fake_st = _ResultsStreamlit()
    missing = tmp_path / "missing.xlsx"
    summary = pd.DataFrame({"monthly_TE": [0.01]})
    _patch_results_globals(
        module,
        fake_st=fake_st,
        workbook=missing,
        summary=summary,
        paths=None,
        render_explain=_CallRecorder("explain"),
        render_comparison=_CallRecorder("comparison"),
        render_png=_CallRecorder("png"),
        render_async=_CallRecorder("async"),
    )
    with pytest.raises(RuntimeError, match="streamlit-stop"):
        module["main"]()
    assert fake_st.infos


def test_results_main_shows_manifest_seed_and_legacy_tracking_error(tmp_path) -> None:
    module = _load_results_module()
    fake_st = _ResultsStreamlit()
    workbook = tmp_path / "results.xlsx"
    workbook.write_bytes(b"workbook")
    manifest = tmp_path / "results.json"
    manifest.write_text(json.dumps({"seed": 4242}))
    summary = pd.DataFrame(
        {
            "Config": ["Base"],
            "TrackingErr": [0.015],
            "monthly_CVaR": [-0.02],
            "monthly_BreachProb": [0.01],
            "Agent": ["Total"],
            "terminal_AnnReturn": [0.03],
        }
    )
    _patch_results_globals(
        module,
        fake_st=fake_st,
        workbook=workbook,
        summary=summary,
        paths=pd.DataFrame([[0.01]], columns=["Base"]),
        render_explain=_CallRecorder("explain"),
        render_comparison=_CallRecorder("comparison"),
        render_png=_CallRecorder("png"),
        render_async=_CallRecorder("async"),
        manifest_path=manifest,
    )
    module["main"]()
    assert fake_st.captions == ["Seed: 4242"]
    metrics = [pair for col in fake_st.column_containers[0] for pair in col.metrics]
    assert metrics[0] == ("Tracking Error", "1.50%")


def test_results_main_surfaces_png_kaleido_runtime_error(tmp_path) -> None:
    module = _load_results_module()
    fake_st = _ResultsStreamlit()
    workbook = tmp_path / "results.xlsx"
    workbook.write_bytes(b"workbook")
    summary = pd.DataFrame(
        {
            "monthly_TE": [0.02],
            "monthly_CVaR": [-0.01],
            "monthly_BreachProb": [0.03],
            "Agent": ["Total"],
            "terminal_AnnReturn": [0.04],
            "Config": ["Base"],
        }
    )

    def _raise_kaleido(*_a: object, **_k: object) -> None:
        raise RuntimeError("kaleido is not installed")

    _patch_results_globals(
        module,
        fake_st=fake_st,
        workbook=workbook,
        summary=summary,
        paths=pd.DataFrame([[0.01]], columns=["Base"]),
        render_explain=_CallRecorder("explain"),
        render_comparison=_CallRecorder("comparison"),
        render_png=_raise_kaleido,
        render_async=_CallRecorder("async"),
    )
    module["main"]()
    assert any("Kaleido" in message for message in fake_st.warnings)


def test_render_comparison_panel_shows_info_when_previous_run_unavailable() -> None:
    module = _load_results_module()
    fake_st = _ResultsStreamlit()
    render_comparison = module["_render_comparison_panel"]
    render_comparison.__globals__["st"] = fake_st
    availability = render_comparison(
        summary=pd.DataFrame({"monthly_TE": [0.01]}),
        manifest_data={"seed": 1},
        xlsx="current.xlsx",
    )
    assert availability.available is False
    assert fake_st.infos
    assert fake_st.subheaders == ["LLM Comparison"]


def _exit_code(call: Callable[[], None]) -> int:
    try:
        call()
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 0
    return 0


@dataclass
class _ScenarioSummary:
    scenario_id: str
    created_at: str
    seed: int | None
    code_version: str


@dataclass
class _ScenarioRecord:
    scenario_id: str
    created_at: str
    seed: int | None
    code_version: str
    index_hash: str = "hash"


def test_pa_main_uses_sys_argv_when_argv_is_none(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["pa", "describe"])
    pa_entry.main()
    captured = capsys.readouterr()
    assert "tool_descriptor" in captured.out or captured.out.startswith("{")


def test_pa_sweep_requires_index_when_not_in_config(tmp_path) -> None:
    config = tmp_path / "sweep.yml"
    config.write_text("output_path: out.xlsx\n")
    with pytest.raises(ValueError, match="index path is required"):
        pa_entry.main(["sweep", "--config", str(config)])


def test_pa_registry_list_and_get(monkeypatch, capsys) -> None:
    monkeypatch.setattr("pa_core.scenario_registry.list", lambda: [])
    pa_entry.main(["registry", "list"])
    assert "No registered scenarios" in capsys.readouterr().out

    summary = _ScenarioSummary("abc", "2026-01-01", 7, "v1")
    monkeypatch.setattr("pa_core.scenario_registry.list", lambda: [summary])
    capsys.readouterr()
    pa_entry.main(["registry", "list"])
    out = capsys.readouterr().out
    assert "abc" in out
    assert "seed=7" in out

    record = _ScenarioRecord("abc", "2026-01-01", 7, "v1")
    monkeypatch.setattr("pa_core.scenario_registry.get", lambda _id: record)
    capsys.readouterr()
    pa_entry.main(["registry", "get", "abc"])
    assert '"scenario_id": "abc"' in capsys.readouterr().out


def test_pa_calibrate_returns_path_and_input_requires_index_id(
    monkeypatch, tmp_path, capsys
) -> None:
    returns_csv = tmp_path / "returns.csv"
    returns_csv.write_text("Date,IDX,H,E,M\n2020-01-31,0.01,0.02,0.03,0.04\n")
    returns_xlsx = tmp_path / "returns.xlsx"
    pd.DataFrame(
        {"Date": ["2020-01-31"], "IDX": [0.01], "H": [0.02], "E": [0.03], "M": [0.04]}
    ).to_excel(returns_xlsx, index=False)

    class _Result:
        def to_scenario(self) -> dict[str, object]:
            return {"N_SIMULATIONS": 1}

        def to_model_config(self, *, alpha_map: dict[str, str]) -> dict[str, object]:
            return {"mu_H": 0.01}

    monkeypatch.setattr(
        "pa_core.calibration.calibrate_returns",
        lambda *_a, **_k: _Result(),
    )
    monkeypatch.setattr(
        "pa_core.calibration.infer_series_ids",
        lambda columns, **_k: ("IDX", ["H", "E", "M"]),
    )
    monkeypatch.setattr(
        "pa_core.calibration.build_calibration_report",
        lambda *_a, **_k: {"status": "ok"},
    )
    monkeypatch.setattr("pa_core.schema.save_scenario", lambda *_a, **_k: None)

    out_yaml = tmp_path / "params.yml"
    pa_entry.main(
        [
            "calibrate",
            "--returns",
            str(returns_csv),
            "--output",
            str(out_yaml),
            "--index-id",
            "IDX",
        ]
    )
    assert out_yaml.exists()
    assert "Wrote calibrated parameters" in capsys.readouterr().out

    out_xlsx = tmp_path / "params-xlsx.yml"
    capsys.readouterr()
    pa_entry.main(
        [
            "calibrate",
            "--returns",
            str(returns_xlsx),
            "--output",
            str(out_xlsx),
            "--index-id",
            "IDX",
        ]
    )
    assert out_xlsx.exists()

    with pytest.raises(ValueError, match="index-id is required"):
        pa_entry.main(["calibrate", "--input", str(returns_csv), "--output", str(out_yaml)])


def test_pa_convert_command_emits_deprecation_and_writes_yaml(tmp_path, capsys) -> None:
    csv_path = tmp_path / "params.csv"
    csv_path.write_text("Parameter,Value\nNumber of simulations,5\n")
    yaml_path = tmp_path / "params.yml"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        pa_entry.main(["convert", "--csv", str(csv_path), "--yaml", str(yaml_path)])
    assert any(isinstance(item.message, DeprecationWarning) for item in caught)
    assert yaml_path.exists()
    assert "N_SIMULATIONS" in yaml_path.read_text()


def test_facade_scalar_helpers_tolerate_missing_numpy(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "numpy":
            raise ImportError("numpy unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from pa_core import facade

    assert facade._to_builtin_scalar({"x": 1}) == {"x": 1}
    inputs: dict[str, Any] = {"_agent_semantics_df": [{"Agent": "Base"}]}
    facade._serialize_agent_semantics_input(inputs)
    assert isinstance(inputs["_agent_semantics_df"], list)


def test_facade_run_single_and_sweep_reject_non_series_index() -> None:
    cfg = ModelConfig(N_SIMULATIONS=1, N_MONTHS=1, financing_mode="broadcast")
    bad_frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="convertible to pandas Series"):
        run_single(cfg, bad_frame, RunOptions(seed=1))
    with pytest.raises(ValueError, match="convertible to pandas Series"):
        run_sweep(cfg, bad_frame, {"analysis_mode": "vol_mult"}, RunOptions(seed=1))
    with pytest.raises(ValueError, match="must be a pandas Series"):
        run_single(cfg, [[1, 2, 3]], RunOptions(seed=1))  # type: ignore[arg-type]


def test_render_explain_results_handles_panel_failures() -> None:
    module = _load_results_module()
    render_explain = module["_render_explain_results"]
    fake_st = _ResultsStreamlit()
    render_explain.__globals__["st"] = fake_st

    def _missing_module(**_: object) -> None:
        raise ModuleNotFoundError("langchain")

    render_explain.__globals__["render_explain_results_panel"] = _missing_module
    render_explain(summary=pd.DataFrame({"monthly_TE": [0.01]}), manifest_data=None, xlsx="x.xlsx")
    assert fake_st.infos

    def _boom(**_: object) -> None:
        raise RuntimeError("panel failed")

    fake_st.infos.clear()
    render_explain.__globals__["render_explain_results_panel"] = _boom
    render_explain(summary=pd.DataFrame({"monthly_TE": [0.01]}), manifest_data=None, xlsx="x.xlsx")
    assert fake_st.infos


def test_check_previous_run_availability_rejects_non_mapping_manifest() -> None:
    module = _load_results_module()
    result = module["_check_previous_run_availability"](None)
    assert result.available is False
    assert "missing artifact" in str(result.message).lower()


def test_results_main_png_non_kaleido_runtime_error_uses_error(tmp_path) -> None:
    module = _load_results_module()
    fake_st = _ResultsStreamlit()
    workbook = tmp_path / "results.xlsx"
    workbook.write_bytes(b"workbook")
    summary = pd.DataFrame(
        {
            "Config": ["Base"],
            "monthly_TE": [0.02],
            "monthly_CVaR": [-0.01],
            "monthly_BreachProb": [0.03],
            "Agent": ["Total"],
            "terminal_AnnReturn": [0.04],
        }
    )

    def _raise_other(*_a: object, **_k: object) -> None:
        raise RuntimeError("png backend unavailable")

    _patch_results_globals(
        module,
        fake_st=fake_st,
        workbook=workbook,
        summary=summary,
        paths=pd.DataFrame([[0.01]], columns=["Base"]),
        render_explain=_CallRecorder("explain"),
        render_comparison=_CallRecorder("comparison"),
        render_png=_raise_other,
        render_async=_CallRecorder("async"),
    )
    module["main"]()
    assert any("PNG export error" in message for message in fake_st.errors)


def test_results_main_multiselect_uses_index_when_config_missing(tmp_path) -> None:
    module = _load_results_module()
    fake_st = _ResultsStreamlit()
    workbook = tmp_path / "results.xlsx"
    workbook.write_bytes(b"workbook")
    summary = pd.DataFrame({"monthly_TE": [0.02]}, index=pd.Index(["AgentA"], name="Agent"))
    _patch_results_globals(
        module,
        fake_st=fake_st,
        workbook=workbook,
        summary=summary,
        paths=pd.DataFrame([[0.01]], columns=["AgentA"]),
        render_explain=_CallRecorder("explain"),
        render_comparison=_CallRecorder("comparison"),
        render_png=_CallRecorder("png"),
        render_async=_CallRecorder("async"),
    )
    module["main"]()
    assert fake_st.titles == ["Results"]


def test_results_main_committee_packet_ready_callback(tmp_path, monkeypatch) -> None:
    module = _load_results_module()
    fake_st = _ResultsStreamlit()
    workbook = tmp_path / "results.xlsx"
    summary = pd.DataFrame(
        {
            "Config": ["Base"],
            "monthly_TE": [0.02],
            "monthly_CVaR": [-0.01],
            "monthly_BreachProb": [0.03],
            "Agent": ["Total"],
            "terminal_AnnReturn": [0.04],
        }
    )
    summary.to_excel(workbook, sheet_name="Summary", index=False)
    prior_manifest = tmp_path / "prior.json"
    prior_output = tmp_path / "prior.xlsx"
    pd.DataFrame({"monthly_TE": [0.01]}).to_excel(prior_output, sheet_name="Summary", index=False)
    prior_manifest.write_text(json.dumps({"cli_args": {"output": str(prior_output)}, "seed": 9}))
    manifest = tmp_path / "results.json"
    manifest.write_text(json.dumps({"seed": 1, "previous_run": str(prior_manifest)}))

    packet_pptx = tmp_path / "packet.pptx"
    packet_xlsx = tmp_path / "packet.xlsx"
    packet_pptx.write_bytes(b"pptx")
    packet_xlsx.write_bytes(b"xlsx")

    async def _create_packet_async(**_: object) -> tuple[str, str]:
        return str(packet_pptx), str(packet_xlsx)

    monkeypatch.setattr(
        "pa_core.reporting.export_packet.create_export_packet_async",
        _create_packet_async,
    )
    monkeypatch.setattr("dashboard.browser_exports.is_browser_runtime", lambda: False)

    from dashboard import browser_exports

    class _ButtonStreamlit(_ResultsStreamlit):
        def button(self, _label: str, *, key: str | None = None) -> bool:
            return True

    fake_st = _ButtonStreamlit()
    _patch_results_globals(
        module,
        fake_st=fake_st,
        workbook=workbook,
        summary=summary,
        paths=pd.DataFrame([[0.01]], columns=["Base"]),
        render_explain=_CallRecorder("explain"),
        render_comparison=_CallRecorder("comparison"),
        render_png=_CallRecorder("png"),
        render_async=browser_exports.render_async_export_button,
        manifest_path=manifest,
    )
    module["main"]()
    assert any("PowerPoint" in label for label, _ in fake_st.downloads)


def test_pa_calibrate_returns_rejects_unknown_suffix_and_uses_raw_frame(
    monkeypatch,
    tmp_path,
) -> None:
    bad = tmp_path / "returns.txt"
    bad.write_text("x")
    with pytest.raises(ValueError, match="unsupported returns file type"):
        pa_entry.main(["calibrate", "--returns", str(bad), "--output", str(tmp_path / "o.yml")])

    raw_csv = tmp_path / "raw.csv"
    raw_csv.write_text("IDX,H,E,M\n0.01,0.02,0.03,0.04\n")

    class _Result:
        def to_scenario(self) -> dict[str, object]:
            return {}

        def to_model_config(self, *, alpha_map: dict[str, str]) -> dict[str, object]:
            return {}

    monkeypatch.setattr("pa_core.calibration.calibrate_returns", lambda *_a, **_k: _Result())
    monkeypatch.setattr(
        "pa_core.calibration.infer_series_ids",
        lambda columns, **_k: ("IDX", ["H", "E", "M"]),
    )
    monkeypatch.setattr(
        "pa_core.calibration.build_calibration_report", lambda *_a, **_k: {"status": "ok"}
    )
    out_yaml = tmp_path / "params.yml"
    pa_entry.main(
        [
            "calibrate",
            "--returns",
            str(raw_csv),
            "--output",
            str(out_yaml),
            "--index-id",
            "IDX",
            "--date-col",
            "MissingDate",
        ]
    )
    assert out_yaml.exists()


def test_facade_run_single_requires_regime_transition_when_regimes_set() -> None:
    cfg = ModelConfig.model_construct(
        N_SIMULATIONS=1,
        N_MONTHS=1,
        financing_mode="broadcast",
        regimes=[{"name": "stress", "sigma_H": 0.2}],
        regime_transition=None,
    )
    idx = pd.Series([0.01, 0.02])
    with pytest.raises(ValueError, match="regime_transition is required"):
        run_single(cfg, idx, RunOptions(seed=1))


def test_facade_run_sweep_rejects_scalar_index() -> None:
    cfg = ModelConfig(N_SIMULATIONS=1, N_MONTHS=1, financing_mode="broadcast")
    with pytest.raises(ValueError, match="must be a pandas Series"):
        run_sweep(cfg, [0.01, 0.02], {"analysis_mode": "vol_mult"}, RunOptions(seed=1))  # type: ignore[arg-type]


def test_export_run_artifacts_with_show_visualizations(
    run_artifacts_fixture, tmp_path, monkeypatch
) -> None:
    output_file = tmp_path / "run_output.xlsx"
    shown: list[object] = []
    fig = go.Figure()
    fig.show = lambda: shown.append(fig)  # type: ignore[method-assign]

    monkeypatch.setattr(
        "pa_core.viz.compare_scenarios",
        lambda *_a, **_k: {"risk_return": fig},
    )
    export(
        run_artifacts_fixture,
        output_file,
        ExportOptions(
            include_visualizations=True,
            viz_output_dir=tmp_path / "viz",
            show_visualizations=True,
            alt_text="Run",
        ),
    )
    assert shown


@pytest.fixture
def run_artifacts_fixture():
    cfg = load_config("examples/scenarios/my_first_scenario.yml").model_copy(
        update={"N_SIMULATIONS": 2, "N_MONTHS": 3}
    )
    idx = pd.Series([0.01, -0.02, 0.015])
    artifacts = run_single(cfg, idx, RunOptions(seed=123))
    artifacts.inputs["_agent_semantics_df"] = pd.DataFrame(
        [{"Agent": "Base", "mismatch_flag": False}]
    )
    return artifacts


def test_export_prefers_nonempty_agent_semantics_dataframe(run_artifacts_fixture, tmp_path) -> None:
    output_file = tmp_path / "agent_semantics.xlsx"
    export(run_artifacts_fixture, output_file)
    serialized = run_artifacts_fixture.inputs["_agent_semantics_df"]
    assert isinstance(serialized, list)
    assert serialized[0]["Agent"] == "Base"


def test_export_sweep_artifacts_with_visualizations(tmp_path) -> None:
    cfg = load_config("examples/scenarios/my_first_scenario.yml").model_copy(
        update={"N_SIMULATIONS": 2, "N_MONTHS": 3}
    )
    idx = pd.Series([0.01, -0.02, 0.015])
    sweep_params = {
        "analysis_mode": "vol_mult",
        "sd_multiple_min": 1.0,
        "sd_multiple_max": 1.0,
        "sd_multiple_step": 1.0,
    }
    artifacts = run_sweep(cfg, idx, sweep_params, RunOptions(seed=123))
    output_file = tmp_path / "sweep_output.xlsx"
    viz_dir = tmp_path / "viz"
    export(
        artifacts,
        output_file,
        ExportOptions(include_visualizations=True, viz_output_dir=viz_dir, alt_text="Sweep"),
    )
    assert output_file.exists()
    assert any(viz_dir.glob("*.html"))
