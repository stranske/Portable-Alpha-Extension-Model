"""Focused coverage for issue #2381 ranked targets (remainder after PR #2384).

Restored for issue #2476 after closed PR #2484.
Evidence: docs/evidence/issue-2476/
"""

from __future__ import annotations

import asyncio
import builtins
import inspect
import json
import runpy
import sys
import time
import types
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from unittest.mock import create_autospec

import pandas as pd
import plotly.graph_objects as go
import pytest

import pa_core.pa as pa_entry
from pa_core.config import ModelConfig, load_config
from pa_core.facade import (
    ExportOptions,
    RunOptions,
    SweepArtifacts,
    export,
    run_single,
    run_sweep,
)
from pa_core.llm import result_explain


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
    explain_kwargs = render_explain.calls[0][1]
    assert explain_kwargs["summary"] is summary
    assert explain_kwargs["manifest_data"] is None
    assert explain_kwargs["xlsx"] == str(workbook)
    assert len(render_comparison.calls) == 1
    comparison_kwargs = render_comparison.calls[0][1]
    assert comparison_kwargs["summary"] is summary
    assert comparison_kwargs["manifest_data"] is None
    assert comparison_kwargs["xlsx"] == str(workbook)
    assert len(render_png.calls) == 1
    png_args, png_kwargs = render_png.calls[0]
    assert png_args == (fake_st,)
    assert png_kwargs["key"] == "results_headline_png"
    assert png_kwargs["label"] == "Download PNG"
    assert png_kwargs["file_name"] == "risk_return.png"
    assert png_kwargs["fig"] is summary
    assert len(render_async.calls) == 1
    async_args, async_kwargs = render_async.calls[0]
    assert async_args == (fake_st,)
    assert async_kwargs["key"] == "results_committee_packet"
    assert async_kwargs["label"] == "📦 Export Committee Packet"
    assert async_kwargs["pending_message"] == "Creating export packet..."
    assert inspect.iscoroutinefunction(async_kwargs["factory"])
    packet_pptx = workbook.with_name("packet.pptx")
    packet_xlsx = workbook.with_name("packet.xlsx")
    packet_pptx.write_bytes(b"pptx")
    packet_xlsx.write_bytes(b"xlsx")
    async_kwargs["on_ready"]((str(packet_pptx), str(packet_xlsx)))
    assert ("📋 Download PowerPoint", "packet.pptx") in fake_st.downloads
    assert ("📊 Download Enhanced Excel", "packet.xlsx") in fake_st.downloads


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
    render_panel = _CallRecorder("comparison")
    availability = render_comparison(
        summary=pd.DataFrame({"monthly_TE": [0.01]}),
        manifest_data={"seed": 1},
        xlsx="current.xlsx",
        render_panel=render_panel,
    )
    assert availability.available is False
    assert availability.prior_manifest_path is None
    expected_message = (
        "Comparison unavailable: missing artifact `manifest_data['previous_run']`. "
        "Expected file path to prior `manifest.json` in the current manifest."
    )
    assert availability.message == expected_message
    assert fake_st.infos == [expected_message]
    assert fake_st.subheaders == ["LLM Comparison"]
    assert render_panel.calls == []
    assert fake_st.errors == []
    assert fake_st.warnings == []


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

    summary = _ScenarioSummary("abc", "2026-01-01", 0, "v1")
    unseeded = _ScenarioSummary("def", "2026-01-02", None, "v2")
    monkeypatch.setattr("pa_core.scenario_registry.list", lambda: [summary, unseeded])
    capsys.readouterr()
    pa_entry.main(["registry", "list"])
    assert capsys.readouterr().out.splitlines() == [
        "abc\t2026-01-01\tseed=0\tcode=v1",
        "def\t2026-01-02\tseed=none\tcode=v2",
    ]

    record = _ScenarioRecord("abc", "2026-01-01", 0, "v1")
    requested_ids: list[str] = []

    def get_scenario(scenario_id: str) -> _ScenarioRecord:
        requested_ids.append(scenario_id)
        return record

    monkeypatch.setattr("pa_core.scenario_registry.get", get_scenario)
    capsys.readouterr()
    pa_entry.main(["registry", "get", "abc"])
    assert requested_ids == ["abc"]
    assert json.loads(capsys.readouterr().out) == {
        "scenario_id": "abc",
        "created_at": "2026-01-01",
        "seed": 0,
        "code_version": "v1",
        "index_hash": "hash",
    }


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


@pytest.mark.parametrize(
    ("mapping_text", "cli_options", "expected_settings"),
    [
        pytest.param(None, [], ("none", "single", 12), id="defaults"),
        pytest.param(
            "covariance_shrinkage: ledoit_wolf\nvol_regime: two_state\nvol_regime_window: '5'\n",
            [],
            ("ledoit_wolf", "two_state", 5),
            id="flat-template",
        ),
        pytest.param(
            "vol_regime: single\ncalibration:\n  covariance_shrinkage: ledoit_wolf\n"
            "  vol_regime: two_state\n  vol_regime_window: '9'\n",
            [],
            ("ledoit_wolf", "two_state", 9),
            id="nested-template",
        ),
        pytest.param(
            "calibration:\n  covariance_shrinkage: ledoit_wolf\n"
            "  vol_regime: two_state\n  vol_regime_window: 9\n",
            ["--cov-shrinkage", "none", "--vol-regime", "single", "--vol-regime-window", "3"],
            ("none", "single", 3),
            id="cli-overrides-template",
        ),
        pytest.param(
            "calibration:\n  covariance_shrinkage: ledoit_wolf\n",
            [],
            ("ledoit_wolf", "single", 12),
            id="partial-template",
        ),
    ],
)
def test_pa_calibrate_input_forwards_settings_and_exports(
    monkeypatch, tmp_path, capsys, mapping_text, cli_options, expected_settings
) -> None:
    from pa_core import data

    importer_class = create_autospec(data.DataImportAgent)
    calibrator_class = create_autospec(data.CalibrationAgent)
    importer = (
        importer_class.return_value
        if mapping_text is None
        else importer_class.from_template.return_value
    )
    importer.min_obs = 7 if mapping_text is None else 19
    importer.allow_below_min_obs = False
    frame = pd.DataFrame({"id": ["IDX"], "return": [0.01]})
    importer.load.return_value = frame
    monkeypatch.setattr(data, "DataImportAgent", importer_class)
    monkeypatch.setattr(data, "CalibrationAgent", calibrator_class)
    input_path = tmp_path / "returns.csv"
    output_path = tmp_path / "assets.yaml"
    args = [
        "calibrate",
        "--input",
        str(input_path),
        "--index-id",
        "IDX",
        "--output",
        str(output_path),
        "--min-obs",
        "7",
    ]
    if mapping_text is not None:
        mapping_path = tmp_path / "mapping.yaml"
        mapping_path.write_text(mapping_text)
        args.extend(["--mapping", str(mapping_path)])
    pa_entry.main([*args, *cli_options])

    if mapping_text is None:
        importer_class.assert_called_once_with(min_obs=7, allow_below_min_obs=True)
        importer_class.from_template.assert_not_called()
    else:
        importer_class.assert_not_called()
        importer_class.from_template.assert_called_once_with(str(mapping_path))
        assert importer.allow_below_min_obs is True
    importer.load.assert_called_once_with(str(input_path))
    covariance_shrinkage, vol_regime, window = expected_settings
    calibrator_class.assert_called_once_with(
        min_obs=importer.min_obs,
        covariance_shrinkage=covariance_shrinkage,
        vol_regime=vol_regime,
        vol_regime_window=window,
    )
    calibrator = calibrator_class.return_value
    calibrator.calibrate.assert_called_once()
    assert calibrator.calibrate.call_args.args[0] is frame
    assert calibrator.calibrate.call_args.kwargs == {"index_id": "IDX"}
    calibrator.to_yaml.assert_called_once_with(calibrator.calibrate.return_value, str(output_path))
    assert capsys.readouterr().out == f"✓ Wrote asset library to {output_path}\n"


@pytest.mark.parametrize(
    ("setting", "bad_value", "allowed"),
    [
        ("covariance_shrinkage", "unsupported", "['ledoit_wolf', 'none']"),
        ("vol_regime", "unsupported", "['single', 'two_state']"),
    ],
    ids=["invalid-covariance", "invalid-regime"],
)
def test_pa_calibrate_rejects_invalid_template_before_loading_data(
    monkeypatch, tmp_path, setting, bad_value, allowed
) -> None:
    from pa_core import data

    importer_class = create_autospec(data.DataImportAgent)
    calibrator_class = create_autospec(data.CalibrationAgent)
    monkeypatch.setattr(data, "DataImportAgent", importer_class)
    monkeypatch.setattr(data, "CalibrationAgent", calibrator_class)
    mapping_path = tmp_path / "mapping.yaml"
    mapping_path.write_text(f"calibration:\n  {setting}: {bad_value}\n")
    with pytest.raises(ValueError) as exc_info:
        pa_entry.main(
            [
                "calibrate",
                "--input",
                str(tmp_path / "missing.csv"),
                "--index-id",
                "IDX",
                "--mapping",
                str(mapping_path),
            ]
        )
    assert str(exc_info.value) == f"{setting} must be one of {allowed}"
    importer_class.from_template.return_value.load.assert_not_called()
    calibrator_class.assert_not_called()


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


@pytest.mark.parametrize(
    "exception_type", [ModuleNotFoundError, RuntimeError], ids=["missing-llm", "panel-error"]
)
def test_render_explain_results_handles_panel_failures(exception_type: type[Exception]) -> None:
    module = _load_results_module()
    render_explain = module["_render_explain_results"]
    fake_st = _ResultsStreamlit()
    render_explain.__globals__["st"] = fake_st

    def _boom(**_: object) -> None:
        raise exception_type("panel failed")

    render_explain.__globals__["render_explain_results_panel"] = _boom
    render_explain(summary=pd.DataFrame({"monthly_TE": [0.01]}), manifest_data=None, xlsx="x.xlsx")
    assert fake_st.infos == ["LLM features unavailable. Install .[llm] to enable Explain Results."]
    assert fake_st.errors == []
    assert fake_st.warnings == []


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

    class _SelectingStreamlit(_ResultsStreamlit):
        def multiselect(self, label: str, options: list[str], default: list[str]) -> list[str]:
            assert label == "Agents"
            assert options == ["AgentA", "AgentB"]
            assert default == options
            return ["AgentB"]

    fake_st = _SelectingStreamlit()
    workbook = tmp_path / "results.xlsx"
    workbook.write_bytes(b"workbook")
    summary = pd.DataFrame(
        {"monthly_TE": [0.02, 0.03]}, index=pd.Index(["AgentA", "AgentB"], name="Agent")
    )
    paths = pd.DataFrame({"AgentA": [0.01, 0.02], "AgentB": [-0.03, 0.04]})
    _patch_results_globals(
        module,
        fake_st=fake_st,
        workbook=workbook,
        summary=summary,
        paths=paths,
        render_explain=_CallRecorder("explain"),
        render_comparison=_CallRecorder("comparison"),
        render_png=_CallRecorder("png"),
        render_async=_CallRecorder("async"),
    )
    module["main"]()
    assert fake_st.titles == ["Results"]
    assert len(fake_st.charts) == 2
    assert fake_st.charts[0] is summary
    pd.testing.assert_frame_equal(fake_st.charts[1], paths[["AgentB"]])


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
        run_sweep(
            cfg, [0.01, 0.02], {"analysis_mode": "vol_mult"}, RunOptions(seed=1)
        )  # type: ignore[arg-type]


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


def _tiny_config(**updates: object) -> ModelConfig:
    payload: dict[str, object] = {"N_SIMULATIONS": 2, "N_MONTHS": 3}
    payload.update(updates)
    return load_config("examples/scenarios/my_first_scenario.yml").model_copy(update=payload)


def _index_values() -> pd.Series:
    return pd.Series([0.01, -0.02, 0.015])


def test_redact_sensitive_ignores_blank_and_non_string_secrets() -> None:
    redacted = result_explain._redact_sensitive(
        "prefix abc suffix",
        secrets=["", 5, "abc"],  # type: ignore[list-item]
    )
    assert redacted == "prefix [REDACTED] suffix"


def test_extract_response_text_uses_response_string_when_content_is_not_text() -> None:
    response = types.SimpleNamespace(content=None)
    assert result_explain._extract_response_text(response) == str(response).strip()


def test_extract_response_text_skips_items_that_are_not_text_or_mappings() -> None:
    response = types.SimpleNamespace(content=["keep", 5, {"text": "tail"}])
    assert result_explain._extract_response_text(response) == "keep tail"


def test_extract_metric_delta_ignores_non_mapping_summary_values() -> None:
    delta = result_explain._extract_metric_delta(
        {"stress_delta_summary": {"summary": {"note": "skip", "a": {"mean": 0.2}}}}
    )
    assert delta == pytest.approx(0.2)


def test_pa_sweep_omits_substream_metadata_when_runner_has_none(
    monkeypatch, tmp_path, capsys
) -> None:
    config = tmp_path / "sweep.yml"
    config.write_text("index: data.csv\n")
    captured: dict[str, object] = {}

    class _Runner:
        substream_ids: list[str] = []

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

        def run(self) -> list[dict[str, bool]]:
            return [{"ok": True}]

    monkeypatch.setattr("pa_core.config.load_config", lambda _path: object())
    monkeypatch.setattr("pa_core.facade.apply_run_options", lambda cfg, _opts: cfg)
    monkeypatch.setattr("pa_core.backend.resolve_and_set_backend", lambda *_a, **_k: None)
    monkeypatch.setattr("pa_core.backend.get_backend", lambda: "numpy")
    monkeypatch.setattr("pa_core.data.load_index_returns", lambda _path: _index_values())
    monkeypatch.setattr("pa_core.units.get_index_series_unit", lambda: "decimal")
    monkeypatch.setattr("pa_core.units.normalize_index_series", lambda series, _unit: series)
    monkeypatch.setattr("pa_core.sweep.SweepRunner", _Runner)

    def _export(results: object, *, filename: str, metadata: dict[str, object]) -> None:
        captured["results"] = results
        captured["filename"] = filename
        captured["metadata"] = metadata

    monkeypatch.setattr("pa_core.reporting.sweep_excel.export_sweep_results", _export)
    pa_entry.main(["sweep", "--config", str(config)])
    assert captured["metadata"] == {"rng_seed": None}
    assert captured["results"] == [{"ok": True}]
    assert captured["filename"] == "Sweep.xlsx"
    assert "[BACKEND] Using backend: numpy" in capsys.readouterr().out


def test_run_single_and_sweep_accept_single_column_index_frame() -> None:
    cfg = _tiny_config()
    frame = pd.DataFrame({"IDX": [0.01, -0.02, 0.015]})
    single = run_single(cfg, frame, RunOptions(seed=7))
    assert not single.summary.empty
    sweep = run_sweep(
        cfg,
        frame,
        {
            "analysis_mode": "vol_mult",
            "sd_multiple_min": 1.0,
            "sd_multiple_max": 1.0,
            "sd_multiple_step": 1.0,
        },
        RunOptions(seed=7),
    )
    assert not sweep.summary.empty


def test_run_single_continues_when_sleeve_validation_reports_no_violations(monkeypatch) -> None:
    seen: list[pd.DataFrame] = []

    def _validate(summary: pd.DataFrame, _cfg: object) -> list[str]:
        seen.append(summary)
        return []

    monkeypatch.setattr("pa_core.reporting.constraints.validate_sleeve_constraints", _validate)
    artifacts = run_single(
        _tiny_config(sleeve_validate_on_run=True),
        _index_values(),
        RunOptions(seed=11),
    )
    assert seen
    assert not artifacts.summary.empty


def _inject_correlation_repair(monkeypatch: pytest.MonkeyPatch, info: dict[str, object]) -> None:
    import pa_core.sim as sim_mod

    real = sim_mod.draw_joint_returns

    def _wrapped(*args: object, **kwargs: object) -> object:
        result = real(*args, **kwargs)
        params = kwargs.get("params")
        if isinstance(params, dict):
            params["_correlation_repair_info"] = dict(info)
        return result

    monkeypatch.setattr(sim_mod, "draw_joint_returns", _wrapped)


def test_run_single_records_repair_info_when_correlation_matrices_absent(monkeypatch) -> None:
    _inject_correlation_repair(
        monkeypatch,
        {
            "corr_before": None,
            "corr_after": None,
            "repair_mode": "absent",
            "method": "none",
            "shrinkage": None,
            "min_eigenvalue_before": None,
            "min_eigenvalue_after": None,
            "max_abs_delta": None,
            "repair_applied": False,
        },
    )
    artifacts = run_single(_tiny_config(), _index_values(), RunOptions(seed=13))
    assert "_corr_before_df" not in artifacts.inputs
    info = artifacts.inputs["_corr_repair_info_df"]
    assert info.loc[0, "mode"] == "absent"


def test_run_single_skips_correlation_frames_when_matrix_shapes_differ(monkeypatch) -> None:
    _inject_correlation_repair(
        monkeypatch,
        {
            "corr_before": [[1.0, 0.0], [0.0, 1.0]],
            "corr_after": [[1.0]],
            "corr_delta": None,
            "repair_mode": "mismatch",
            "method": "none",
            "shrinkage": 0.0,
            "min_eigenvalue_before": 1.0,
            "min_eigenvalue_after": 1.0,
            "max_abs_delta": 0.0,
            "repair_applied": False,
        },
    )
    artifacts = run_single(_tiny_config(), _index_values(), RunOptions(seed=17))
    assert "_corr_before_df" not in artifacts.inputs
    assert "_corr_delta_df" not in artifacts.inputs
    info = artifacts.inputs["_corr_repair_info_df"]
    assert info.loc[0, "mode"] == "mismatch"


def test_export_sweep_without_manifest_passes_no_metadata(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def _export(results: object, *, filename: str, metadata: object) -> None:
        captured["results"] = results
        captured["filename"] = filename
        captured["metadata"] = metadata

    monkeypatch.setattr("pa_core.reporting.sweep_excel.export_sweep_results", _export)
    cfg = _tiny_config(N_SIMULATIONS=1, N_MONTHS=1)
    artifacts = SweepArtifacts(
        config=cfg,
        index_series=_index_values(),
        results=[],
        summary=pd.DataFrame({"Config": ["Base"]}),
        inputs={},
        manifest=None,
    )
    output = tmp_path / "sweep.xlsx"
    export(artifacts, output)
    assert captured["metadata"] is None
    assert captured["filename"] == str(output)
    assert captured["results"] == []


def test_export_normalizes_tuple_agent_semantics_to_a_list(run_artifacts_fixture, tmp_path) -> None:
    import numpy as np

    run_artifacts_fixture.inputs["_agent_semantics_df"] = ({"Agent": "Base", "n": np.int64(3)},)
    export(run_artifacts_fixture, tmp_path / "tuple.xlsx")
    stored = run_artifacts_fixture.inputs["_agent_semantics_df"]
    assert isinstance(stored, list)
    assert stored[0]["Agent"] == "Base"
    assert type(stored[0]["n"]) is int
    assert stored[0]["n"] == 3


def test_render_explain_results_forwards_summary_manifest_and_workbook() -> None:
    module = _load_results_module()
    render_explain = module["_render_explain_results"]
    summary = pd.DataFrame({"monthly_TE": [0.02]})
    manifest = {"seed": 3}
    captured: dict[str, object] = {}

    def _panel(*, summary_df: pd.DataFrame, manifest: dict[str, object], xlsx_path: str) -> None:
        captured["summary_df"] = summary_df
        captured["manifest"] = manifest
        captured["xlsx_path"] = xlsx_path

    render_explain.__globals__["render_explain_results_panel"] = _panel
    render_explain(summary=summary, manifest_data=manifest, xlsx="results.xlsx")
    assert captured["summary_df"] is summary
    assert captured["manifest"] == {"seed": 3}
    assert captured["xlsx_path"] == "results.xlsx"


def test_results_main_drops_manifest_when_json_cannot_be_parsed(tmp_path) -> None:
    module = _load_results_module()
    fake_st = _ResultsStreamlit()
    workbook = tmp_path / "results.xlsx"
    workbook.write_bytes(b"workbook")
    manifest = tmp_path / "results.json"
    manifest.write_text("{")
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
    render_explain = _CallRecorder("explain")
    _patch_results_globals(
        module,
        fake_st=fake_st,
        workbook=workbook,
        summary=summary,
        paths=pd.DataFrame([[0.01]], columns=["Base"]),
        render_explain=render_explain,
        render_comparison=_CallRecorder("comparison"),
        render_png=_CallRecorder("png"),
        render_async=_CallRecorder("async"),
        manifest_path=manifest,
    )
    module["main"]()
    assert fake_st.captions == []
    assert render_explain.calls[0][1]["manifest_data"] is None


def test_results_main_omits_seed_caption_when_manifest_seed_is_absent(tmp_path) -> None:
    module = _load_results_module()
    fake_st = _ResultsStreamlit()
    workbook = tmp_path / "results.xlsx"
    workbook.write_bytes(b"workbook")
    manifest = tmp_path / "results.json"
    manifest.write_text(json.dumps({"previous_run": None}))
    summary = pd.DataFrame({"monthly_TE": [0.02], "Config": ["Base"]})
    _patch_results_globals(
        module,
        fake_st=fake_st,
        workbook=workbook,
        summary=summary,
        paths=None,
        render_explain=_CallRecorder("explain"),
        render_comparison=_CallRecorder("comparison"),
        render_png=_CallRecorder("png"),
        render_async=_CallRecorder("async"),
        manifest_path=manifest,
    )
    module["main"]()
    assert fake_st.captions == []


def test_results_main_omits_tracking_error_when_neither_column_is_present(tmp_path) -> None:
    module = _load_results_module()
    fake_st = _ResultsStreamlit()
    workbook = tmp_path / "results.xlsx"
    workbook.write_bytes(b"workbook")
    summary = pd.DataFrame({"monthly_CVaR": [-0.02], "Config": ["Base"]})
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
    )
    module["main"]()
    metrics = [pair for col in fake_st.column_containers[0] for pair in col.metrics]
    assert ("monthly_CVaR", "-2.00%") in metrics
    assert all(label != "Tracking Error" for label, _value in metrics)


def test_results_main_skips_non_headline_charts_when_paths_are_missing(tmp_path) -> None:
    module = _load_results_module()
    fake_st = _ResultsStreamlit()
    workbook = tmp_path / "results.xlsx"
    workbook.write_bytes(b"workbook")
    summary = pd.DataFrame({"monthly_TE": [0.02], "Config": ["Base"]})
    _patch_results_globals(
        module,
        fake_st=fake_st,
        workbook=workbook,
        summary=summary,
        paths=None,
        render_explain=_CallRecorder("explain"),
        render_comparison=_CallRecorder("comparison"),
        render_png=_CallRecorder("png"),
        render_async=_CallRecorder("async"),
    )
    module["main"]()
    assert fake_st.charts == [summary]


def _packet_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    manifest_text: str | None,
    stress: pd.DataFrame,
) -> dict[str, object]:
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
    with pd.ExcelWriter(workbook) as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        stress.to_excel(writer, sheet_name="StressDelta", index=False)
    manifest = tmp_path / "results.json"
    if manifest_text is None:
        manifest_path = tmp_path / "missing-manifest.json"
    else:
        manifest.write_text(manifest_text)
        manifest_path = manifest
    captured: dict[str, object] = {}

    async def _create(**kwargs: object) -> tuple[str, str]:
        captured["packet"] = kwargs
        return "packet.pptx", "packet.xlsx"

    def _render_async(*_args: object, **kwargs: object) -> None:
        captured["factory"] = kwargs["factory"]

    monkeypatch.setattr("pa_core.reporting.export_packet.create_export_packet_async", _create)
    _patch_results_globals(
        module,
        fake_st=fake_st,
        workbook=workbook,
        summary=summary,
        paths=pd.DataFrame([[0.01]], columns=["Base"]),
        render_explain=_CallRecorder("explain"),
        render_comparison=_CallRecorder("comparison"),
        render_png=_CallRecorder("png"),
        render_async=_render_async,
        manifest_path=manifest_path,
    )
    module["main"]()
    asyncio.run(captured["factory"]())  # type: ignore[arg-type]
    packet = captured["packet"]
    assert isinstance(packet, dict)
    return packet


def test_results_packet_keeps_nonempty_stress_without_a_manifest(tmp_path, monkeypatch) -> None:
    packet = _packet_arguments(
        tmp_path,
        monkeypatch,
        manifest_text=None,
        stress=pd.DataFrame({"delta": [0.5]}),
    )
    stress = packet["stress_delta_df"]
    assert isinstance(stress, pd.DataFrame)
    assert stress.loc[0, "delta"] == pytest.approx(0.5)
    assert packet["prev_manifest"] is None
    assert packet["prev_summary_df"] is None


def test_results_packet_drops_empty_stress_and_non_string_previous_run(
    tmp_path, monkeypatch
) -> None:
    packet = _packet_arguments(
        tmp_path,
        monkeypatch,
        manifest_text=json.dumps({"previous_run": 5}),
        stress=pd.DataFrame(),
    )
    assert packet["stress_delta_df"] is None
    assert packet["prev_manifest"] is None
    assert packet["prev_summary_df"] is None


def test_results_packet_ignores_missing_previous_manifest_path(tmp_path, monkeypatch) -> None:
    missing = tmp_path / "absent-prior.json"
    packet = _packet_arguments(
        tmp_path,
        monkeypatch,
        manifest_text=json.dumps({"previous_run": str(missing)}),
        stress=pd.DataFrame({"delta": [0.1]}),
    )
    assert packet["prev_manifest"] is None
    assert isinstance(packet["stress_delta_df"], pd.DataFrame)


def test_results_packet_skips_missing_previous_workbook(tmp_path, monkeypatch) -> None:
    prior_manifest = tmp_path / "prior.json"
    missing_output = tmp_path / "missing-prior.xlsx"
    prior_manifest.write_text(json.dumps({"cli_args": {"output": str(missing_output)}, "seed": 4}))
    packet = _packet_arguments(
        tmp_path,
        monkeypatch,
        manifest_text=json.dumps({"previous_run": str(prior_manifest)}),
        stress=pd.DataFrame({"delta": [0.1]}),
    )
    assert packet["prev_manifest"]["seed"] == 4
    assert packet["prev_summary_df"] is None


def test_results_packet_clears_previous_manifest_when_json_is_invalid(
    tmp_path, monkeypatch
) -> None:
    prior_manifest = tmp_path / "prior.json"
    prior_manifest.write_text("{")
    packet = _packet_arguments(
        tmp_path,
        monkeypatch,
        manifest_text=json.dumps({"previous_run": str(prior_manifest)}),
        stress=pd.DataFrame({"delta": [0.1]}),
    )
    assert packet["prev_manifest"] is None
    assert packet["prev_summary_df"] is None


@pytest.mark.parametrize(
    ("exc", "error_text", "info_text"),
    [
        (
            RuntimeError("kaleido is not installed"),
            "Export packet requires Kaleido or Chrome/Chromium",
            "pip install kaleido",
        ),
        (
            RuntimeError("chrome is unavailable"),
            "Export packet requires Kaleido or Chrome/Chromium",
            "pip install kaleido",
        ),
        (RuntimeError("disk full"), "Export packet failed: disk full", None),
        (
            ValueError("bad packet"),
            "Unexpected error creating export packet: bad packet",
            "individual exports",
        ),
    ],
)
def test_results_main_reports_packet_export_failures(
    tmp_path, exc: Exception, error_text: str, info_text: str | None
) -> None:
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

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise exc

    _patch_results_globals(
        module,
        fake_st=fake_st,
        workbook=workbook,
        summary=summary,
        paths=pd.DataFrame([[0.01]], columns=["Base"]),
        render_explain=_CallRecorder("explain"),
        render_comparison=_CallRecorder("comparison"),
        render_png=_CallRecorder("png"),
        render_async=_raise,
    )
    module["main"]()
    assert any(error_text in message for message in fake_st.errors)
    if info_text is not None:
        assert any(info_text in message for message in fake_st.infos)


def test_results_main_auto_refresh_sleeps_then_reruns(tmp_path, monkeypatch) -> None:
    module = _load_results_module()

    class _Refreshing(_ResultsStreamlit):
        def __init__(self) -> None:
            super().__init__()
            self.reruns = 0

        def checkbox(self, label: str) -> bool:
            return label == "Auto-refresh"

        def number_input(self, label: str, *args: object, **kwargs: object) -> float:
            if args:
                return float(args[-1])  # type: ignore[arg-type]
            return float(kwargs.get("value", 0.0))  # type: ignore[arg-type]

        def rerun(self) -> None:
            self.reruns += 1

    fake_st = _Refreshing()
    workbook = tmp_path / "results.xlsx"
    workbook.write_bytes(b"workbook")
    summary = pd.DataFrame({"monthly_TE": [0.02], "Config": ["Base"]})
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda seconds: slept.append(seconds))
    _patch_results_globals(
        module,
        fake_st=fake_st,
        workbook=workbook,
        summary=summary,
        paths=None,
        render_explain=_CallRecorder("explain"),
        render_comparison=_CallRecorder("comparison"),
        render_png=_CallRecorder("png"),
        render_async=_CallRecorder("async"),
    )
    module["main"]()
    assert slept == [60.0]
    assert fake_st.reruns == 1


def test_results_script_entrypoint_invokes_main(monkeypatch) -> None:
    import streamlit as st

    seen: list[str] = []

    def _title(text: str) -> None:
        seen.append(text)
        raise RuntimeError("entrypoint-stop")

    monkeypatch.setattr(st, "title", _title)
    previous_main = sys.modules.get("__main__")
    try:
        with pytest.raises(RuntimeError, match="entrypoint-stop"):
            runpy.run_path("dashboard/pages/4_Results.py", run_name="__main__")
    finally:
        if previous_main is not None:
            sys.modules["__main__"] = previous_main
        else:
            sys.modules.pop("__main__", None)
    assert seen == ["Results"]
