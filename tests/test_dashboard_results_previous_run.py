from __future__ import annotations

import json
import runpy
from pathlib import Path

import pandas as pd


def _load_results_module() -> dict[str, object]:
    return runpy.run_path(str(Path("dashboard/pages/4_Results.py")))


def test_check_previous_run_availability_missing_field() -> None:
    module = _load_results_module()
    check = module["_check_previous_run_availability"]
    result = check({"seed": 7})

    assert result.available is False
    assert result.prior_manifest_path is None
    assert "missing artifact" in str(result.message).lower()
    assert "previous_run" in str(result.message)


def test_check_previous_run_availability_missing_file_path(tmp_path) -> None:
    module = _load_results_module()
    check = module["_check_previous_run_availability"]
    missing = tmp_path / "missing_manifest.json"
    result = check({"previous_run": str(missing)})

    assert result.available is False
    assert result.prior_manifest_path == missing
    assert str(missing) in str(result.message)
    assert "expected file path" in str(result.message).lower()


def test_check_previous_run_availability_unreadable_json(tmp_path) -> None:
    module = _load_results_module()
    check = module["_check_previous_run_availability"]
    invalid_manifest = tmp_path / "invalid_manifest.json"
    invalid_manifest.write_text("{broken-json")
    result = check({"previous_run": str(invalid_manifest)})

    assert result.available is False
    assert result.prior_manifest_path == invalid_manifest
    assert "unreadable artifact" in str(result.message).lower()
    assert str(invalid_manifest) in str(result.message)


def test_check_previous_run_availability_readable_manifest(tmp_path) -> None:
    module = _load_results_module()
    check = module["_check_previous_run_availability"]
    prior_output = tmp_path / "prior.xlsx"
    pd.DataFrame({"monthly_TE": [0.02]}).to_excel(prior_output, sheet_name="Summary", index=False)
    prior_manifest = tmp_path / "prior_manifest.json"
    prior_manifest.write_text(json.dumps({"seed": 42, "cli_args": {"output": str(prior_output)}}))
    result = check({"previous_run": str(prior_manifest)})

    assert result.available is True
    assert result.prior_manifest_path == prior_manifest
    assert result.message is None


def test_check_previous_run_availability_missing_prior_summary(tmp_path) -> None:
    module = _load_results_module()
    check = module["_check_previous_run_availability"]
    prior_manifest = tmp_path / "prior_manifest.json"
    prior_manifest.write_text(json.dumps({"seed": 42, "cli_args": {"output": "missing.xlsx"}}))
    result = check({"previous_run": str(prior_manifest)})

    assert result.available is False
    assert result.prior_manifest_path == prior_manifest
    assert "missing prior output workbook" in str(result.message).lower()
    assert str(tmp_path / "missing.xlsx") in str(result.message)


def test_check_previous_run_availability_unreadable_prior_summary(tmp_path) -> None:
    module = _load_results_module()
    check = module["_check_previous_run_availability"]
    prior_output = tmp_path / "prior.xlsx"
    prior_output.write_text("not an excel workbook")
    prior_manifest = tmp_path / "prior_manifest.json"
    prior_manifest.write_text(json.dumps({"seed": 42, "cli_args": {"output": str(prior_output)}}))
    result = check({"previous_run": str(prior_manifest)})

    assert result.available is False
    assert result.prior_manifest_path == prior_manifest
    assert "unreadable prior output workbook" in str(result.message).lower()
    assert str(prior_output) in str(result.message)


def test_check_previous_run_availability_missing_prior_output_path(tmp_path) -> None:
    module = _load_results_module()
    check = module["_check_previous_run_availability"]
    prior_manifest = tmp_path / "prior_manifest.json"
    prior_manifest.write_text(json.dumps({"seed": 42, "cli_args": {}}))
    result = check({"previous_run": str(prior_manifest)})

    assert result.available is False
    assert result.prior_manifest_path == prior_manifest
    assert "missing `cli_args.output` path" in str(result.message)


class _FakeStreamlit:
    def __init__(self) -> None:
        self.subheaders: list[str] = []
        self.infos: list[str] = []

    def subheader(self, message: str) -> None:
        self.subheaders.append(message)

    def info(self, message: str) -> None:
        self.infos.append(message)


class _ResultsContainer:
    def __enter__(self) -> "_ResultsContainer":
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def metric(self, *args: object, **kwargs: object) -> None:
        return None


class _ResultsStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, object] = {}
        self.sidebar = self
        self.titles: list[str] = []
        self.plot_count = 0
        self.downloads: list[str] = []

    def title(self, value: str) -> None:
        self.titles.append(value)

    def slider(self, *_args: object, **kwargs: object) -> object:
        return kwargs.get("value", _args[-1])

    def multiselect(self, _label: str, _options: list[str], default: list[str]) -> list[str]:
        return default

    def number_input(self, *_args: object, **kwargs: object) -> object:
        return kwargs.get("value", _args[-1])

    def checkbox(self, *_args: object, **_kwargs: object) -> bool:
        return False

    def subheader(self, *_args: object, **_kwargs: object) -> None:
        return None

    def columns(self, count: int) -> list[_ResultsContainer]:
        return [_ResultsContainer() for _ in range(count)]

    def tabs(self, names: list[str]) -> list[_ResultsContainer]:
        return [_ResultsContainer() for _ in names]

    def plotly_chart(self, *_args: object, **_kwargs: object) -> None:
        self.plot_count += 1

    def dataframe(self, *_args: object, **_kwargs: object) -> None:
        return None

    def download_button(self, label: str, *_args: object, **_kwargs: object) -> None:
        self.downloads.append(label)


def test_main_renders_data_and_export_controls(tmp_path) -> None:
    """Exercise the results page's normal, artifact-backed rendering path."""
    module = _load_results_module()
    main = module["main"]
    fake_st = _ResultsStreamlit()
    workbook = tmp_path / "results.xlsx"
    workbook.write_bytes(b"placeholder")
    summary = pd.DataFrame(
        {
            "monthly_TE": [0.01],
            "monthly_CVaR": [0.02],
            "monthly_BreachProb": [0.03],
            "Agent": ["Total"],
            "AnnReturn": [0.04],
        }
    )
    globals_ = main.__globals__
    globals_.update(
        {
            "st": fake_st,
            "render_settings_sidebar": lambda _default: (str(workbook), "theme.toml"),
            "apply_theme": lambda _theme: None,
            "load_data": lambda _xlsx: (summary, None),
            "manifest_path_for_output": lambda _xlsx: tmp_path / "missing-manifest.json",
            "_render_explain_results": lambda **_kwargs: None,
            "_render_comparison_panel": lambda **_kwargs: None,
            "PLOTS": {"Headline": "headline"},
            "_get_plot_fn": lambda _name: lambda frame: {"rows": len(frame)},
            "render_png_download": lambda *_args, **_kwargs: None,
            "render_async_export_button": lambda *_args, **_kwargs: None,
        }
    )

    main()

    assert fake_st.titles == ["Results"]
    assert fake_st.plot_count == 1
    assert fake_st.downloads == ["Download XLSX"]


def test_render_comparison_panel_calls_component_when_previous_run_readable(tmp_path) -> None:
    module = _load_results_module()
    render_comparison = module["_render_comparison_panel"]
    fake_st = _FakeStreamlit()
    render_comparison.__globals__["st"] = fake_st

    current_output = tmp_path / "current.xlsx"
    current_output.write_bytes(b"placeholder")
    prior_output = tmp_path / "prior.xlsx"
    pd.DataFrame({"monthly_TE": [0.02]}).to_excel(prior_output, sheet_name="Summary", index=False)
    prior_manifest = tmp_path / "prior_manifest.json"
    prior_manifest.write_text(json.dumps({"seed": 42, "cli_args": {"output": str(prior_output)}}))
    manifest_data = {"previous_run": str(prior_manifest)}
    summary_df = pd.DataFrame({"monthly_TE": [0.02]})
    captured: dict[str, object] = {}

    def _fake_render_panel(*, summary_df, manifest_data, run_key) -> None:
        captured["summary_df"] = summary_df
        captured["manifest_data"] = manifest_data
        captured["run_key"] = run_key

    availability = render_comparison(
        summary=summary_df,
        manifest_data=manifest_data,
        xlsx=str(current_output),
        render_panel=_fake_render_panel,
    )

    assert availability.available is True
    assert captured["summary_df"] is summary_df
    assert captured["manifest_data"] == manifest_data
    assert captured["run_key"] == f"{current_output.resolve()}::{prior_manifest.resolve()}"
    assert fake_st.subheaders == ["LLM Comparison"]
    assert fake_st.infos == []
