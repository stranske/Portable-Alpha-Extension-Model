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
    prior_manifest = tmp_path / "prior_manifest.json"
    prior_manifest.write_text(json.dumps({"seed": 42}))
    result = check({"previous_run": str(prior_manifest)})

    assert result.available is True
    assert result.prior_manifest_path == prior_manifest
    assert result.message is None


class _FakeStreamlit:
    def __init__(self) -> None:
        self.subheaders: list[str] = []
        self.infos: list[str] = []

    def subheader(self, message: str) -> None:
        self.subheaders.append(message)

    def info(self, message: str) -> None:
        self.infos.append(message)


def test_render_comparison_panel_calls_component_when_previous_run_readable(tmp_path) -> None:
    module = _load_results_module()
    render_comparison = module["_render_comparison_panel"]
    fake_st = _FakeStreamlit()
    render_comparison.__globals__["st"] = fake_st

    current_output = tmp_path / "current.xlsx"
    current_output.write_bytes(b"placeholder")
    prior_manifest = tmp_path / "prior_manifest.json"
    prior_manifest.write_text(json.dumps({"seed": 42}))
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
