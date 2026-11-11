"""Coverage for the three gaps documented as exceptions for issue #2476.

The evidence in ``docs/evidence/issue-2476/README.md`` listed these as
unreachable on the production path:

- ``pa_core/facade.py`` statements 750/756 in ``export()`` (DataFrame and
  tuple agent-semantics values, pre-converted by
  ``_serialize_agent_semantics_input`` on the normal path);
- ``pa_core/pa.py`` arc 349->exit (a registry subcommand other than
  ``list``/``get``, which argparse cannot produce);
- ``dashboard/pages/4_Results.py`` arc 178->180 (an unavailable previous
  run carrying no message, which ``_check_previous_run_availability``
  never returns).

Each test drives the branch through the public function with the
collaborator that normally makes it unreachable replaced, and asserts the
observable result, so the branches are protected against regressions if
those collaborators change.
"""

from __future__ import annotations

import argparse
import runpy
import sys
import types
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

import pa_core.pa as pa_entry
import pa_core.reporting
from pa_core import facade


def _run_artifacts(agent_semantics: Any) -> facade.RunArtifacts:
    return facade.RunArtifacts(
        config=cast(Any, None),
        index_series=cast(Any, pd.Series(dtype=float)),
        returns={},
        summary=pd.DataFrame([{"Agent": "A"}]),
        inputs={"_agent_semantics_df": agent_semantics},
        raw_returns={},
    )


def _export_captured_inputs(
    monkeypatch: pytest.MonkeyPatch, artifacts: facade.RunArtifacts, tmp_path: Path
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def _fake_export_to_excel(inputs: dict[str, Any], *_args: Any, **_kwargs: Any) -> None:
        captured.update(inputs)

    monkeypatch.setattr(facade, "_serialize_agent_semantics_input", lambda _inputs: None)
    monkeypatch.setattr(pa_core.reporting, "export_to_excel", _fake_export_to_excel)
    result = facade.export(artifacts, tmp_path / "out.xlsx")
    assert result == tmp_path / "out.xlsx"
    return captured


def test_export_converts_dataframe_agent_semantics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifacts = _run_artifacts(pd.DataFrame([{"Agent": "A", "n": 1}, {"Agent": "B", "n": 2}]))
    captured = _export_captured_inputs(monkeypatch, artifacts, tmp_path)
    expected = [{"Agent": "A", "n": 1}, {"Agent": "B", "n": 2}]
    assert captured["_agent_semantics_df"] == expected
    assert artifacts.inputs["_agent_semantics_df"] == expected


def test_export_converts_tuple_agent_semantics_to_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifacts = _run_artifacts(({"Agent": "A"}, {"Agent": "B"}))
    captured = _export_captured_inputs(monkeypatch, artifacts, tmp_path)
    expected = [{"Agent": "A"}, {"Agent": "B"}]
    assert captured["_agent_semantics_df"] == expected
    assert type(captured["_agent_semantics_df"]) is list
    assert artifacts.inputs["_agent_semantics_df"] == expected


def test_pa_registry_unknown_subcommand_is_a_no_op(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _fake_parse_known_args(
        _self: argparse.ArgumentParser, _argv: Any = None, _namespace: Any = None
    ) -> tuple[argparse.Namespace, list[str]]:
        return argparse.Namespace(command="registry", registry_cmd="unknown"), []

    monkeypatch.setattr(argparse.ArgumentParser, "parse_known_args", _fake_parse_known_args)
    assert pa_entry.main(["registry", "unknown"]) is None
    assert capsys.readouterr().out == ""


class _ComparisonStreamlit:
    def __init__(self) -> None:
        self.subheaders: list[str] = []
        self.infos: list[str] = []

    def subheader(self, value: str) -> None:
        self.subheaders.append(value)

    def info(self, value: str) -> None:
        self.infos.append(value)


def _load_results_module() -> dict[str, Any]:
    module_name = "dashboard.pages.page_4_Results_remaining_gaps"
    module_dict = runpy.run_path(str(Path("dashboard/pages/4_Results.py")), run_name=module_name)
    if module_name not in sys.modules:
        module = types.ModuleType(module_name)
        module.__dict__.update(module_dict)
        sys.modules[module_name] = module
    return module_dict


def test_render_comparison_panel_without_message_shows_no_info() -> None:
    module = _load_results_module()
    fake_st = _ComparisonStreamlit()
    render_comparison = module["_render_comparison_panel"]
    availability_type = module["_PreviousRunAvailability"]
    unavailable_without_message = availability_type(
        available=False, prior_manifest_path=None, message=None
    )
    render_comparison.__globals__["st"] = fake_st
    render_comparison.__globals__["_check_previous_run_availability"] = (
        lambda _manifest_data: unavailable_without_message
    )

    def _fail_render_panel(**_kwargs: Any) -> None:
        raise AssertionError("render_panel must not run for an unavailable previous run")

    result = render_comparison(
        summary=pd.DataFrame({"monthly_TE": [0.01]}),
        manifest_data={"seed": 1},
        xlsx="current.xlsx",
        render_panel=_fail_render_panel,
    )
    assert result is unavailable_without_message
    assert fake_st.subheaders == ["LLM Comparison"]
    assert fake_st.infos == []
