from __future__ import annotations

import json
import runpy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import dashboard.components.comparison_llm as comparison_module


class _FakeContext:
    def __enter__(self) -> "_FakeContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


@dataclass
class _DownloadCall:
    label: str
    data: str
    file_name: str
    mime: str


class FakeStreamlit:
    def __init__(self, *, button_value: bool) -> None:
        self.session_state: dict[str, Any] = {}
        self.button_value = button_value
        self.downloads: list[_DownloadCall] = []
        self.messages: list[tuple[str, str]] = []

    def subheader(self, message: str) -> None:
        self.messages.append(("subheader", message))

    def caption(self, message: str) -> None:
        self.messages.append(("caption", message))

    def info(self, message: str) -> None:
        self.messages.append(("info", message))

    def markdown(self, message: str) -> None:
        self.messages.append(("markdown", message))

    def text_area(self, label: str, value: str, key: str, help: str) -> str:
        self.session_state.setdefault(key, value)
        return str(self.session_state[key])

    def expander(self, label: str, expanded: bool = False) -> _FakeContext:
        self.messages.append(("expander", label))
        return _FakeContext()

    def selectbox(self, label: str, options: list[str], index: int, key: str) -> str:
        self.session_state.setdefault(key, options[index])
        return str(self.session_state[key])

    def text_input(
        self, label: str, value: str, key: str, type: str | None = None, help: str | None = None
    ) -> str:
        self.session_state.setdefault(key, value)
        return str(self.session_state[key])

    def button(self, label: str, key: str | None = None) -> bool:
        return self.button_value

    def spinner(self, text: str) -> _FakeContext:
        self.messages.append(("spinner", text))
        return _FakeContext()

    def error(self, message: str) -> None:
        self.messages.append(("error", message))

    def columns(self, n: int) -> list[_FakeContext]:
        return [_FakeContext() for _ in range(n)]

    def download_button(self, label: str, data: str, file_name: str, mime: str) -> None:
        self.downloads.append(_DownloadCall(label=label, data=data, file_name=file_name, mime=mime))


def test_render_comparison_llm_panel_shows_default_message(monkeypatch) -> None:
    fake_st = FakeStreamlit(button_value=False)
    monkeypatch.setattr(comparison_module, "st", fake_st)

    comparison_module.render_comparison_llm_panel(
        summary_df=pd.DataFrame({"monthly_TE": [0.02]}),
        manifest_data={"previous_run": "prior_manifest.json"},
        run_key="run-a-vs-run-b",
    )

    assert "comparison_llm_cache" in fake_st.session_state
    assert any(kind == "subheader" and "LLM Comparison" in text for kind, text in fake_st.messages)
    assert any(kind == "info" and "Click Compare Runs" in text for kind, text in fake_st.messages)


def test_render_comparison_llm_panel_generates_output_and_downloads(monkeypatch) -> None:
    fake_st = FakeStreamlit(button_value=True)
    question = "Explain the rise in tracking error"
    fake_st.session_state[f"{comparison_module._QUESTIONS_KEY}::run-a-vs-run-b"] = question
    captured: dict[str, Any] = {}
    monkeypatch.setattr(comparison_module, "st", fake_st)
    monkeypatch.setattr(comparison_module, "default_api_key", lambda provider: "test-key")
    monkeypatch.setattr(comparison_module, "resolve_api_key_input", lambda raw: raw)
    monkeypatch.setattr(comparison_module, "resolve_llm_provider_config", lambda **kwargs: object())
    monkeypatch.setattr(
        comparison_module,
        "compare_runs",
        lambda **kwargs: captured.update(kwargs)
        or (
            "Generated comparison",
            "https://smith.langchain.com/r/trace-id",
            comparison_module.CompareRunsPayload(
                config_diff="- seed: 1 -> 2",
                metric_catalog_a={"monthly_TE": 0.02},
                metric_catalog_b={"monthly_TE": 0.01},
                prompt="prompt",
                questions="questions",
                prior_manifest_path="prior_manifest.json",
                prior_summary_path="prior.xlsx",
            ),
        ),
    )
    monkeypatch.setattr(comparison_module, "load_prior_manifest", lambda manifest_data: ({}, None))
    monkeypatch.setattr(
        comparison_module,
        "load_prior_summary",
        lambda *args, **kwargs: (pd.DataFrame({"monthly_TE": [0.01]}), None),
    )

    comparison_module.render_comparison_llm_panel(
        summary_df=pd.DataFrame({"monthly_TE": [0.02]}),
        manifest_data={"previous_run": "prior_manifest.json"},
        run_key="run-a-vs-run-b",
    )

    assert any(
        kind == "markdown" and "Generated comparison" in text for kind, text in fake_st.messages
    )
    assert any(kind == "caption" and "Trace URL" in text for kind, text in fake_st.messages)
    assert any(item.label == "Download TXT" for item in fake_st.downloads)
    assert any(item.label == "Download JSON" for item in fake_st.downloads)

    txt_export = next(item for item in fake_st.downloads if item.label == "Download TXT")
    assert "Config Diff" in txt_export.data
    assert "- seed: 1 -> 2" in txt_export.data
    assert "Trace URL: https://smith.langchain.com/r/trace-id" in txt_export.data

    json_export = next(item for item in fake_st.downloads if item.label == "Download JSON")
    assert '"config_diff": "- seed: 1 -> 2"' in json_export.data
    assert '"trace_url": "https://smith.langchain.com/r/trace-id"' in json_export.data

    assert captured["questions"] == question


def test_same_path_rerun_invalidates_comparison(monkeypatch, tmp_path) -> None:
    import pa_core.llm.compare_runs as backend
    from contextlib import nullcontext
    from dashboard.components import llm_settings

    results_module = runpy.run_path(str(Path("dashboard/pages/4_Results.py")))
    render_comparison_panel = results_module["_render_comparison_panel"]

    fake_st = FakeStreamlit(button_value=True)
    render_comparison_panel.__globals__["st"] = fake_st
    monkeypatch.setattr(comparison_module, "st", fake_st)

    env = {
        "PA_LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "synthetic-openai-secret",
        "AZURE_OPENAI_API_KEY": "synthetic-azure-secret",
        "PA_LLM_BASE_URL": "https://initial.openai.azure.com",
        "PA_LLM_API_VERSION": "2025-01-01",
    }
    monkeypatch.setattr(llm_settings, "read_secret", env.get)
    monkeypatch.setattr(backend, "langsmith_tracing_context", lambda **kwargs: nullcontext())
    monkeypatch.setattr(backend, "record_fleet_event", lambda *args, **kwargs: None)

    calls: list[dict[str, Any]] = []

    def fake_invoke(prompt: str, **kwargs: Any) -> str:
        provider_config = kwargs.get("provider_config")
        calls.append(
            {
                "prompt": prompt,
                "provider": getattr(provider_config, "provider_name", kwargs.get("provider_name")),
                "model": getattr(provider_config, "model_name", kwargs.get("model_name")),
            }
        )
        return f"Narrative #{len(calls)}"

    monkeypatch.setattr(backend, "_invoke_comparison_llm", fake_invoke)

    current_output = tmp_path / "Outputs.xlsx"
    prior_output = tmp_path / "PriorOutputs.xlsx"
    prior_manifest_path = tmp_path / "prior_manifest.json"

    current_summary = pd.DataFrame({"monthly_TE": [0.02], "monthly_CVaR": [-0.03]})
    prior_summary = pd.DataFrame({"monthly_TE": [0.01], "monthly_CVaR": [-0.02]})
    current_summary.to_excel(current_output, sheet_name="Summary", index=False)
    prior_summary.to_excel(prior_output, sheet_name="Summary", index=False)

    prior_manifest: dict[str, Any] = {
        "seed": 11,
        "run_name": "prior-run-1",
        "config_hash": "prior-cfg-1",
        "config": {"theta_extpa": 0.2},
        "cli_args": {"output": str(prior_output), "n_sims": 100},
    }
    prior_manifest_path.write_text(json.dumps(prior_manifest))

    current_manifest: dict[str, Any] = {
        "seed": 22,
        "run_name": "current-run-1",
        "config_hash": "current-cfg-1",
        "config": {"theta_extpa": 0.5},
        "cli_args": {"output": str(current_output), "n_sims": 100},
        "previous_run": str(prior_manifest_path),
    }
    run_key = f"{current_output.resolve()}::{prior_manifest_path.resolve()}"

    def render(*, button: bool) -> None:
        fake_st.button_value = button
        fake_st.messages.clear()
        fake_st.downloads.clear()
        render_comparison_panel(
            summary=current_summary,
            manifest_data=current_manifest,
            xlsx=str(current_output),
            render_panel=comparison_module.render_comparison_llm_panel,
        )

    def assert_stale_hidden(expected_calls: int) -> None:
        assert len(calls) == expected_calls
        assert not any(kind == "markdown" for kind, _ in fake_st.messages)
        assert any(
            kind == "info" and comparison_module._DEFAULT_TEXT in text
            for kind, text in fake_st.messages
        )
        assert fake_st.downloads == []

    def assert_active_output(expected_calls: int) -> tuple[str, dict[str, Any]]:
        assert len(calls) == expected_calls
        expected_narrative = f"Narrative #{expected_calls}"
        assert any(
            kind == "markdown" and expected_narrative in text for kind, text in fake_st.messages
        )
        assert len(fake_st.downloads) == 2
        txt_data = next(item.data for item in fake_st.downloads if item.mime == "text/plain")
        json_data = json.loads(
            next(item.data for item in fake_st.downloads if item.mime == "application/json")
        )
        assert expected_narrative in txt_data
        assert json_data["text"] == expected_narrative
        assert json_data["comparison_id"]
        assert json_data["comparison_id"] in txt_data
        assert all("synthetic-" not in item.data for item in fake_st.downloads)
        return txt_data, json_data

    # 1. Initial generation at the two paths.
    render(button=True)
    _, first_json = assert_active_output(1)

    # 2. Second render: same-path current summary replacement must hide old narrative/downloads.
    current_summary = pd.DataFrame({"monthly_TE": [0.04], "monthly_CVaR": [-0.05]})
    current_summary.to_excel(current_output, sheet_name="Summary", index=False)
    render(button=False)
    assert_stale_hidden(1)

    render(button=True)
    _, second_json = assert_active_output(2)
    assert second_json["comparison_id"] != first_json["comparison_id"]
    assert second_json["metric_catalog_a"]["monthly_TE"] == pytest.approx(0.04)

    # Unchanged-input reuse (both passive and active renders).
    render(button=False)
    _, reused_passive_json = assert_active_output(2)
    assert reused_passive_json["comparison_id"] == second_json["comparison_id"]

    render(button=True)
    _, reused_active_json = assert_active_output(2)
    assert reused_active_json["comparison_id"] == second_json["comparison_id"]

    # 3. Same-path prior summary replacement invalidates stale narrative and downloads.
    prior_summary = pd.DataFrame({"monthly_TE": [0.015], "monthly_CVaR": [-0.025]})
    prior_summary.to_excel(prior_output, sheet_name="Summary", index=False)
    render(button=False)
    assert_stale_hidden(2)

    render(button=True)
    _, third_json = assert_active_output(3)
    assert third_json["comparison_id"] != second_json["comparison_id"]
    assert third_json["metric_catalog_b"]["monthly_TE"] == pytest.approx(0.015)

    # 4. Same-path current manifest change invalidates stale narrative and downloads.
    current_manifest = {
        **current_manifest,
        "seed": 33,
        "config": {"theta_extpa": 0.75},
    }
    render(button=False)
    assert_stale_hidden(3)

    render(button=True)
    txt_after_curr_manifest, fourth_json = assert_active_output(4)
    assert fourth_json["comparison_id"] != third_json["comparison_id"]
    assert "config.theta_extpa: 0.2 -> 0.75" in fourth_json["config_diff"]
    assert "config.theta_extpa: 0.2 -> 0.75" in txt_after_curr_manifest

    # 5. Same-path prior manifest replacement invalidates stale narrative and downloads.
    prior_manifest = {
        **prior_manifest,
        "seed": 19,
        "config": {"theta_extpa": 0.35},
    }
    prior_manifest_path.write_text(json.dumps(prior_manifest))
    render(button=False)
    assert_stale_hidden(4)

    render(button=True)
    txt_after_prior_manifest, fifth_json = assert_active_output(5)
    assert fifth_json["comparison_id"] != fourth_json["comparison_id"]
    assert "config.theta_extpa: 0.35 -> 0.75" in fifth_json["config_diff"]
    assert "config.theta_extpa: 0.35 -> 0.75" in txt_after_prior_manifest

    # 6. Credential changes alone do NOT invalidate cached output or leak into identity/exports.
    fake_st.session_state[f"{comparison_module._API_KEY_KEY}::{run_key}::openai"] = (
        "synthetic-rotated-openai-key"
    )
    render(button=False)
    _, rotated_key_json = assert_active_output(5)
    assert rotated_key_json["comparison_id"] == fifth_json["comparison_id"]
    assert "synthetic-rotated-openai-key" not in rotated_key_json["comparison_id"]

    # 7. Request-setting changes: questions, effective model, provider, endpoint, API version.
    fake_st.session_state[f"{comparison_module._QUESTIONS_KEY}::{run_key}"] = (
        "Focus on CVaR movement after parameter change."
    )
    render(button=False)
    assert_stale_hidden(5)
    render(button=True)
    _, sixth_json = assert_active_output(6)
    assert sixth_json["questions"] == "Focus on CVaR movement after parameter change."

    fake_st.session_state[f"{comparison_module._MODEL_KEY}::{run_key}"] = "gpt-4.1-mini"
    render(button=False)
    assert_stale_hidden(6)
    render(button=True)
    assert_active_output(7)

    fake_st.session_state[f"{comparison_module._PROVIDER_KEY}::{run_key}"] = "azure_openai"
    fake_st.session_state[f"{comparison_module._API_KEY_KEY}::{run_key}::azure_openai"] = (
        "synthetic-azure-secret"
    )
    render(button=False)
    assert_stale_hidden(7)
    render(button=True)
    assert_active_output(8)

    fake_st.session_state[f"{comparison_module._BASE_URL_KEY}::{run_key}"] = (
        "https://updated.openai.azure.com"
    )
    render(button=False)
    assert_stale_hidden(8)
    render(button=True)
    assert_active_output(9)

    fake_st.session_state[f"{comparison_module._API_VERSION_KEY}::{run_key}"] = (
        "2025-04-01-preview"
    )
    render(button=False)
    assert_stale_hidden(9)
    render(button=True)
    assert_active_output(10)


def test_comparison_panel_passes_config_snapshot_diff_to_provider_and_exports(
    monkeypatch, tmp_path
):
    import pa_core.llm.compare_runs as backend
    from contextlib import nullcontext

    from pa_core.llm.provider import LLMProviderConfig

    prior_output = tmp_path / "prior.xlsx"
    pd.DataFrame({"monthly_TE": [0.01]}).to_excel(prior_output, sheet_name="Summary", index=False)
    prior_manifest_path = tmp_path / "prior.json"
    prior_manifest_path.write_text(
        json.dumps(
            {
                "seed": 17,
                "config": {"theta_extpa": 0.2},
                "cli_args": {"output": str(prior_output)},
            }
        )
    )
    fake_st = FakeStreamlit(button_value=True)
    monkeypatch.setattr(comparison_module, "st", fake_st)
    monkeypatch.setattr(comparison_module, "default_api_key", lambda provider: "test-key")
    monkeypatch.setattr(comparison_module, "resolve_api_key_input", lambda raw: raw)
    monkeypatch.setattr(
        comparison_module,
        "resolve_llm_provider_config",
        lambda **kwargs: LLMProviderConfig(
            provider_name="openai", credentials={"api_key": "test-key"}
        ),
    )
    monkeypatch.setattr(backend, "langsmith_tracing_context", lambda **kwargs: nullcontext())
    monkeypatch.setattr(backend, "record_fleet_event", lambda *args, **kwargs: None)
    captured_prompts: list[str] = []
    monkeypatch.setattr(
        backend,
        "_invoke_comparison_llm",
        lambda prompt, **kwargs: captured_prompts.append(prompt) or "Provider comparison",
    )

    summary_df = pd.DataFrame({"monthly_TE": [0.02]})
    manifest_data = {
        "seed": 17,
        "config": {"theta_extpa": 0.8},
        "previous_run": str(prior_manifest_path),
    }
    comparison_module.render_comparison_llm_panel(
        summary_df=summary_df,
        manifest_data=manifest_data,
        run_key="config-snapshot",
    )

    identity = comparison_module._comparison_identity(
        summary_df=summary_df,
        manifest_data=manifest_data,
        run_key="config-snapshot",
    )
    payload = fake_st.session_state[f"comparison_payload::{identity}"]
    assert captured_prompts == [payload.prompt]
    assert "config.theta_extpa: 0.2 -> 0.8" in captured_prompts[0]
    txt_export = next(item.data for item in fake_st.downloads if item.mime == "text/plain")
    json_export = json.loads(
        next(item.data for item in fake_st.downloads if item.mime == "application/json")
    )
    assert "config.theta_extpa: 0.2 -> 0.8" in txt_export
    assert "config.theta_extpa: 0.2 -> 0.8" in json_export["config_diff"]


def test_render_comparison_llm_panel_passes_azure_provider_config(monkeypatch) -> None:
    fake_st = FakeStreamlit(button_value=True)
    run_key = "run-a-vs-run-b"
    fake_st.session_state[f"{comparison_module._PROVIDER_KEY}::{run_key}"] = "azure_openai"
    fake_st.session_state[f"{comparison_module._API_KEY_KEY}::{run_key}::azure_openai"] = (
        "sk-azure-test"
    )
    fake_st.session_state[f"{comparison_module._BASE_URL_KEY}::{run_key}"] = (
        "https://example.openai.azure.com"
    )
    fake_st.session_state[f"{comparison_module._API_VERSION_KEY}::{run_key}"] = "2025-01-01-preview"
    monkeypatch.setattr(comparison_module, "st", fake_st)
    monkeypatch.setattr(comparison_module, "load_prior_manifest", lambda manifest_data: ({}, None))
    monkeypatch.setattr(
        comparison_module,
        "load_prior_summary",
        lambda *args, **kwargs: (pd.DataFrame({"monthly_TE": [0.01]}), None),
    )

    captured: dict[str, Any] = {}

    def fake_compare_runs(**kwargs):
        captured.update(kwargs)
        return (
            "Generated comparison",
            None,
            comparison_module.CompareRunsPayload(
                config_diff="- seed: 1 -> 2",
                metric_catalog_a={"monthly_TE": 0.02},
                metric_catalog_b={"monthly_TE": 0.01},
                prompt="prompt",
                questions="questions",
                prior_manifest_path="prior_manifest.json",
                prior_summary_path="prior.xlsx",
            ),
        )

    monkeypatch.setattr(comparison_module, "compare_runs", fake_compare_runs)

    comparison_module.render_comparison_llm_panel(
        summary_df=pd.DataFrame({"monthly_TE": [0.02]}),
        manifest_data={"previous_run": "prior_manifest.json"},
        run_key=run_key,
    )

    provider_config = captured["provider_config"]
    assert provider_config.provider_name == "azure_openai"
    assert provider_config.credentials["api_key"] == "sk-azure-test"
    assert provider_config.credentials["azure_endpoint"] == "https://example.openai.azure.com"
    assert provider_config.credentials["api_version"] == "2025-01-01-preview"
    assert captured["provider_name"] == "azure_openai"


def test_render_comparison_llm_panel_ignores_stale_azure_fields_for_openai(monkeypatch) -> None:
    fake_st = FakeStreamlit(button_value=True)
    run_key = "run-a-vs-run-b"
    fake_st.session_state[f"{comparison_module._PROVIDER_KEY}::{run_key}"] = "openai"
    fake_st.session_state[f"{comparison_module._API_KEY_KEY}::{run_key}::openai"] = "sk-openai-test"
    fake_st.session_state[f"{comparison_module._BASE_URL_KEY}::{run_key}"] = (
        "https://example.openai.azure.com"
    )
    fake_st.session_state[f"{comparison_module._API_VERSION_KEY}::{run_key}"] = "2025-01-01-preview"
    monkeypatch.setattr(comparison_module, "st", fake_st)
    monkeypatch.setattr(comparison_module, "load_prior_manifest", lambda manifest_data: ({}, None))
    monkeypatch.setattr(
        comparison_module,
        "load_prior_summary",
        lambda *args, **kwargs: (pd.DataFrame({"monthly_TE": [0.01]}), None),
    )

    captured_config_kwargs: dict[str, Any] = {}

    def fake_resolve_llm_provider_config(**kwargs):
        captured_config_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(
        comparison_module, "resolve_llm_provider_config", fake_resolve_llm_provider_config
    )
    monkeypatch.setattr(
        comparison_module,
        "compare_runs",
        lambda **kwargs: (
            "Generated comparison",
            None,
            comparison_module.CompareRunsPayload(
                config_diff="- seed: 1 -> 2",
                metric_catalog_a={"monthly_TE": 0.02},
                metric_catalog_b={"monthly_TE": 0.01},
                prompt="prompt",
                questions="questions",
                prior_manifest_path="prior_manifest.json",
                prior_summary_path="prior.xlsx",
            ),
        ),
    )

    comparison_module.render_comparison_llm_panel(
        summary_df=pd.DataFrame({"monthly_TE": [0.02]}),
        manifest_data={"previous_run": "prior_manifest.json"},
        run_key=run_key,
    )

    assert captured_config_kwargs["provider"] == "openai"
    assert captured_config_kwargs["base_url"] is None
    assert captured_config_kwargs["api_version"] is None


@pytest.mark.parametrize("destination", ["anthropic", "azure_openai"])
def test_comparison_provider_defaults_switches_and_secret_free_exports(
    monkeypatch, socket_connect_guard, destination
):
    import importlib
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace

    import pa_core.llm.provider as provider_module
    from dashboard.components import llm_settings

    class Prompt:
        @staticmethod
        def from_messages(messages):
            return Prompt()

        def __or__(self, client):
            return client

    monkeypatch.setitem(
        sys.modules, "langchain_core.prompts", SimpleNamespace(ChatPromptTemplate=Prompt)
    )
    core = importlib.import_module("pa_core.llm.compare_runs")
    env = {
        "PA_LLM_PROVIDER": "anthropic",
        "OPENAI_API_KEY": "synthetic-openai",
        "CLAUDE_API_STRANSKE": "synthetic-anthropic",
        "AZURE_OPENAI_API_KEY": "synthetic-azure",
        "PA_LLM_BASE_URL": "https://example.openai.azure.com",
        "PA_LLM_API_VERSION": "2025-01-01",
    }
    monkeypatch.setattr(llm_settings, "read_secret", env.get)
    fake = FakeStreamlit(button_value=True)
    run_key = "credential-switch"
    fake.session_state[f"{comparison_module._API_KEY_KEY}::{run_key}"] = "synthetic-legacy-openai"
    monkeypatch.setattr(comparison_module, "st", fake)

    def prior(manifest):
        return {"seed": 0}, None

    monkeypatch.setattr(comparison_module, "load_prior_manifest", prior)
    monkeypatch.setattr(
        comparison_module,
        "load_prior_summary",
        lambda *args, **kwargs: (pd.DataFrame({"monthly_TE": [0.01]}), None),
    )
    monkeypatch.setattr(core, "load_prior_manifest", prior)
    monkeypatch.setattr(
        core,
        "load_prior_summary",
        lambda *args, **kwargs: (pd.DataFrame({"monthly_TE": [0.01]}), None),
    )
    monkeypatch.setattr(core, "langsmith_tracing_context", lambda **kwargs: nullcontext())
    monkeypatch.setattr(core, "record_fleet_event", lambda *args, **kwargs: None)
    captured = []

    def factory(config):
        captured.append(config)
        return SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content="Safe comparison"))

    monkeypatch.setattr(provider_module, "create_llm", factory)

    def render():
        comparison_module.render_comparison_llm_panel(
            summary_df=pd.DataFrame({"monthly_TE": [0.02]}),
            manifest_data={"seed": 1},
            run_key=run_key,
        )

    render()
    assert fake.session_state[f"{comparison_module._PROVIDER_KEY}::{run_key}"] == "anthropic"
    assert captured[-1].credentials["api_key"] == "synthetic-anthropic"
    assert f"{comparison_module._API_KEY_KEY}::{run_key}" not in fake.session_state
    fake.session_state[f"{comparison_module._PROVIDER_KEY}::{run_key}"] = "openai"
    fake.session_state[f"{comparison_module._API_KEY_KEY}::{run_key}::openai"] = (
        "synthetic-typed-openai"
    )
    render()
    assert captured[-1].credentials["api_key"] == "synthetic-typed-openai"
    fake.session_state[f"{comparison_module._PROVIDER_KEY}::{run_key}"] = destination
    if destination == "azure_openai":
        # Azure has no automatic provider key in this dashboard's contract.
        render()
        assert len(captured) == 2
        fake.session_state[f"{comparison_module._API_KEY_KEY}::{run_key}::azure_openai"] = (
            "synthetic-azure"
        )
    render()
    assert captured[-1].provider_name == destination
    assert (
        captured[-1].credentials["api_key"]
        == env["CLAUDE_API_STRANSKE" if destination == "anthropic" else "AZURE_OPENAI_API_KEY"]
    )
    if destination == "azure_openai":
        assert captured[-1].credentials["azure_endpoint"] == env["PA_LLM_BASE_URL"]
        assert captured[-1].credentials["api_version"] == env["PA_LLM_API_VERSION"]
    assert all("synthetic" not in item.data for item in fake.downloads)
    assert socket_connect_guard[0] == []


def test_tracing_only_never_constructs_comparison_client(monkeypatch, socket_connect_guard):
    from dashboard.components import llm_settings

    monkeypatch.setattr(llm_settings, "read_secret", {"LANGSMITH_API_KEY": "synthetic-tracing"}.get)
    fake = FakeStreamlit(button_value=True)
    monkeypatch.setattr(comparison_module, "st", fake)
    monkeypatch.setattr(comparison_module, "load_prior_manifest", lambda manifest: ({}, None))
    monkeypatch.setattr(
        comparison_module,
        "load_prior_summary",
        lambda *args, **kwargs: (pd.DataFrame({"monthly_TE": [0.01]}), None),
    )
    calls = []
    monkeypatch.setattr(comparison_module, "compare_runs", lambda **kwargs: calls.append(kwargs))
    comparison_module.render_comparison_llm_panel(
        summary_df=pd.DataFrame({"monthly_TE": [0.02]}), manifest_data={}, run_key="tracing-only"
    )
    assert calls == []
    assert any("No API key found" in message for _, message in fake.messages)
    assert fake.downloads == []
    assert socket_connect_guard[0] == []


def test_comparison_fallback_keeps_display_downloads_and_logs_secret_free(monkeypatch, caplog):
    import importlib
    from contextlib import nullcontext

    backend = importlib.import_module("pa_core.llm.compare_runs")
    fake_st = FakeStreamlit(button_value=True)
    marker = "synthetic-dashboard-secret-2363"
    monkeypatch.setattr(comparison_module, "st", fake_st)
    monkeypatch.setattr(comparison_module, "default_api_key", lambda provider: marker)
    monkeypatch.setattr(comparison_module, "resolve_api_key_input", lambda raw: raw)
    monkeypatch.setattr(comparison_module, "load_prior_manifest", lambda _: ({}, None))
    monkeypatch.setattr(
        comparison_module,
        "load_prior_summary",
        lambda *a, **k: (pd.DataFrame({"monthly_TE": [0.01]}), None),
    )
    monkeypatch.setattr(backend, "load_prior_manifest", lambda _: ({}, None))
    monkeypatch.setattr(
        backend, "load_prior_summary", lambda *a, **k: (pd.DataFrame({"monthly_TE": [0.01]}), None)
    )
    monkeypatch.setattr(backend, "langsmith_tracing_context", lambda **k: nullcontext())
    monkeypatch.setattr(backend, "record_fleet_event", lambda *a, **k: None)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    def fail(*args, **kwargs):
        raise RuntimeError(f"authentication failed for {marker}")

    monkeypatch.setattr(backend, "_invoke_comparison_llm", fail)
    summary_df = pd.DataFrame({"monthly_TE": [0.02]})
    manifest_data = {"previous_run": "prior.json"}
    with caplog.at_level("WARNING", logger=backend.__name__):
        comparison_module.render_comparison_llm_panel(
            summary_df=summary_df,
            manifest_data=manifest_data,
            run_key="redaction-2363",
        )
    assert any(
        kind == "markdown" and "Run-to-run comparison" in text for kind, text in fake_st.messages
    )
    assert len(fake_st.downloads) == 2
    assert all(marker not in text for _, text in fake_st.messages)
    assert all(marker not in item.data for item in fake_st.downloads)
    assert marker not in caplog.text
    identity = comparison_module._comparison_identity(
        summary_df=summary_df,
        manifest_data=manifest_data,
        run_key="redaction-2363",
    )
    assert fake_st.session_state[f"comparison_payload::{identity}"].status == "fallback"


def test_comparison_missing_extra_labels_fallback_and_keeps_offline_output(monkeypatch) -> None:
    fake_st = FakeStreamlit(button_value=True)
    monkeypatch.setattr(comparison_module, "st", fake_st)
    monkeypatch.setattr(comparison_module, "load_prior_manifest", lambda _: ({}, None))
    monkeypatch.setattr(
        comparison_module,
        "load_prior_summary",
        lambda *args, **kwargs: (pd.DataFrame({"monthly_TE": [0.01]}), None),
    )
    monkeypatch.setattr(comparison_module, "resolve_llm_provider_config", lambda **kwargs: object())
    monkeypatch.setattr(
        comparison_module,
        "compare_runs",
        lambda **kwargs: (
            "Offline comparison remains available.",
            None,
            comparison_module.CompareRunsPayload(
                config_diff="- seed: 1 -> 2",
                metric_catalog_a={},
                metric_catalog_b={},
                prompt="prompt",
                questions="questions",
                prior_manifest_path="prior.json",
                prior_summary_path="prior.xlsx",
                status="fallback",
                error_category="ModuleNotFoundError",
            ),
        ),
    )

    comparison_module.render_comparison_llm_panel(
        summary_df=pd.DataFrame({"monthly_TE": [0.02]}),
        manifest_data={"previous_run": "prior.json"},
        run_key="missing-extra",
    )

    assert any(
        kind == "markdown" and "Offline comparison remains available" in text
        for kind, text in fake_st.messages
    )
    assert any(
        kind == "caption" and "fallback (ModuleNotFoundError)" in text
        for kind, text in fake_st.messages
    )
    assert any(
        kind == "info" and "install .[llm]" in text.lower() for kind, text in fake_st.messages
    )
    assert len(fake_st.downloads) == 2
    json_download = next(item for item in fake_st.downloads if item.mime == "application/json")
    assert '"status": "fallback"' in json_download.data
    assert '"error_category": "ModuleNotFoundError"' in json_download.data
