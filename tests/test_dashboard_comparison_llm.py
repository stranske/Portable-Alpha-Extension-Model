from __future__ import annotations

import json
from dataclasses import dataclass
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

    comparison_module.render_comparison_llm_panel(
        summary_df=pd.DataFrame({"monthly_TE": [0.02]}),
        manifest_data={
            "seed": 17,
            "config": {"theta_extpa": 0.8},
            "previous_run": str(prior_manifest_path),
        },
        run_key="config-snapshot",
    )

    payload = fake_st.session_state["comparison_payload::config-snapshot"]
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
    with caplog.at_level("WARNING", logger=backend.__name__):
        comparison_module.render_comparison_llm_panel(
            summary_df=pd.DataFrame({"monthly_TE": [0.02]}),
            manifest_data={"previous_run": "prior.json"},
            run_key="redaction-2363",
        )
    assert any(
        kind == "markdown" and "Run-to-run comparison" in text for kind, text in fake_st.messages
    )
    assert len(fake_st.downloads) == 2
    assert all(marker not in text for _, text in fake_st.messages)
    assert all(marker not in item.data for item in fake_st.downloads)
    assert marker not in caplog.text
    assert fake_st.session_state["comparison_payload::redaction-2363"].status == "fallback"


def test_comparison_missing_extra_labels_fallback_and_keeps_offline_output(monkeypatch) -> None:
    fake_st = FakeStreamlit(button_value=True)
    monkeypatch.setattr(comparison_module, "st", fake_st)
    monkeypatch.setattr(comparison_module, "load_prior_manifest", lambda _: ({}, None))
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
