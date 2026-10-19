from __future__ import annotations

import json
import runpy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import dashboard.components.explain_results as explain_module


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

    def info(self, message: str) -> None:
        self.messages.append(("info", message))

    def error(self, message: str) -> None:
        self.messages.append(("error", message))

    def markdown(self, text: str) -> None:
        self.messages.append(("markdown", text))

    def caption(self, text: str) -> None:
        self.messages.append(("caption", text))

    def columns(self, n: int) -> list[_FakeContext]:
        return [_FakeContext() for _ in range(n)]

    def download_button(self, label: str, data: str, file_name: str, mime: str) -> None:
        self.downloads.append(_DownloadCall(label=label, data=data, file_name=file_name, mime=mime))


def test_render_explain_results_caches_and_download_json_fields(monkeypatch) -> None:
    fake_st = FakeStreamlit(button_value=True)
    monkeypatch.setattr(explain_module, "st", fake_st)
    monkeypatch.setattr(explain_module, "default_api_key", lambda provider: "test-key")
    monkeypatch.setattr(explain_module, "resolve_api_key_input", lambda raw: raw)
    monkeypatch.setattr(
        explain_module,
        "resolve_llm_provider_config",
        lambda **kwargs: SimpleNamespace(
            provider_name=kwargs["provider"],
            model_name=kwargs.get("model") or "gpt-4o-mini",
        ),
    )

    call_count = {"count": 0}

    def _fake_explain(
        summary_df: pd.DataFrame,
        manifest: dict[str, Any] | None,
        *,
        questions: str,
        llm_config: Any,
    ):
        call_count["count"] += 1
        assert questions
        assert llm_config.provider_name == "openai"
        assert llm_config.model_name == "gpt-4o-mini"
        return (
            "Explanation text",
            "https://smith.langchain.com/r/trace123",
            {"rows": len(summary_df)},
        )

    monkeypatch.setattr(explain_module, "explain_results_details", _fake_explain)

    summary_df = pd.DataFrame({"monthly_TE": [0.01, 0.02], "monthly_CVaR": [-0.03, -0.02]})
    manifest = {"seed": 123}

    explain_module.render_explain_results_panel(
        summary_df=summary_df,
        manifest=manifest,
        xlsx_path="/tmp/Outputs.xlsx",
    )
    explain_module.render_explain_results_panel(
        summary_df=summary_df,
        manifest=manifest,
        xlsx_path="/tmp/Outputs.xlsx",
    )

    assert call_count["count"] == 1

    json_downloads = [item for item in fake_st.downloads if item.file_name.endswith(".json")]
    assert json_downloads
    payload = json.loads(json_downloads[0].data)
    assert "trace_url" in payload
    assert "created_at" in payload
    assert "inputs_summary" in payload
    assert payload["inputs_summary"]["manifest_seed"] == 123
    assert "api_key" not in payload["inputs_summary"]


def test_results_page_llm_fallback_message(monkeypatch) -> None:
    module = runpy.run_path(str(Path("dashboard/pages/4_Results.py")))

    class _PageFakeSt:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def info(self, message: str) -> None:
            self.messages.append(message)

    fake_st = _PageFakeSt()

    def _raise(*args: Any, **kwargs: Any) -> None:
        raise ModuleNotFoundError("langchain_openai")

    module["_render_explain_results"].__globals__["render_explain_results_panel"] = _raise
    module["_render_explain_results"].__globals__["st"] = fake_st

    module["_render_explain_results"](
        summary=pd.DataFrame({"x": [1.0]}),
        manifest_data={"seed": 1},
        xlsx="Outputs.xlsx",
    )

    assert any("LLM features unavailable" in message for message in fake_st.messages)
    assert any("install .[llm]" in message.lower() for message in fake_st.messages)


def test_render_explain_results_panel_reports_missing_llm_extra(monkeypatch) -> None:
    fake_st = FakeStreamlit(button_value=True)
    monkeypatch.setattr(explain_module, "st", fake_st)
    monkeypatch.setattr(explain_module, "default_api_key", lambda provider: "test-key")
    monkeypatch.setattr(explain_module, "resolve_api_key_input", lambda raw: raw)
    monkeypatch.setattr(
        explain_module,
        "resolve_llm_provider_config",
        lambda **kwargs: SimpleNamespace(provider_name="openai", model_name="gpt-4o-mini"),
    )

    def _raise_missing_extra(*args: Any, **kwargs: Any) -> None:
        raise ModuleNotFoundError("langchain_openai")

    monkeypatch.setattr(explain_module, "explain_results_details", _raise_missing_extra)

    explain_module.render_explain_results_panel(
        summary_df=pd.DataFrame({"monthly_TE": [0.01]}),
        manifest={"seed": 1},
        xlsx_path="/tmp/Outputs.xlsx",
    )

    assert any(
        kind == "error" and "LLM features unavailable" in text for kind, text in fake_st.messages
    )
    assert any(
        kind == "error" and "install .[llm]" in text.lower() for kind, text in fake_st.messages
    )


def test_explain_results_missing_extra_is_not_cached_and_retries(monkeypatch) -> None:
    import pa_core.llm.result_explain as result_explain

    fake_st = FakeStreamlit(button_value=True)
    monkeypatch.setattr(explain_module, "st", fake_st)
    monkeypatch.setattr(explain_module, "default_api_key", lambda provider: "test-key")
    monkeypatch.setattr(explain_module, "resolve_api_key_input", lambda raw: raw)
    monkeypatch.setattr(
        explain_module,
        "resolve_llm_provider_config",
        lambda **kwargs: SimpleNamespace(
            provider_name="openai",
            model_name="gpt-4o-mini",
            credentials={"api_key": "test-key"},
        ),
    )
    monkeypatch.setattr(result_explain, "_is_tracing_enabled", lambda enabled: False)
    monkeypatch.setattr(result_explain, "record_fleet_event", lambda *args, **kwargs: None)

    attempts = 0

    def fake_create_llm(config):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ModuleNotFoundError("langchain_openai")
        return SimpleNamespace(
            invoke=lambda prompt: SimpleNamespace(content="Recovered explanation")
        )

    monkeypatch.setattr(result_explain, "create_llm", fake_create_llm)
    kwargs = {
        "summary_df": pd.DataFrame({"monthly_TE": [0.01]}),
        "manifest": {"seed": 1},
        "xlsx_path": "/tmp/Outputs.xlsx",
    }

    explain_module.render_explain_results_panel(**kwargs)

    assert attempts == 1
    assert fake_st.session_state[explain_module._CACHE_KEY] == {}
    assert fake_st.downloads == []
    assert any(
        kind == "error" and "install .[llm]" in text.lower() for kind, text in fake_st.messages
    )

    explain_module.render_explain_results_panel(**kwargs)

    assert attempts == 2
    assert fake_st.downloads
    assert any(
        kind == "markdown" and "Recovered explanation" in text for kind, text in fake_st.messages
    )


def test_explain_results_empty_error_is_not_cached_and_retries(monkeypatch) -> None:
    import pa_core.llm.result_explain as result_explain

    fake_st = FakeStreamlit(button_value=True)
    monkeypatch.setattr(explain_module, "st", fake_st)
    monkeypatch.setattr(explain_module, "default_api_key", lambda provider: "test-key")
    monkeypatch.setattr(explain_module, "resolve_api_key_input", lambda raw: raw)
    monkeypatch.setattr(
        explain_module,
        "resolve_llm_provider_config",
        lambda **kwargs: SimpleNamespace(
            provider_name="openai",
            model_name="gpt-4o-mini",
            credentials={"api_key": "test-key"},
        ),
    )
    monkeypatch.setattr(result_explain, "_is_tracing_enabled", lambda enabled: False)
    monkeypatch.setattr(result_explain, "record_fleet_event", lambda *args, **kwargs: None)

    attempts = 0

    def fake_create_llm(config):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError()
        return SimpleNamespace(
            invoke=lambda prompt: SimpleNamespace(content="Recovered empty-error explanation")
        )

    monkeypatch.setattr(result_explain, "create_llm", fake_create_llm)
    kwargs = {
        "summary_df": pd.DataFrame({"monthly_TE": [0.01]}),
        "manifest": {"seed": 1},
        "xlsx_path": "/tmp/Outputs.xlsx",
    }

    explain_module.render_explain_results_panel(**kwargs)

    assert attempts == 1
    assert fake_st.session_state[explain_module._CACHE_KEY] == {}
    assert fake_st.downloads == []
    assert any(
        kind == "error" and text == "Failed to generate explanation."
        for kind, text in fake_st.messages
    )

    explain_module.render_explain_results_panel(**kwargs)

    assert attempts == 2
    assert fake_st.downloads
    assert any(
        kind == "markdown" and "Recovered empty-error explanation" in text
        for kind, text in fake_st.messages
    )


def test_explain_results_evicts_legacy_error_cache_and_retries(monkeypatch) -> None:
    fake_st = FakeStreamlit(button_value=True)
    monkeypatch.setattr(explain_module, "st", fake_st)
    monkeypatch.setattr(explain_module, "default_api_key", lambda provider: "test-key")
    monkeypatch.setattr(explain_module, "resolve_api_key_input", lambda raw: raw)
    monkeypatch.setattr(
        explain_module,
        "resolve_llm_provider_config",
        lambda **kwargs: SimpleNamespace(provider_name="openai", model_name="gpt-4o-mini"),
    )

    calls = 0

    def fake_explain(*args: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        return "Recovered legacy-cache explanation", None, {"status": "success"}

    monkeypatch.setattr(explain_module, "explain_results_details", fake_explain)
    summary_df = pd.DataFrame({"monthly_TE": [0.01]})
    manifest = {"seed": 1}
    questions = explain_module.DEFAULT_QUESTIONS
    payload_fingerprint = explain_module._payload_fingerprint(summary_df, manifest)
    cache_key = explain_module._cache_key(
        xlsx_path="/tmp/Outputs.xlsx",
        manifest=manifest,
        provider="openai",
        model="",
        questions=questions,
        payload_fingerprint=payload_fingerprint,
    )
    fake_st.session_state[explain_module._CACHE_KEY] = {
        cache_key: {
            "text": "Legacy error output",
            "payload_fingerprint": payload_fingerprint,
            "trace_url": None,
            "payload": {"error": ""},
            "created_at": "2026-10-18T00:00:00+00:00",
            "inputs_summary": {},
        }
    }

    explain_module.render_explain_results_panel(
        summary_df=summary_df,
        manifest=manifest,
        xlsx_path="/tmp/Outputs.xlsx",
    )

    assert calls == 1
    assert fake_st.downloads
    assert not any(
        kind == "markdown" and text == "Legacy error output" for kind, text in fake_st.messages
    )
    cached = fake_st.session_state[explain_module._CACHE_KEY][cache_key]
    assert cached["text"] == "Recovered legacy-cache explanation"
    assert cached["payload"] == {"status": "success"}


@pytest.mark.parametrize("destination", ["anthropic", "azure_openai"])
def test_provider_defaults_switches_and_secret_free_exports(
    monkeypatch, socket_connect_guard, destination
):
    import pa_core.llm.result_explain as core
    from dashboard.components import llm_settings

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
    fake.session_state[explain_module._API_KEY_KEY] = "synthetic-legacy-openai"
    monkeypatch.setattr(explain_module, "st", fake)
    captured = []

    def factory(config):
        captured.append(config)
        return SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content="Safe explanation"))

    monkeypatch.setattr(core, "create_llm", factory)
    monkeypatch.setattr(core, "_is_tracing_enabled", lambda enabled: False)
    monkeypatch.setattr(core, "record_fleet_event", lambda *args, **kwargs: None)

    def render():
        fake.session_state[explain_module._CACHE_KEY] = {}
        explain_module.render_explain_results_panel(
            summary_df=pd.DataFrame({"monthly_TE": [0.02]}),
            manifest={"seed": 1},
            xlsx_path="/tmp/results.xlsx",
        )

    render()
    assert fake.session_state[explain_module._PROVIDER_KEY] == "anthropic"
    assert captured[-1].credentials["api_key"] == "synthetic-anthropic"
    assert explain_module._API_KEY_KEY not in fake.session_state
    fake.session_state[explain_module._PROVIDER_KEY] = "openai"
    fake.session_state[f"{explain_module._API_KEY_KEY}::openai"] = "synthetic-typed-openai"
    render()
    assert captured[-1].credentials["api_key"] == "synthetic-typed-openai"
    fake.session_state[explain_module._PROVIDER_KEY] = destination
    if destination == "azure_openai":
        # Azure has no automatic provider key in this dashboard's contract.
        render()
        assert len(captured) == 2
        fake.session_state[f"{explain_module._API_KEY_KEY}::azure_openai"] = "synthetic-azure"
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


def test_tracing_only_never_constructs_explanation_client(monkeypatch, socket_connect_guard):
    import pa_core.llm.result_explain as core
    from dashboard.components import llm_settings

    monkeypatch.setattr(llm_settings, "read_secret", {"LANGSMITH_API_KEY": "synthetic-tracing"}.get)
    fake = FakeStreamlit(button_value=True)
    monkeypatch.setattr(explain_module, "st", fake)
    calls = []
    monkeypatch.setattr(core, "create_llm", lambda config: calls.append(config))
    explain_module.render_explain_results_panel(
        summary_df=pd.DataFrame({"monthly_TE": [0.02]}), manifest={}, xlsx_path="/tmp/results.xlsx"
    )
    assert calls == []
    assert any("No API key found" in message for _, message in fake.messages)
    assert fake.downloads == []
    assert socket_connect_guard[0] == []


@pytest.mark.parametrize(
    "changed_input",
    [
        "summary",
        "config_hash",
        "data_files",
        "run_name",
        "seed",
        "cli_n_sims",
        "cli_horizon_months",
        "cli_output",
        "cli_benchmark",
        "cli_config",
    ],
)
def test_explanation_cache_tracks_current_run_and_downloads(monkeypatch, changed_input):
    fake = FakeStreamlit(button_value=True)
    monkeypatch.setattr(explain_module, "st", fake)
    monkeypatch.setattr(explain_module, "default_api_key", lambda provider: "test-key")
    monkeypatch.setattr(explain_module, "resolve_api_key_input", lambda raw: raw)
    monkeypatch.setattr(
        explain_module,
        "resolve_llm_provider_config",
        lambda **kwargs: SimpleNamespace(provider_name="openai", model_name="test-model"),
    )
    calls = []

    def explain(summary, manifest, **kwargs):
        text = f"Explanation {len(calls) + 1}: TE={summary.iloc[0]['monthly_TE']}"
        calls.append((summary.copy(), dict(manifest)))
        return text, None, {"rows": len(summary)}

    monkeypatch.setattr(explain_module, "explain_results_details", explain)
    summary = pd.DataFrame({"monthly_TE": [0.01]})
    manifest = {
        "seed": 123,
        "run_name": "run-a",
        "config_hash": "config-a",
        "data_files": {"returns.csv": "data-a"},
        "cli_args": {
            "n_sims": 100,
            "horizon_months": 12,
            "output": "Outputs.xlsx",
            "benchmark": "benchmark.csv",
            "config": "config.yaml",
        },
    }

    def render():
        explain_module.render_explain_results_panel(
            summary_df=summary, manifest=manifest, xlsx_path="/tmp/Outputs.xlsx"
        )

    render()
    first_payload = json.loads(next(d.data for d in fake.downloads if d.mime == "application/json"))
    render()
    assert len(calls) == 1
    if changed_input == "summary":
        summary.loc[0, "monthly_TE"] = 0.09
    elif changed_input == "config_hash":
        manifest["config_hash"] = "config-b"
    elif changed_input == "data_files":
        manifest["data_files"] = {"returns.csv": "data-b"}
    elif changed_input == "run_name":
        manifest["run_name"] = "run-b"
    elif changed_input == "seed":
        manifest["seed"] = 456
    else:
        key = changed_input.removeprefix("cli_")
        old_value = manifest["cli_args"][key]
        new_value = old_value + 1 if isinstance(old_value, int) else old_value + ".changed"
        manifest["cli_args"] = {**manifest["cli_args"], key: new_value}

    fake.button_value = False
    fake.messages.clear()
    fake.downloads.clear()
    render()
    assert not any(kind == "markdown" for kind, _ in fake.messages)
    assert not fake.downloads
    assert len(calls) == 1

    fake.button_value = True
    render()
    assert len(calls) == 2
    current_text = next(text for kind, text in fake.messages if kind == "markdown")
    assert current_text.startswith("Explanation 2:")
    payload = json.loads(next(d.data for d in fake.downloads if d.mime == "application/json"))
    fingerprint = payload["payload_fingerprint"]
    assert fingerprint != first_payload["payload_fingerprint"]
    assert payload["inputs_summary"]["payload_fingerprint"] == fingerprint
    assert payload["text"] == current_text
    txt = next(d.data for d in fake.downloads if d.mime == "text/plain")
    assert fingerprint in txt and current_text in txt
    bucket = fake.session_state[explain_module._CACHE_KEY]
    assert any(entry["payload_fingerprint"] == fingerprint for entry in bucket.values())
    render()
    assert len(calls) == 2


def test_payload_fingerprint_is_stable_for_equivalent_manifest_order():
    summary = pd.DataFrame({"monthly_TE": [0.01], "monthly_CVaR": [float("nan")]})
    first = {"config_hash": "config-a", "data_files": {"a.csv": "a", "b.csv": "b"}, "seed": 1}
    reordered = {"seed": 1, "data_files": {"b.csv": "b", "a.csv": "a"}, "config_hash": "config-a"}
    assert explain_module._payload_fingerprint(
        summary, first
    ) == explain_module._payload_fingerprint(summary.copy(), reordered)
