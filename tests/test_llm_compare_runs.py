"""Tests for pa_core.llm.compare_runs helpers."""

from __future__ import annotations

import json
import sys
import types

import pandas as pd
import pytest

from pa_core.llm.compare_runs import (
    _invoke_comparison_llm,
    build_metric_catalog,
    compare_runs,
    format_config_diff,
    load_prior_manifest,
    load_prior_summary,
)
from pa_core.llm.provider import LLMProviderConfig


def test_load_prior_manifest_reads_previous_run_file(tmp_path):
    prev_manifest = {"seed": 99, "cli_args": {"output": "old.xlsx"}}
    prev_manifest_path = tmp_path / "previous-manifest.json"
    prev_manifest_path.write_text(json.dumps(prev_manifest))

    loaded, path = load_prior_manifest({"previous_run": str(prev_manifest_path)})

    assert loaded == prev_manifest
    assert path == prev_manifest_path


def test_load_prior_manifest_returns_none_when_previous_run_missing():
    loaded, path = load_prior_manifest({"seed": 7})

    assert loaded is None
    assert path is None


def test_load_prior_manifest_returns_path_when_file_missing(tmp_path):
    missing_path = tmp_path / "does-not-exist.json"

    loaded, path = load_prior_manifest({"previous_run": str(missing_path)})

    assert loaded is None
    assert path == missing_path


def test_load_prior_summary_reads_previous_output(tmp_path):
    prior_output = tmp_path / "prior.xlsx"
    pd.DataFrame({"monthly_TE": [0.02], "monthly_CVaR": [-0.03]}).to_excel(
        prior_output, sheet_name="Summary", index=False
    )
    prior_manifest = {"cli_args": {"output": str(prior_output)}}

    loaded, path = load_prior_summary(prior_manifest)

    assert isinstance(loaded, pd.DataFrame)
    assert path == prior_output
    assert "monthly_TE" in loaded.columns


def test_load_prior_summary_resolves_output_relative_to_manifest(tmp_path):
    run_dir = tmp_path / "prior-run"
    run_dir.mkdir()
    prior_output = run_dir / "prior.xlsx"
    pd.DataFrame({"monthly_TE": [0.02], "monthly_CVaR": [-0.03]}).to_excel(
        prior_output, sheet_name="Summary", index=False
    )
    prior_manifest_path = run_dir / "manifest.json"
    prior_manifest_path.write_text(json.dumps({"cli_args": {"output": "prior.xlsx"}}))
    prior_manifest = {"cli_args": {"output": "prior.xlsx"}}

    loaded, path = load_prior_summary(prior_manifest, manifest_path=prior_manifest_path)

    assert isinstance(loaded, pd.DataFrame)
    assert path == prior_output
    assert loaded["monthly_TE"].iloc[0] == 0.02


def test_format_config_diff_includes_seed_cli_and_wizard_changes():
    current_manifest = {
        "seed": 11,
        "cli_args": {"capital": 1000000, "distribution": "gaussian"},
        "wizard_config": {"risk": {"cvar_limit": 0.07}},
    }
    prior_manifest = {
        "seed": 7,
        "cli_args": {"capital": 750000, "distribution": "student_t"},
        "wizard_config": {"risk": {"cvar_limit": 0.09}},
    }

    text = format_config_diff(current_manifest, prior_manifest)

    assert "seed" in text
    assert "cli_args.capital" in text
    assert "wizard_config.risk.cvar_limit" in text


def test_format_config_diff_reports_no_differences_when_manifests_match():
    manifest = {
        "seed": 7,
        "cli_args": {"capital": 750000, "distribution": "student_t"},
        "wizard_config": {"risk": {"cvar_limit": 0.09}},
    }

    text = format_config_diff(manifest, manifest)

    assert text == "No config differences detected."


def test_format_config_diff_handles_missing_manifest_inputs():
    text = format_config_diff(None, {"seed": 1})

    assert text == "No config differences available."


def test_format_config_diff_includes_wizard_add_and_remove_paths():
    current_manifest = {"wizard_inputs": {"alpha": {"limit": 0.2}, "beta": {"enabled": True}}}
    prior_manifest = {"wizard_inputs": {"alpha": {"limit": 0.1}, "gamma": {"enabled": False}}}

    text = format_config_diff(current_manifest, prior_manifest)

    assert "wizard_inputs.alpha.limit" in text
    assert "wizard_inputs.beta.enabled" in text
    assert "wizard_inputs.gamma.enabled" in text


def test_build_metric_catalog_extracts_supported_metrics():
    df = pd.DataFrame(
        {
            "monthly_TE": [0.02, 0.03],
            "TrackingErr": [0.01, 0.02],
            "monthly_CVaR": [-0.04, -0.06],
            "monthly_BreachProb": [0.1, 0.2],
            "other": ["a", "b"],
        }
    )

    catalog = build_metric_catalog(df)

    assert catalog["monthly_TE"] == 0.025
    assert catalog["TrackingErr"] == 0.015
    assert catalog["monthly_CVaR"] == -0.05
    assert catalog["monthly_BreachProb"] == pytest.approx(0.15)


def test_compare_runs_returns_text_trace_and_payload(monkeypatch, tmp_path):
    current_summary = pd.DataFrame({"monthly_TE": [0.03], "monthly_CVaR": [-0.05]})
    prior_output = tmp_path / "prior.xlsx"
    pd.DataFrame({"monthly_TE": [0.01], "monthly_CVaR": [-0.04]}).to_excel(
        prior_output, sheet_name="Summary", index=False
    )
    prior_manifest_path = tmp_path / "prior_manifest.json"
    prior_manifest_path.write_text(
        json.dumps({"cli_args": {"output": str(prior_output)}, "seed": 1})
    )

    monkeypatch.setenv("LANGSMITH_API_KEY", "test-langsmith-key")

    text, trace_url, payload = compare_runs(
        current_summary=current_summary,
        current_manifest={"previous_run": str(prior_manifest_path), "seed": 2},
        questions="What changed?",
        provider_name="openai",
        model_name="gpt-4o-mini",
    )

    assert isinstance(text, str)
    assert "comparison" in text.lower() or len(text) > 0
    assert trace_url is not None
    assert payload.config_diff
    assert payload.prior_manifest_path == str(prior_manifest_path)


def test_compare_runs_prefers_llm_text_when_available(monkeypatch, tmp_path):
    current_summary = pd.DataFrame({"monthly_TE": [0.03], "monthly_CVaR": [-0.05]})
    prior_output = tmp_path / "prior.xlsx"
    pd.DataFrame({"monthly_TE": [0.01], "monthly_CVaR": [-0.04]}).to_excel(
        prior_output, sheet_name="Summary", index=False
    )
    prior_manifest_path = tmp_path / "prior_manifest.json"
    prior_manifest_path.write_text(
        json.dumps({"cli_args": {"output": str(prior_output)}, "seed": 1})
    )

    monkeypatch.setenv("LANGSMITH_API_KEY", "test-langsmith-key")
    monkeypatch.setattr(
        "pa_core.llm.compare_runs._invoke_comparison_llm",
        lambda *args, **kwargs: "LLM generated run comparison explanation.",
    )

    text, trace_url, payload = compare_runs(
        current_summary=current_summary,
        current_manifest={"previous_run": str(prior_manifest_path), "seed": 2},
        questions="What changed?",
        provider_name="openai",
        model_name="gpt-4o-mini",
        api_key="test-openai-key",
    )

    assert text == "LLM generated run comparison explanation."
    assert trace_url is not None
    assert trace_url.startswith("https://smith.langchain.com/r/")
    assert payload.prior_manifest_path == str(prior_manifest_path)


def test_compare_runs_threads_provider_config_to_llm(monkeypatch, tmp_path) -> None:
    current_summary = pd.DataFrame({"monthly_TE": [0.03], "monthly_CVaR": [-0.05]})
    prior_output = tmp_path / "prior.xlsx"
    pd.DataFrame({"monthly_TE": [0.01], "monthly_CVaR": [-0.04]}).to_excel(
        prior_output, sheet_name="Summary", index=False
    )
    prior_manifest_path = tmp_path / "prior_manifest.json"
    prior_manifest_path.write_text(
        json.dumps({"cli_args": {"output": str(prior_output)}, "seed": 1})
    )

    captured: dict[str, LLMProviderConfig] = {}

    def fake_create_llm(config: LLMProviderConfig):
        captured["config"] = config
        return object()

    _install_fake_chat_prompt(monkeypatch)
    monkeypatch.setattr("pa_core.llm.provider.create_llm", fake_create_llm)

    provider_config = LLMProviderConfig(
        provider_name="anthropic",
        credentials={"api_key": "sk-ant-test"},
        model_name="claude-test",
    )

    text, _, payload = compare_runs(
        current_summary=current_summary,
        current_manifest={"previous_run": str(prior_manifest_path), "seed": 2},
        questions="What changed?",
        provider_name="openai",
        provider_config=provider_config,
    )

    assert text == "provider response"
    assert captured["config"] == provider_config
    assert payload.prior_manifest_path == str(prior_manifest_path)


def _install_fake_chat_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResponse:
        content = "provider response"

    class _FakeChain:
        def invoke(self, payload):
            assert "prompt" in payload
            return _FakeResponse()

    class _FakePrompt:
        def __or__(self, llm):
            return _FakeChain()

    class _FakeChatPromptTemplate:
        @staticmethod
        def from_messages(messages):
            assert messages == [("system", "{prompt}")]
            return _FakePrompt()

    fake_prompts = types.ModuleType("langchain_core.prompts")
    fake_prompts.ChatPromptTemplate = _FakeChatPromptTemplate
    fake_langchain_core = types.ModuleType("langchain_core")
    fake_langchain_core.prompts = fake_prompts
    monkeypatch.setitem(sys.modules, "langchain_core", fake_langchain_core)
    monkeypatch.setitem(sys.modules, "langchain_core.prompts", fake_prompts)


def test_invoke_comparison_llm_threads_anthropic_provider_config(monkeypatch) -> None:
    captured: dict[str, LLMProviderConfig] = {}

    def fake_create_llm(config: LLMProviderConfig):
        captured["config"] = config
        return object()

    _install_fake_chat_prompt(monkeypatch)
    monkeypatch.setattr("pa_core.llm.provider.create_llm", fake_create_llm)

    text = _invoke_comparison_llm(
        "Compare the runs.",
        provider_name="anthropic",
        model_name="claude-test",
        api_key="sk-ant-test",
    )

    assert text == "provider response"
    assert captured["config"].provider_name == "anthropic"
    assert captured["config"].model_name == "claude-test"
    assert captured["config"].credentials["api_key"] == "sk-ant-test"


def test_compare_runs_with_anthropic_provider_name_and_no_prebuilt_config(
    monkeypatch, tmp_path
) -> None:
    """End-to-end through ``compare_runs`` with ``provider_name="anthropic"`` and
    no pre-built ``provider_config`` — i.e. the path a dashboard user takes when
    they pick "anthropic" from the provider dropdown rather than supplying a
    prepared LLMProviderConfig.

    Follow-up to #1786 / PR #1787. The existing test
    ``test_compare_runs_threads_provider_config_to_llm`` exercises the
    short-circuit where a caller hands compare_runs a pre-built provider_config
    (provider_name="openai" but provider_config=<anthropic>); that proves the
    config-pass-through path but does NOT exercise the
    provider_name="anthropic" + provider_credentials code path the dashboard
    actually uses. The existing
    ``test_invoke_comparison_llm_threads_anthropic_provider_config`` exercises
    the lower-level helper directly, again bypassing compare_runs.

    This test closes the gap: ``compare_runs`` is called with
    ``provider_name="anthropic"`` (no pre-built provider_config) and asserts
    the LLMProviderConfig actually constructed and passed to ``create_llm``
    has provider_name=="anthropic", the model_name we requested, and the
    api_key+credentials we provided.
    """
    current_summary = pd.DataFrame({"monthly_TE": [0.03], "monthly_CVaR": [-0.05]})
    prior_output = tmp_path / "prior.xlsx"
    pd.DataFrame({"monthly_TE": [0.01], "monthly_CVaR": [-0.04]}).to_excel(
        prior_output, sheet_name="Summary", index=False
    )
    prior_manifest_path = tmp_path / "prior_manifest.json"
    prior_manifest_path.write_text(
        json.dumps({"cli_args": {"output": str(prior_output)}, "seed": 1})
    )

    captured: dict[str, LLMProviderConfig] = {}

    def fake_create_llm(config: LLMProviderConfig):
        captured["config"] = config
        return object()

    _install_fake_chat_prompt(monkeypatch)
    monkeypatch.setattr("pa_core.llm.provider.create_llm", fake_create_llm)
    # The default-key fallback reads CLAUDE_API_STRANSKE for anthropic; clear
    # it so this test exercises ONLY the explicitly-passed api_key path and
    # doesn't depend on the host environment.
    monkeypatch.delenv("CLAUDE_API_STRANSKE", raising=False)

    text, _trace_url, payload = compare_runs(
        current_summary=current_summary,
        current_manifest={"previous_run": str(prior_manifest_path), "seed": 2},
        questions="What changed?",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        api_key="sk-ant-test",
        provider_credentials={"anthropic_version": "2026-01-01"},
    )

    assert text == "provider response"
    assert payload.prior_manifest_path == str(prior_manifest_path)

    # The end-to-end path constructed a fresh LLMProviderConfig from the
    # provider_name + model_name + api_key inputs (not from a pre-supplied
    # provider_config). This is the regression the test exists to guard
    # against: if a future refactor accidentally requires provider_config to
    # be pre-built, dashboard flows would silently fall through to the
    # heuristic-text path.
    cfg = captured["config"]
    assert cfg.provider_name == "anthropic"
    assert cfg.model_name == "claude-sonnet-4-6"
    assert cfg.credentials["api_key"] == "sk-ant-test"
    # Extra credentials (e.g. anthropic_version) thread through too.
    assert cfg.credentials["anthropic_version"] == "2026-01-01"


def test_invoke_comparison_llm_threads_azure_provider_credentials(monkeypatch) -> None:
    captured: dict[str, LLMProviderConfig] = {}

    def fake_create_llm(config: LLMProviderConfig):
        captured["config"] = config
        return object()

    _install_fake_chat_prompt(monkeypatch)
    monkeypatch.setattr("pa_core.llm.provider.create_llm", fake_create_llm)

    text = _invoke_comparison_llm(
        "Compare the runs.",
        provider_name="azure_openai",
        model_name="gpt-4o-mini",
        api_key="sk-azure-test",
        provider_credentials={
            "azure_endpoint": "https://example.openai.azure.com",
            "api_version": "2025-01-01-preview",
        },
    )

    assert text == "provider response"
    assert captured["config"].provider_name == "azure_openai"
    assert captured["config"].credentials["api_key"] == "sk-azure-test"
    assert captured["config"].credentials["azure_endpoint"] == "https://example.openai.azure.com"
    assert captured["config"].credentials["api_version"] == "2025-01-01-preview"


@pytest.mark.parametrize(
    "credential_source", ["explicit", "credentials", "config", "environment", "bearer"]
)
def test_comparison_fallback_redacts_credentials_and_chained_errors(
    monkeypatch, caplog, credential_source
):
    import importlib
    from contextlib import nullcontext

    module = importlib.import_module("pa_core.llm.compare_runs")
    marker = "synthetic-secret-2363"
    bearer = "header+secret/2363~=="
    options = {"provider_name": "openai"}
    if credential_source == "explicit":
        options["api_key"] = marker
    elif credential_source == "credentials":
        options["provider_credentials"] = {"api_key": marker}
    elif credential_source == "config":
        options["provider_config"] = LLMProviderConfig(
            provider_name="openai", credentials={"api_key": marker}, model_name=None
        )
    elif credential_source == "environment":
        monkeypatch.setenv("OPENAI_API_KEY", marker)
    else:
        marker = bearer
    monkeypatch.setattr(module, "load_prior_manifest", lambda _: ({}, None))
    monkeypatch.setattr(
        module, "load_prior_summary", lambda *a, **k: (pd.DataFrame({"monthly_TE": [0.01]}), None)
    )
    monkeypatch.setattr(module, "langsmith_tracing_context", lambda **k: nullcontext())
    monkeypatch.setattr(module, "record_fleet_event", lambda *a, **k: None)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    def fail(*args, **kwargs):
        try:
            raise ValueError("cause contains synthetic-cause-secret-2363")
        except ValueError as cause:
            message = f"provider rejected {marker}; Bearer {bearer}"
            if credential_source == "bearer":
                message = f"provider rejected Bearer {bearer}"
            raise RuntimeError(message) from cause

    monkeypatch.setattr(module, "_invoke_comparison_llm", fail)
    with caplog.at_level("WARNING", logger=module.__name__):
        text, trace, payload = module.compare_runs(
            current_summary=pd.DataFrame({"monthly_TE": [0.02]}),
            current_manifest={},
            questions="What changed?",
            **options,
        )
    assert payload.status == "fallback"
    assert payload.error_category == "RuntimeError"
    assert trace is None
    assert text.startswith("Run-to-run comparison summary:")
    assert marker not in caplog.text and bearer not in caplog.text
    assert "synthetic-cause-secret-2363" not in caplog.text
    assert "[REDACTED]" in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def _question_comparison_input(tmp_path):
    prior_output = tmp_path / "question-prior.xlsx"
    pd.DataFrame({"monthly_TE": [0.01], "monthly_CVaR": [-0.04]}).to_excel(
        prior_output, sheet_name="Summary", index=False
    )
    prior_manifest = tmp_path / "question-prior.json"
    prior_manifest.write_text(json.dumps({"cli_args": {"output": str(prior_output)}, "seed": 1}))
    return {"previous_run": str(prior_manifest), "seed": 2}


@pytest.mark.parametrize("question", ["Why did tracking error rise?", "", " \t\n"])
def test_comparison_question_reaches_provider_prompt(monkeypatch, tmp_path, question):
    import importlib
    from contextlib import nullcontext

    from pa_core.llm.compare_runs import DEFAULT_COMPARISON_QUESTIONS

    module = importlib.import_module("pa_core.llm.compare_runs")
    captured = []
    monkeypatch.setattr(
        module,
        "_invoke_comparison_llm",
        lambda prompt, **kwargs: captured.append(prompt) or "Provider comparison",
    )
    monkeypatch.setattr(module, "langsmith_tracing_context", lambda **kwargs: nullcontext())
    monkeypatch.setattr(module, "record_fleet_event", lambda *args, **kwargs: None)
    text, _, payload = compare_runs(
        current_summary=pd.DataFrame({"monthly_TE": [0.03], "monthly_CVaR": [-0.05]}),
        current_manifest=_question_comparison_input(tmp_path),
        questions=question,
    )
    expected = question.strip() or DEFAULT_COMPARISON_QUESTIONS
    assert text == "Provider comparison"
    assert captured == [payload.prompt]
    assert expected in captured[0] and payload.questions == expected
    assert '"label": "current"' in captured[0] and '"label": "prior"' in captured[0]
    assert '"monthly_TE": 0.03' in captured[0] and '"monthly_TE": 0.01' in captured[0]


def test_dashboard_question_changes_actual_provider_prompt(
    monkeypatch, tmp_path, socket_connect_guard
):
    import importlib
    from contextlib import nullcontext

    from test_dashboard_comparison_llm import FakeStreamlit

    import dashboard.components.comparison_llm as panel

    module = importlib.import_module("pa_core.llm.compare_runs")
    fake = FakeStreamlit(button_value=True)
    monkeypatch.setattr(panel, "st", fake)
    monkeypatch.setattr(panel, "default_api_key", lambda provider: "test-key")
    monkeypatch.setattr(panel, "resolve_api_key_input", lambda raw: raw)
    monkeypatch.setattr(
        panel,
        "resolve_llm_provider_config",
        lambda **kwargs: LLMProviderConfig(
            provider_name="openai", credentials={"api_key": "test-key"}
        ),
    )
    monkeypatch.setattr(module, "langsmith_tracing_context", lambda **kwargs: nullcontext())
    monkeypatch.setattr(module, "record_fleet_event", lambda *args, **kwargs: None)
    captured = []
    monkeypatch.setattr(
        module,
        "_invoke_comparison_llm",
        lambda prompt, **kwargs: captured.append(prompt) or "Provider comparison",
    )
    manifest = _question_comparison_input(tmp_path)
    summary = pd.DataFrame({"monthly_TE": [0.03], "monthly_CVaR": [-0.05]})
    questions = ["Explain tracking error changes", "Explain downside risk changes"]
    payloads = []
    for question in questions:
        fake.session_state[f"{panel._QUESTIONS_KEY}::questions"] = question
        panel.render_comparison_llm_panel(
            summary_df=summary,
            manifest_data=manifest,
            run_key="questions",
        )
        payloads.append(fake.session_state["comparison_payload::questions"])

    assert len(captured) == 2
    assert [payload.questions for payload in payloads] == questions
    assert [payload.prompt for payload in payloads] == captured
    prompt_data = []
    for prompt, question in zip(captured, questions, strict=True):
        data, separator, question_block = prompt.partition("\n\nQuestions:\n")
        assert separator
        assert question_block == f"- {question}"
        prompt_data.append(data)
        current_data, prior_data = data.split("\n\nItem B:\n", maxsplit=1)
        assert '"label": "current"' in current_data
        assert '"monthly_TE": 0.03' in current_data
        assert '"monthly_CVaR": -0.05' in current_data
        assert '"label": "prior"' in prior_data
        assert '"monthly_TE": 0.01' in prior_data
        assert '"monthly_CVaR": -0.04' in prior_data
    assert prompt_data[0] == prompt_data[1]
    assert captured[0] != captured[1]

    exports = [json.loads(item.data) for item in fake.downloads if item.mime == "application/json"]
    assert len(exports) == 2
    for export, payload, question in zip(exports, payloads, questions, strict=True):
        assert export["questions"] == question
        assert export["metric_catalog_a"] == payload.metric_catalog_a
        assert export["metric_catalog_b"] == payload.metric_catalog_b
    assert socket_connect_guard[0] == []


def test_empty_question_fallback_uses_same_default(monkeypatch, tmp_path):
    import importlib
    from contextlib import nullcontext

    from pa_core.llm.compare_runs import DEFAULT_COMPARISON_QUESTIONS

    module = importlib.import_module("pa_core.llm.compare_runs")

    def fail(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(module, "_invoke_comparison_llm", fail)
    monkeypatch.setattr(module, "langsmith_tracing_context", lambda **kwargs: nullcontext())
    monkeypatch.setattr(module, "record_fleet_event", lambda *args, **kwargs: None)
    text, _, payload = compare_runs(
        current_summary=pd.DataFrame({"monthly_TE": [0.03]}),
        current_manifest=_question_comparison_input(tmp_path),
        questions=" ",
    )
    assert text.startswith("Run-to-run comparison summary:")
    assert f"Questions: {DEFAULT_COMPARISON_QUESTIONS}" in text
    assert "Config differences:" in text and "Metric deltas:" in text
    assert payload.questions == DEFAULT_COMPARISON_QUESTIONS
