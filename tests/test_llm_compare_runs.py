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
from pa_core.manifest import ManifestWriter


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
        "config": {"nested": {"theta_extpa": 0.2}},
    }

    text = format_config_diff(manifest, manifest)

    assert text == "No config differences detected."


def test_format_config_diff_compares_manifest_writer_config_snapshots(tmp_path):
    def write_manifest(name: str, snapshot: str) -> dict[str, object]:
        config_path = tmp_path / f"{name}.yaml"
        config_path.write_text(snapshot)
        manifest_path = tmp_path / f"{name}.json"
        ManifestWriter(manifest_path).write(
            config_path=config_path,
            data_files=[],
            seed=17,
            cli_args={"output": f"{name}.xlsx"},
        )
        return json.loads(manifest_path.read_text())

    prior = write_manifest("prior", "theta_extpa: 0.2\n")
    current = write_manifest("current", "theta_extpa: 0.8\n")

    text = format_config_diff(current, prior)

    assert "config.theta_extpa: 0.2 -> 0.8" in text


def test_format_config_diff_sorts_nested_config_changes_and_preserves_missing_vs_null():
    current_manifest = {
        "config": {
            "alpha": {"added": None, "shared": 2},
            "zeta": {"stable": True},
        }
    }
    prior_manifest = {
        "config": {
            "alpha": {"removed": None, "shared": 1},
            "zeta": {"stable": True},
        }
    }

    lines = format_config_diff(current_manifest, prior_manifest).splitlines()

    assert lines == [
        "- config.alpha.added: <missing> -> None",
        "- config.alpha.removed: None -> <missing>",
        "- config.alpha.shared: 1 -> 2",
    ]


def test_format_config_diff_preserves_empty_mapping_additions_and_removals():
    current_manifest = {"config": {"added": {}, "stable": {"value": 1}}}
    prior_manifest = {"config": {"removed": {}, "stable": {"value": 1}}}

    lines = format_config_diff(current_manifest, prior_manifest).splitlines()

    assert lines == [
        "- config.added: <missing> -> {}",
        "- config.removed: {} -> <missing>",
    ]


def test_format_config_diff_keeps_literal_dot_keys_distinct_from_nested_paths():
    current_manifest = {"config": {"a.b": 3, "a": {"b": 4}}}
    prior_manifest = {"config": {"a.b": 1, "a": {"b": 2}}}

    lines = format_config_diff(current_manifest, prior_manifest).splitlines()

    assert lines == [
        "- config.a.b: 2 -> 4",
        '- config["a.b"]: 1 -> 3',
    ]


def test_format_config_diff_empty_root_reports_only_changed_child_paths():
    empty = {"config": {}}
    populated = {"config": {"theta_extpa": 0.8}}

    assert format_config_diff(populated, empty) == "- config.theta_extpa: <missing> -> 0.8"
    assert format_config_diff(empty, populated) == "- config.theta_extpa: 0.8 -> <missing>"


def test_format_config_diff_skips_config_for_legacy_non_mapping_snapshot():
    current_manifest = {"seed": 7, "cli_args": {"capital": 2}, "config": "unavailable"}
    prior_manifest = {"seed": 7, "cli_args": {"capital": 1}, "config": {"theta_extpa": 0.2}}

    text = format_config_diff(current_manifest, prior_manifest)

    assert text == "- cli_args.capital: 1 -> 2"


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

    from contextlib import nullcontext

    monkeypatch.setattr(
        "pa_core.llm.compare_runs.langsmith_tracing_context", lambda **kwargs: nullcontext()
    )
    monkeypatch.setattr(
        "pa_core.llm.compare_runs._invoke_comparison_llm",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline fallback")),
    )

    text, trace_url, payload = compare_runs(
        current_summary=current_summary,
        current_manifest={"previous_run": str(prior_manifest_path), "seed": 2},
        questions="What changed?",
        provider_name="openai",
        model_name="gpt-4o-mini",
    )

    assert isinstance(text, str)
    assert "comparison" in text.lower() or len(text) > 0
    assert trace_url is None
    assert payload.status == "fallback"
    assert payload.config_diff
    assert payload.prior_manifest_path == str(prior_manifest_path)


@pytest.mark.parametrize("trace_failure", [None, "enter", "exit"])
def test_compare_runs_prefers_llm_text_when_available(
    monkeypatch, tmp_path, fake_langsmith_run, trace_failure
):
    fake_langsmith_run.fail_on = trace_failure
    current_summary = pd.DataFrame({"monthly_TE": [0.03], "monthly_CVaR": [-0.05]})
    prior_output = tmp_path / "prior.xlsx"
    pd.DataFrame({"monthly_TE": [0.01], "monthly_CVaR": [-0.04]}).to_excel(
        prior_output, sheet_name="Summary", index=False
    )
    prior_manifest_path = tmp_path / "prior_manifest.json"
    prior_manifest_path.write_text(
        json.dumps({"cli_args": {"output": str(prior_output)}, "seed": 1})
    )

    events = []
    monkeypatch.setattr("pa_core.llm.compare_runs.record_fleet_event", events.append)

    def invoke(*args, **kwargs):
        expected = None if trace_failure == "enter" else fake_langsmith_run.run
        assert fake_langsmith_run.active is expected
        return "LLM generated run comparison explanation."

    monkeypatch.setattr("pa_core.llm.compare_runs._invoke_comparison_llm", invoke)

    text, trace_url, payload = compare_runs(
        current_summary=current_summary,
        current_manifest={"previous_run": str(prior_manifest_path), "seed": 2},
        questions="What changed?",
        provider_name="openai",
        model_name="gpt-4o-mini",
        api_key="test-openai-key",
    )

    assert text == "LLM generated run comparison explanation."
    assert trace_url == (None if trace_failure == "enter" else fake_langsmith_run.run.url)
    assert events[0].trace_id == (None if trace_failure == "enter" else fake_langsmith_run.run.id)
    assert events[0].trace_url == trace_url
    assert fake_langsmith_run.calls[0]["name"] == "portable-alpha-run-comparison"
    assert fake_langsmith_run.active is None
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


def _question_comparison_input(tmp_path, prior_theta=0.2, current_theta=0.8):
    prior_output = tmp_path / "question-prior.xlsx"
    pd.DataFrame({"monthly_TE": [0.01], "monthly_CVaR": [-0.04]}).to_excel(
        prior_output, sheet_name="Summary", index=False
    )
    prior_manifest = tmp_path / "question-prior.json"
    prior_manifest.write_text(
        json.dumps(
            {
                "cli_args": {"output": str(prior_output)},
                "seed": 1,
                "config": {"theta_extpa": prior_theta},
            }
        )
    )
    return {
        "previous_run": str(prior_manifest),
        "seed": 2,
        "config": {"theta_extpa": current_theta},
    }


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


@pytest.mark.parametrize("prior_theta,current_theta", [(0.2, 0.8), (1.0000001, 1.0000002)])
def test_dashboard_question_changes_actual_provider_prompt(
    monkeypatch, tmp_path, socket_connect_guard, prior_theta, current_theta
):
    import importlib
    from contextlib import nullcontext

    from test_dashboard_comparison_llm import FakeStreamlit

    import dashboard.components.comparison_llm as panel
    import pa_core.llm.provider as provider

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

    class FakeProvider:
        def invoke(self, messages):
            assert len(messages) == 1
            assert messages[0].type == "system"
            captured.append(messages[0].content)
            return types.SimpleNamespace(content="Provider comparison")

    class FakeChatPromptTemplate:
        @staticmethod
        def from_messages(messages):
            assert messages == [("system", "{prompt}")]
            return FakeChatPromptTemplate()

        def __or__(self, client):
            class Chain:
                def invoke(self, payload):
                    return client.invoke(
                        [types.SimpleNamespace(type="system", content=payload["prompt"])]
                    )

            return Chain()

    fake_prompts = types.ModuleType("langchain_core.prompts")
    fake_prompts.ChatPromptTemplate = FakeChatPromptTemplate
    monkeypatch.setitem(sys.modules, "langchain_core.prompts", fake_prompts)
    monkeypatch.setattr(provider, "create_llm", lambda config: FakeProvider())
    manifest = _question_comparison_input(tmp_path, prior_theta, current_theta)
    expected_diff = f"config.theta_extpa: {prior_theta} -> {current_theta}"
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
    assert [payload.status for payload in payloads] == ["success", "success"]
    assert [payload.questions for payload in payloads] == questions
    assert [payload.prompt for payload in payloads] == captured
    assert all(expected_diff in payload.config_diff for payload in payloads)
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
        assert expected_diff in prompt
    assert prompt_data[0] == prompt_data[1]
    assert captured[0] != captured[1]

    exports = [json.loads(item.data) for item in fake.downloads if item.mime == "application/json"]
    assert len(exports) == 2
    for export, payload, question in zip(exports, payloads, questions, strict=True):
        assert export["questions"] == question
        assert export["metric_catalog_a"] == payload.metric_catalog_a
        assert export["metric_catalog_b"] == payload.metric_catalog_b
        assert expected_diff in export["config_diff"]
    txt_exports = [item.data for item in fake.downloads if item.mime == "text/plain"]
    assert len(txt_exports) == 2
    assert all(expected_diff in export for export in txt_exports)
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
