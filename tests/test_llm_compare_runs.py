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
