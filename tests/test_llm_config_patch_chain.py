"""Tests for pa_core.llm.config_patch_chain helpers."""

from __future__ import annotations

from pa_core.llm.config_patch_chain import parse_chain_output, run_config_patch_chain


def test_parse_chain_output_strips_unknown_keys_and_flags_risk() -> None:
    result = parse_chain_output(
        {
            "patch": {"set": {"n_simulations": 5000}},
            "summary": "Increase simulation count.",
            "risk_flags": ["low_risk"],
            "hallucinated": {"ignored": True},
        }
    )

    assert result.patch.set == {"n_simulations": 5000}
    assert result.summary == "Increase simulation count."
    assert result.unknown_output_keys == ["hallucinated"]
    assert "low_risk" in result.risk_flags
    assert "stripped_unknown_output_keys" in result.risk_flags


def test_run_config_patch_chain_includes_trace_url_when_langsmith_enabled(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")

    class _CallbackRun:
        def get_run_url(self) -> str:
            return "https://smith.langchain.com/r/real-run-123"

    result = run_config_patch_chain(
        current_config={"n_simulations": 1000},
        instruction="increase simulations to 5000",
        invoke_llm=lambda _prompt: {
            "patch": {"set": {"n_simulations": 5000}},
            "summary": "Updated n_simulations.",
            "risk_flags": [],
            "callback": _CallbackRun(),
        },
    )

    assert result.trace_url == "https://smith.langchain.com/r/real-run-123"
    assert result.patch.set["n_simulations"] == 5000


def test_run_config_patch_chain_has_no_trace_url_when_langsmith_disabled(monkeypatch) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    result = run_config_patch_chain(
        current_config={"n_simulations": 1000},
        instruction="increase simulations to 5000",
        invoke_llm=lambda _prompt: {
            "patch": {"set": {"n_simulations": 5000}},
            "summary": "Updated n_simulations.",
            "risk_flags": [],
        },
    )

    assert result.trace_url is None


def test_run_config_patch_chain_has_no_trace_url_when_callback_metadata_missing(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")

    result = run_config_patch_chain(
        current_config={"n_simulations": 1000},
        instruction="increase simulations to 5000",
        invoke_llm=lambda _prompt: {
            "patch": {"set": {"n_simulations": 5000}},
            "summary": "Updated n_simulations.",
            "risk_flags": [],
        },
    )

    assert result.trace_url is None
