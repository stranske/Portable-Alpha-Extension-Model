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
    assert result.risk_flags_detail["unknown_keys"] == ["hallucinated"]


def test_parse_chain_output_does_not_duplicate_unknown_output_risk_flag() -> None:
    result = parse_chain_output(
        {
            "patch": {"set": {"n_simulations": 5000}},
            "summary": "Increase simulation count.",
            "risk_flags": ["stripped_unknown_output_keys"],
            "hallucinated": {"ignored": True},
        }
    )

    assert result.unknown_output_keys == ["hallucinated"]
    assert result.risk_flags == ["stripped_unknown_output_keys"]


def test_parse_chain_output_rejects_invalid_patch_and_returns_structured_error() -> None:
    result = parse_chain_output(
        {
            "patch": {"set": {"fake_field": 123}},
            "summary": "Try to set unknown field.",
            "risk_flags": [],
        }
    )

    assert result.status == "rejected"
    assert result.error is not None
    assert result.error["unknown_keys"] == ["fake_field"]
    assert result.error["unknown_paths"] == ["patch.set.fake_field"]
    assert result.rejected_patch_keys == ["fake_field"]
    assert result.rejected_patch_paths == ["patch.set.fake_field"]
    assert "rejected_unknown_patch_fields" in result.risk_flags
    assert result.risk_flags_detail["rejected_patch_keys"] == ["fake_field"]


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


def test_run_config_patch_chain_skips_tracing_when_helpers_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setattr("pa_core.llm.config_patch_chain._langsmith_tracing_context", None)
    monkeypatch.setattr("pa_core.llm.config_patch_chain._resolve_trace_url", None)

    result = run_config_patch_chain(
        current_config={"n_simulations": 1000},
        instruction="increase simulations to 5000",
        invoke_llm=lambda _prompt: {
            "patch": {"set": {"n_simulations": 5000}},
            "summary": "Updated n_simulations.",
            "risk_flags": [],
            "trace_url": "https://smith.langchain.com/r/real-run-123",
        },
    )

    assert result.trace_url is None
