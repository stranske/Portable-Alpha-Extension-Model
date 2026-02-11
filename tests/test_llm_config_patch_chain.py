"""Tests for pa_core.llm.config_patch_chain."""

from __future__ import annotations

from pa_core.llm.config_patch_chain import (
    ConfigPatchChain,
    build_config_patch_prompt,
    parse_chain_output,
)


def test_build_config_patch_prompt_contains_schema_and_safety_rules() -> None:
    text = build_config_patch_prompt(
        current_config={"n_simulations": 1000},
        instruction="increase simulations to 5000",
    )

    assert "Allowlisted schema" in text
    assert "Safety rules" in text
    assert "n_simulations" in text


def test_parse_chain_output_strips_unknown_keys_and_fields() -> None:
    result = parse_chain_output(
        {
            "patch": {
                "set": {"n_simulations": 5000, "fake_field": 1},
                "merge": {"risk_metrics": ["Return"]},
                "remove": ["regime_start", "unknown_remove"],
                "replace": {"n_months": 24},
            },
            "summary": "apply requested updates",
            "risk_flags": ["from-model"],
            "unknown_root": "ignored",
        }
    )

    assert result.patch["set"] == {"n_simulations": 5000}
    assert result.patch["merge"] == {"risk_metrics": ["Return"]}
    assert result.patch["remove"] == ["regime_start"]
    assert "from-model" in result.risk_flags
    assert "stripped_unknown_output_key:unknown_root" in result.risk_flags
    assert "stripped_unknown_patch_ops:replace" in result.risk_flags
    assert "stripped_unknown_patch_field:set.fake_field" in result.risk_flags
    assert "stripped_unknown_patch_field:remove.unknown_remove" in result.risk_flags
    assert "unknown_keys_stripped" in result.risk_flags


def test_chain_run_sets_trace_url_when_langsmith_enabled(monkeypatch) -> None:
    class _FixedUUID:
        hex = "trace-id-123"

    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setattr("pa_core.llm.config_patch_chain.uuid4", lambda: _FixedUUID())

    chain = ConfigPatchChain(
        lambda _prompt: {
            "patch": {"set": {"n_simulations": 4000}},
            "summary": "ok",
            "risk_flags": [],
        }
    )
    result = chain.run(current_config={"n_simulations": 1000}, instruction="set to 4000")

    assert result.trace_url == "https://smith.langchain.com/r/trace-id-123"


def test_chain_run_without_langsmith_key_returns_no_trace_url(monkeypatch) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    chain = ConfigPatchChain(
        lambda _prompt: {
            "patch": {"set": {"n_simulations": 3000}},
            "summary": "ok",
            "risk_flags": [],
        }
    )

    result = chain.run(current_config={"n_simulations": 1000}, instruction="set to 3000")

    assert result.trace_url is None
