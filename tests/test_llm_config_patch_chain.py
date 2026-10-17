"""Tests for pa_core.llm.config_patch_chain helpers."""

from __future__ import annotations

from contextlib import nullcontext

import pytest

from pa_core.llm.config_patch import ConfigPatchValidationError
from pa_core.llm.config_patch_chain import (
    _build_validation_error,
    _extract_trace_candidate,
    _split_trace_metadata,
    build_config_patch_prompt,
    parse_chain_output,
    run_config_patch_chain,
)


def test_build_prompt_validates_inputs_and_includes_custom_rules() -> None:
    with pytest.raises(TypeError, match="current_config"):
        build_config_patch_prompt(current_config=[], instruction="adjust")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="instruction"):
        build_config_patch_prompt(current_config={}, instruction="  ")

    prompt = build_config_patch_prompt(
        current_config={"n_simulations": 100}, instruction="  adjust  ", safety_rules=["No writes"]
    )

    assert '"n_simulations": 100' in prompt
    assert '[\n  "No writes"\n]' in prompt
    assert prompt.endswith("Instruction:\nadjust")


@pytest.mark.parametrize(
    ("raw_output", "expected_flags"),
    [
        ('{"patch": {"set": {"n_simulations": 5}}}', []),
        ("patch:\n  set:\n    n_simulations: 5", []),
        ({"patch": {}, "risk_flags": None}, []),
        ({"patch": {}, "risk_flags": "review"}, ["review"]),
        ({"patch": {}, "risk_flags": [" review ", "", "review", 3]}, ["review", "3"]),
    ],
)
def test_parse_chain_output_accepts_json_yaml_and_normalizes_risk_flags(
    raw_output, expected_flags
) -> None:
    assert parse_chain_output(raw_output).risk_flags == expected_flags


@pytest.mark.parametrize("raw_output", ["", "[]", 3, ("not", "a supported tuple")])
def test_parse_chain_output_rejects_non_mapping_payloads(raw_output) -> None:
    with pytest.raises(ValueError, match="mapping"):
        parse_chain_output(raw_output)  # type: ignore[arg-type]


def test_validation_error_includes_optional_type_metadata() -> None:
    error = ConfigPatchValidationError(
        "bad type", field_name="n_simulations", expected_type="integer", actual_type="string"
    )

    assert _build_validation_error(error) == {
        "kind": "validation_error",
        "message": "bad type",
        "unknown_keys": [],
        "unknown_paths": [],
        "field_name": "n_simulations",
        "expected_type": "integer",
        "actual_type": "string",
    }


def test_split_trace_metadata_handles_tuple_and_nested_candidates(monkeypatch) -> None:
    monkeypatch.setattr(
        "pa_core.llm.config_patch_chain._resolve_trace_url",
        lambda candidate: f"resolved:{candidate}" if candidate == "run-object" else None,
    )

    payload, trace_url = _split_trace_metadata(
        ({"patch": {}, "trace": {"nested": [None, "run-object"]}}, "fallback")
    )

    assert payload == {"patch": {}}
    assert trace_url == "resolved:run-object"
    assert _extract_trace_candidate(None) is None
    assert _extract_trace_candidate([{}, ["run-object"]]) == "resolved:run-object"


def test_split_trace_metadata_handles_string_tuple_alternate_url_and_missing_resolver(
    monkeypatch,
) -> None:
    # `_resolve_trace_url` honours LANGSMITH_TRACE_BASE_URL: pin it so a shell that sets it
    # cannot change the default URL asserted below.
    monkeypatch.delenv("LANGSMITH_TRACE_BASE_URL", raising=False)
    payload, trace_url = _split_trace_metadata(("payload", "trace-url"))
    assert payload == "payload"
    assert trace_url == "https://smith.langchain.com/r/trace-url"

    payload, trace_url = _split_trace_metadata(
        {"patch": {}, "langsmith_trace_url": " https://trace "}
    )
    assert payload == {"patch": {}}
    assert trace_url == "https://trace"

    assert _split_trace_metadata(3) == (3, None)
    monkeypatch.setattr("pa_core.llm.config_patch_chain._resolve_trace_url", None)
    assert _extract_trace_candidate({"first": "unresolved", "second": ["also-unresolved"]}) is None


def test_trace_candidate_skips_unresolved_metadata_before_using_next_match(monkeypatch) -> None:
    monkeypatch.setattr(
        "pa_core.llm.config_patch_chain._resolve_trace_url",
        lambda candidate: "resolved" if candidate == "good" else None,
    )

    assert _extract_trace_candidate({"trace_id": None, "run_id": "good"}) == "resolved"


def test_run_chain_records_rejections_errors_and_recording_failures(monkeypatch, caplog) -> None:
    recorded = []
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setattr(
        "pa_core.llm.config_patch_chain._langsmith_tracing_context", lambda **_: nullcontext()
    )
    monkeypatch.setattr("pa_core.llm.config_patch_chain.record_fleet_event", recorded.append)

    rejected = run_config_patch_chain(
        current_config={"n_simulations": 5},
        instruction="adjust",
        provider_name="provider",
        model_name="model",
        invoke_llm=lambda _: {"patch": {"set": {"unknown": 1}}},
    )
    assert rejected.status == "rejected"
    assert recorded[-1].status == "validation_failed"
    assert recorded[-1].error_category == "validation_error"

    def fail_record(_event) -> None:
        raise RuntimeError("telemetry unavailable")

    monkeypatch.setattr("pa_core.llm.config_patch_chain.record_fleet_event", fail_record)
    with pytest.raises(RuntimeError, match="provider failed"):
        run_config_patch_chain(
            current_config={},
            instruction="adjust",
            invoke_llm=lambda _: (_ for _ in ()).throw(RuntimeError("provider failed")),
        )
    assert "Failed to record config-patch-chain fleet event" in caplog.text


def test_run_chain_rejects_non_callable_provider() -> None:
    with pytest.raises(TypeError, match="invoke_llm"):
        run_config_patch_chain(current_config={}, instruction="adjust", invoke_llm=None)  # type: ignore[arg-type]


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


def test_parse_chain_output_rejects_out_of_float_range_numbers_without_crashing() -> None:
    """Untrusted JSON integers must enter the rejected state, not escape as OverflowError."""

    result = parse_chain_output(
        {
            "patch": {"set": {"total_fund_capital": 10**10000}},
            "summary": "Set capital.",
            "risk_flags": [],
        }
    )

    assert result.status == "rejected"
    assert result.patch.set == {}
    assert result.error == {
        "kind": "validation_error",
        "message": "field 'total_fund_capital' must be a finite float",
        "unknown_keys": [],
        "unknown_paths": [],
    }


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
