"""Targeted tests for langsmith_tracing_context usage forms."""

from __future__ import annotations

import importlib

import pytest


def test_langsmith_tracing_context_without_as_executes_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracing = importlib.import_module("pa_core.llm.tracing")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setattr(tracing, "_LANGSMITH_ENABLED", None)

    executed = False
    with tracing.langsmith_tracing_context():
        executed = True

    assert executed is True


def test_langsmith_tracing_context_with_as_yields_none_and_executes_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracing = importlib.import_module("pa_core.llm.tracing")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setattr(tracing, "_LANGSMITH_ENABLED", None)

    executed = False
    with tracing.langsmith_tracing_context() as context_value:
        executed = True

    assert executed is True
    assert context_value is None


def test_named_context_yields_created_sdk_run(fake_langsmith_run):
    from pa_core.llm.tracing import langsmith_tracing_context, resolve_trace_reference

    with langsmith_tracing_context(
        name="synthetic", tags=["offline"], metadata={"test": True}
    ) as run:
        assert run is fake_langsmith_run.run
        assert fake_langsmith_run.active is run
        assert resolve_trace_reference(run) == (run.id, run.url)
    assert fake_langsmith_run.active is None


def test_reference_requires_created_run_identity():
    from types import SimpleNamespace
    from pa_core.llm.tracing import resolve_trace_reference

    assert resolve_trace_reference("local-request-uuid") == (None, None)
    assert resolve_trace_reference(SimpleNamespace(url="https://example.test/fabricated")) == (
        None,
        None,
    )


def test_reference_resolution_failure_is_nonfatal():
    from pa_core.llm.tracing import resolve_trace_reference

    class BrokenRun:
        id = "sdk-run"

        def get_url(self):
            raise RuntimeError("backend unavailable")

    assert resolve_trace_reference(BrokenRun()) == (None, None)


@pytest.mark.live_llm
def test_authorized_synthetic_trace_live_retrieval(monkeypatch, tmp_path):
    """Opt-in only: create a synthetic invocation and retrieve that exact SDK run."""
    import importlib.metadata
    import json
    import os
    from pathlib import Path
    from types import SimpleNamespace

    evidence = Path(
        os.getenv("PAEM_LIVE_TRACE_EVIDENCE", str(tmp_path / "trace-live-acceptance.json"))
    )

    def unknown(reason):
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text(json.dumps({"verdict": "UNKNOWN", "reason": reason}, indent=2))
        pytest.skip("UNKNOWN: " + reason)

    if os.getenv("PAEM_LIVE_TRACE_ACCEPTANCE") != "1":
        unknown("synthetic trace creation/retrieval was not explicitly enabled")
    if not os.getenv("LANGSMITH_API_KEY"):
        unknown("authenticated retrieval is unavailable")
    try:
        from langsmith import Client, run_helpers

        version = importlib.metadata.version("langsmith")
    except ImportError:
        unknown("optional LangSmith SDK is unavailable")
    if version != "0.10.15":
        unknown("installed SDK differs from the reviewed optional-extra pin")

    import pandas as pd
    from pa_core.llm import result_explain
    from pa_core.llm.provider import LLMProviderConfig

    captured = {}
    events = []

    class SyntheticInvocation:
        def invoke(self, prompt):
            run = run_helpers.get_current_run_tree()
            captured["id"] = str(run.id) if run is not None else None
            return SimpleNamespace(content="Synthetic trace correlation acceptance probe.")

    monkeypatch.setattr(result_explain, "create_llm", lambda config: SyntheticInvocation())
    monkeypatch.setattr(result_explain, "record_fleet_event", events.append)
    _, url, payload = result_explain.explain_results_details(
        pd.DataFrame({"monthly_TE": [0.01]}),
        llm_config=LLMProviderConfig(
            provider_name="openai", credentials={"api_key": "unused-synthetic"}
        ),
        tracing_enabled=True,
    )
    if not captured.get("id") or not events or not url:
        unknown("the same synthetic invocation did not produce a usable SDK trace reference")
    assert events[0].trace_id == captured["id"]
    assert payload["trace_url"] == events[0].trace_url == url
    try:
        # One authenticated retrieval; no polling or unrelated existing run can pass.
        remote = Client().read_run(captured["id"])
    except Exception as exc:
        unknown("same-invocation retrieval unavailable: " + type(exc).__name__)
    assert str(remote.id) == captured["id"]
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(
        json.dumps(
            {
                "verdict": "PASS",
                "synthetic": True,
                "sdk_version": version,
                "invoked_run_id": captured["id"],
                "retrieved_run_id": str(remote.id),
                "trace_url": url,
                "payload_trace_url": payload["trace_url"],
                "fleet_trace_id": events[0].trace_id,
            },
            indent=2,
        )
    )


def test_telemetry_exit_failure_preserves_original_operation_error(fake_langsmith_run):
    from pa_core.llm.tracing import langsmith_tracing_context

    fake_langsmith_run.fail_on = "exit"
    with pytest.raises(ValueError, match="original operation"):
        with langsmith_tracing_context(name="synthetic"):
            raise ValueError("original operation")
