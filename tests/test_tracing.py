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
