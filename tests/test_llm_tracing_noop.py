"""Tracing module tests focused on offline-safe behavior."""

from __future__ import annotations

import importlib
import socket
import sys

import pytest


def _resolve_trace_url():
    """Lazy import to avoid loading tracing module before socket guard."""
    from pa_core.llm.tracing import resolve_trace_url

    return resolve_trace_url


def test_import_tracing_no_network(socket_connect_guard):
    attempts, blocked = socket_connect_guard

    sys.modules.pop("pa_core.llm.tracing", None)
    importlib.import_module("pa_core.llm.tracing")

    assert socket.socket.connect is blocked
    assert attempts == []


def test_tracing_context_noop_without_api_key(
    monkeypatch: pytest.MonkeyPatch, socket_connect_guard
):
    attempts, blocked = socket_connect_guard

    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    tracing = importlib.import_module("pa_core.llm.tracing")
    with tracing.langsmith_tracing_context(project_name="unit-test"):
        pass

    assert socket.socket.connect is blocked
    assert attempts == []


def test_maybe_enable_langsmith_tracing_sets_defaults(monkeypatch: pytest.MonkeyPatch):
    tracing = importlib.import_module("pa_core.llm.tracing")

    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.setattr(tracing, "_LANGSMITH_ENABLED", None)

    assert tracing.maybe_enable_langsmith_tracing() is True
    assert tracing.maybe_enable_langsmith_tracing() is True
    assert tracing.os.environ["LANGCHAIN_API_KEY"] == "test-key"
    assert tracing.os.environ["LANGCHAIN_TRACING_V2"] == "true"


def test_resolve_trace_url_accepts_run_object_url_attribute():
    resolve_trace_url = _resolve_trace_url()

    class _Run:
        url = "https://smith.langchain.com/r/from-object"

    assert resolve_trace_url(_Run()) == "https://smith.langchain.com/r/from-object"


# ---------- resolve_trace_url tests ----------


def test_resolve_trace_url_returns_url_for_valid_id():
    resolve_trace_url = _resolve_trace_url()
    url = resolve_trace_url("abc-123")
    assert url == "https://smith.langchain.com/r/abc-123"


def test_resolve_trace_url_returns_none_for_none():
    resolve_trace_url = _resolve_trace_url()
    assert resolve_trace_url(None) is None


def test_resolve_trace_url_returns_none_for_empty_string():
    resolve_trace_url = _resolve_trace_url()
    assert resolve_trace_url("") is None


def test_resolve_trace_url_returns_none_for_whitespace():
    resolve_trace_url = _resolve_trace_url()
    assert resolve_trace_url("   ") is None


def test_resolve_trace_url_strips_whitespace():
    resolve_trace_url = _resolve_trace_url()
    url = resolve_trace_url("  trace-id  ")
    assert url is not None
    assert url.endswith("/trace-id")


def test_resolve_trace_url_encodes_special_chars():
    resolve_trace_url = _resolve_trace_url()
    url = resolve_trace_url("trace/id with spaces")
    assert url is not None
    assert "trace%2Fid%20with%20spaces" in url


def test_resolve_trace_url_uses_custom_base_url():
    resolve_trace_url = _resolve_trace_url()
    url = resolve_trace_url("abc", base_url="https://custom.example.com/traces/")
    assert url == "https://custom.example.com/traces/abc"


def test_resolve_trace_url_strips_trailing_slash_from_base():
    resolve_trace_url = _resolve_trace_url()
    url = resolve_trace_url("abc", base_url="https://example.com///")
    assert url == "https://example.com/abc"


def test_resolve_trace_url_from_run_object_url_attr():
    class _Run:
        url = "https://smith.langchain.com/r/abc-run"

    resolve_trace_url = _resolve_trace_url()
    assert resolve_trace_url(_Run()) == "https://smith.langchain.com/r/abc-run"


def test_resolve_trace_url_from_run_object_get_url_method():
    class _Run:
        def get_url(self) -> str:
            return "https://smith.langchain.com/r/get-url"

    resolve_trace_url = _resolve_trace_url()
    assert resolve_trace_url(_Run()) == "https://smith.langchain.com/r/get-url"
