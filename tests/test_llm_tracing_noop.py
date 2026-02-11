"""Tracing module tests focused on offline-safe behavior."""

from __future__ import annotations

import importlib
import socket
import sys

import pytest

from pa_core.llm.tracing import resolve_trace_url


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


# ---------- resolve_trace_url tests ----------


def test_resolve_trace_url_returns_url_for_valid_id():
    url = resolve_trace_url("abc-123")
    assert url == "https://smith.langchain.com/r/abc-123"


def test_resolve_trace_url_returns_none_for_none():
    assert resolve_trace_url(None) is None


def test_resolve_trace_url_returns_none_for_empty_string():
    assert resolve_trace_url("") is None


def test_resolve_trace_url_returns_none_for_whitespace():
    assert resolve_trace_url("   ") is None


def test_resolve_trace_url_strips_whitespace():
    url = resolve_trace_url("  trace-id  ")
    assert url is not None
    assert url.endswith("/trace-id")


def test_resolve_trace_url_encodes_special_chars():
    url = resolve_trace_url("trace/id with spaces")
    assert url is not None
    assert "trace%2Fid%20with%20spaces" in url


def test_resolve_trace_url_uses_custom_base_url():
    url = resolve_trace_url("abc", base_url="https://custom.example.com/traces/")
    assert url == "https://custom.example.com/traces/abc"


def test_resolve_trace_url_strips_trailing_slash_from_base():
    url = resolve_trace_url("abc", base_url="https://example.com///")
    assert url == "https://example.com/abc"
