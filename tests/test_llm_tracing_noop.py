"""Tracing module tests focused on offline-safe behavior."""

from __future__ import annotations

import importlib
import socket
import sys

import pytest


@pytest.fixture
def socket_connect_guard(monkeypatch: pytest.MonkeyPatch):
    attempts: list[object] = []

    def _blocked_connect(self, address):  # noqa: ANN001
        attempts.append(address)
        raise AssertionError("socket.connect should not be called during this test")

    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)
    return attempts, _blocked_connect


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
