"""Import-safety checks for pa_core.llm."""

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


def test_import_pa_core_llm_no_network(socket_connect_guard):
    attempts, blocked = socket_connect_guard

    for name in list(sys.modules):
        if name == "pa_core.llm" or name.startswith("pa_core.llm."):
            sys.modules.pop(name)

    importlib.import_module("pa_core.llm")

    assert socket.socket.connect is blocked
    assert attempts == []
