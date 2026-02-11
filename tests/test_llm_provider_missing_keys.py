"""Provider validation tests for missing credential keys."""

from __future__ import annotations

import socket

import pytest

from pa_core.llm.provider import LLMProviderConfig, create_llm


@pytest.fixture
def socket_connect_guard(monkeypatch: pytest.MonkeyPatch):
    attempts: list[object] = []

    def _blocked_connect(self, address):  # noqa: ANN001
        attempts.append(address)
        raise AssertionError("socket.connect should not be called during this test")

    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)
    return attempts, _blocked_connect


def test_create_llm_missing_credentials_raises_value_error(socket_connect_guard):
    attempts, blocked = socket_connect_guard

    config = LLMProviderConfig(
        provider_name="azure_openai",
        credentials={
            "api_key": "test-key",
        },
    )

    with pytest.raises(ValueError) as exc_info:
        create_llm(config)

    message = str(exc_info.value)
    assert message == (
        "Missing required credential keys for provider 'azure_openai': "
        "azure_endpoint, api_version"
    )
    assert "azure_endpoint" in message
    assert "api_version" in message
    assert socket.socket.connect is blocked
    assert attempts == []
