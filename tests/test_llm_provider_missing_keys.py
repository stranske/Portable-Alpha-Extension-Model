"""Provider validation tests for missing credential keys."""

from __future__ import annotations

import socket

import pytest

from pa_core.llm.provider import LLMProviderConfig, create_llm


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


def test_create_llm_unsupported_provider_raises_value_error(socket_connect_guard):
    attempts, blocked = socket_connect_guard

    config = LLMProviderConfig(
        provider_name="nonexistent_provider",
        credentials={"api_key": "test-key"},
    )

    with pytest.raises(ValueError, match="Unsupported provider_name"):
        create_llm(config)

    assert socket.socket.connect is blocked
    assert attempts == []


def test_create_llm_empty_credential_treated_as_missing(socket_connect_guard):
    """Empty-string or whitespace-only credentials should be treated as missing."""
    attempts, blocked = socket_connect_guard

    config = LLMProviderConfig(
        provider_name="openai",
        credentials={"api_key": "   "},
    )

    with pytest.raises(ValueError, match="api_key"):
        create_llm(config)

    assert socket.socket.connect is blocked
    assert attempts == []
