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
        "Missing required credential keys for provider 'azure_openai': azure_endpoint, api_version"
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


def test_create_llm_anthropic_claude5_defaults(monkeypatch):
    """Default is Sonnet 5.5, and the always-thinking family gets an explicit max_tokens."""
    import sys
    import types

    import pa_core.llm.provider as provider

    captured: dict = {}

    class FakeChatAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules, "langchain_anthropic", types.SimpleNamespace(ChatAnthropic=FakeChatAnthropic)
    )
    create_llm(LLMProviderConfig(provider_name="anthropic", credentials={"api_key": "k"}))
    assert captured["model_name"] == "claude-sonnet-5-5"
    assert captured["max_tokens"] == provider.ANTHROPIC_THINKING_MAX_TOKENS
    assert "temperature" not in captured

    captured.clear()
    create_llm(
        LLMProviderConfig(
            provider_name="anthropic",
            credentials={"api_key": "k"},
            model_name="claude-sonnet-5-5",
            client_kwargs={"max_tokens": 2048},
        )
    )
    assert captured["max_tokens"] == 2048  # an explicit caller value wins


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-sonnet-5-5", True),
        ("claude-sonnet-5", True),
        ("claude-opus-5-5", True),
        ("claude-haiku-5", True),
        ("claude-fable-5-1", True),
        ("  Claude-Sonnet-5-5 ", True),
        ("claude-sonnet-4-6", False),
        ("claude-haiku-4-5", False),
        ("gpt-4o-mini", False),
    ],
)
def test_is_claude5_family(model: str, expected: bool) -> None:
    from pa_core.llm.provider import _is_claude5_family

    assert _is_claude5_family(model) is expected
