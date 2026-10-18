"""Provider validation tests for missing credential keys."""

from __future__ import annotations

import importlib.util
import socket

import pytest

from dashboard.components.llm_settings import resolve_llm_provider_config
from pa_core.llm.provider import LLMProviderConfig, create_llm


def _require_real_provider_clients() -> None:
    missing = [
        module
        for module in ("langchain_openai", "langchain_anthropic")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        pytest.skip(
            f"real provider option matrix requires optional LLM extras: {', '.join(missing)}"
        )


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


def test_create_llm_provider_options_keep_supported_fields_out_of_anthropic_payload(
    socket_connect_guard,
):
    """Real SDK clients accept only their supported resolver options without network I/O."""

    _require_real_provider_clients()
    attempts, blocked = socket_connect_guard
    openai = create_llm(
        resolve_llm_provider_config(provider="openai", api_key="sk-synthetic", org="org-openai")
    )
    anthropic = create_llm(
        resolve_llm_provider_config(
            provider="anthropic",
            api_key="sk-synthetic",
            base_url="https://anthropic.example.test",
            org="org-anthropic",
        )
    )
    azure = create_llm(
        resolve_llm_provider_config(
            provider="azure_openai",
            api_key="sk-synthetic",
            base_url="https://portable-alpha.openai.azure.com",
            api_version="2024-10-21",
            org="org-azure",
        )
    )

    assert openai.openai_organization == "org-openai"
    assert "organization" not in openai.model_kwargs
    assert "organization" not in openai._get_request_payload([("human", "hello")])

    assert anthropic.anthropic_api_url == "https://anthropic.example.test"
    assert "organization" not in anthropic.model_kwargs
    assert "organization" not in anthropic._get_request_payload([("human", "hello")])

    assert azure.openai_organization == "org-azure"
    assert azure.azure_endpoint == "https://portable-alpha.openai.azure.com"
    assert azure.openai_api_version == "2024-10-21"
    assert "organization" not in azure.model_kwargs
    assert "organization" not in azure._get_request_payload([("human", "hello")])
    assert socket.socket.connect is blocked
    assert attempts == []


def test_real_provider_option_matrix_skips_when_optional_sdk_is_unavailable(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda module: None)

    with pytest.raises(pytest.skip.Exception, match="optional LLM extras"):
        _require_real_provider_clients()


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


@pytest.mark.parametrize("provider", ["openai", "anthropic", "azure_openai"])
def test_tracing_only_is_not_model_authorization(monkeypatch, socket_connect_guard, provider):
    from dashboard.components import llm_settings

    monkeypatch.setattr(llm_settings, "read_secret", {"LANGSMITH_API_KEY": "synthetic-tracing"}.get)
    with pytest.raises(ValueError, match="No API key found for provider"):
        llm_settings.resolve_llm_provider_config(provider=provider)
    assert socket_connect_guard[0] == []
