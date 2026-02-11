"""Provider factory helpers for chat-model clients."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

_REQUIRED_CREDENTIAL_KEYS: dict[str, tuple[str, ...]] = {
    "anthropic": ("api_key",),
    "azure_openai": ("api_key", "azure_endpoint", "api_version"),
    "openai": ("api_key",),
}


@dataclass(frozen=True)
class LLMProviderConfig:
    """Configuration for building an LLM provider client."""

    provider_name: str
    credentials: Mapping[str, str]
    model_name: str | None = None
    client_kwargs: Mapping[str, Any] = field(default_factory=dict)


def _missing_credential_keys(config: LLMProviderConfig) -> list[str]:
    required = _REQUIRED_CREDENTIAL_KEYS.get(config.provider_name.strip().lower())
    if required is None:
        raise ValueError(f"Unsupported provider_name: {config.provider_name}")

    missing: list[str] = []
    for key in required:
        value = config.credentials.get(key)
        if not isinstance(value, str) or not value.strip():
            missing.append(key)
    return missing


def create_llm(config: LLMProviderConfig) -> Any:
    """Create and return a provider client instance from ``config``."""

    missing = _missing_credential_keys(config)
    if missing:
        missing_csv = ", ".join(missing)
        raise ValueError(
            f"Missing required credential keys for provider '{config.provider_name}': {missing_csv}"
        )

    provider_name = config.provider_name.strip().lower()
    model_name = config.model_name or "gpt-4o-mini"

    if provider_name == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            api_key=config.credentials["api_key"],
            **dict(config.client_kwargs),
        )

    if provider_name == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model_name,
            api_key=config.credentials["api_key"],
            **dict(config.client_kwargs),
        )

    if provider_name == "azure_openai":
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            model=model_name,
            api_key=config.credentials["api_key"],
            azure_endpoint=config.credentials["azure_endpoint"],
            api_version=config.credentials["api_version"],
            **dict(config.client_kwargs),
        )

    raise ValueError(f"Unsupported provider_name: {config.provider_name}")
