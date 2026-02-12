"""Shared LLM settings component for the Portable Alpha dashboard.

This module provides UI-agnostic helpers that resolve provider, model, and
API-key configuration for the LLM-powered dashboard features (Explain,
Compare, Config Chat).  It is modelled after the Trend project's
``streamlit_app/components/llm_settings.py`` but adapted for Portable
Alpha's env-var conventions and the ``pa_core.llm`` provider interface.

Key responsibilities:

* Read secrets from ``st.secrets``, environment variables, or user input.
* Mask API keys so they never appear verbatim in the UI or logs.
* Resolve a ``LLMProviderConfig`` compatible with
  ``pa_core.llm.create_llm()``.

Environment variables
---------------------
PA_LLM_PROVIDER      – provider name: "openai" | "anthropic" | "azure_openai"
                       (default: "openai")
PA_LLM_MODEL         – model override (default: per-provider, see
                       ``pa_core.llm.provider``)
PA_LLM_BASE_URL      – optional custom endpoint URL
PA_LLM_ORG           – optional organisation id
PA_STREAMLIT_API_KEY  – explicit API key for Streamlit dashboard use

Fallback key lookup: OPENAI_API_KEY (openai), CLAUDE_API_STRANSKE
(anthropic), LANGSMITH_API_KEY (LangSmith tracing).
"""

from __future__ import annotations

import os
import re
from typing import Any

from pa_core.llm.provider import LLMProviderConfig

# ---------------------------------------------------------------------------
# Environment variable names
# ---------------------------------------------------------------------------

ENV_PROVIDER = "PA_LLM_PROVIDER"
ENV_MODEL = "PA_LLM_MODEL"
ENV_BASE_URL = "PA_LLM_BASE_URL"
ENV_ORG = "PA_LLM_ORG"
ENV_STREAMLIT_API_KEY = "PA_STREAMLIT_API_KEY"

# Provider-specific fallback key env vars
_PROVIDER_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "CLAUDE_API_STRANSKE",
}

# LangSmith tracing key
ENV_LANGSMITH_API_KEY = "LANGSMITH_API_KEY"

# Minimum visible chars when masking a key
_MASK_VISIBLE = 4

# Regex for strings that look like env-var names (all-caps, underscores)
_ENV_VAR_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_PROVIDER = "openai"


def default_provider() -> str:
    """Return provider from ``PA_LLM_PROVIDER`` with a safe default."""
    value = read_secret(ENV_PROVIDER)
    if not value:
        return DEFAULT_PROVIDER
    return value.strip().lower()


def default_model() -> str | None:
    """Return model from ``PA_LLM_MODEL`` when configured."""
    return read_secret(ENV_MODEL)


def default_base_url() -> str | None:
    """Return custom endpoint URL from ``PA_LLM_BASE_URL`` when configured."""
    return read_secret(ENV_BASE_URL)


def default_org() -> str | None:
    """Return organization id from ``PA_LLM_ORG`` when configured."""
    return read_secret(ENV_ORG)


def default_streamlit_api_key() -> str | None:
    """Return explicit dashboard key from ``PA_STREAMLIT_API_KEY`` when set."""
    return read_secret(ENV_STREAMLIT_API_KEY)


# ---------------------------------------------------------------------------
# API key sanitisation
# ---------------------------------------------------------------------------


def sanitize_api_key(key: str | None) -> str:
    """Return a masked version of *key* safe for display.

    >>> sanitize_api_key("sk-abc123xyz")
    'sk-a***xyz'
    >>> sanitize_api_key("")
    '(not set)'
    >>> sanitize_api_key(None)
    '(not set)'
    """
    if not key or not key.strip():
        return "(not set)"
    key = key.strip()
    # Replace non-printable characters so display/log sinks cannot be
    # influenced by control bytes while still masking the key shape.
    key = "".join(ch if ch.isprintable() else "?" for ch in key)
    if len(key) <= _MASK_VISIBLE * 2:
        return "*" * len(key)
    return key[:_MASK_VISIBLE] + "***" + key[-3:]


# ---------------------------------------------------------------------------
# Secret reading
# ---------------------------------------------------------------------------


def read_secret(name: str) -> str | None:
    """Read a secret by *name* from ``st.secrets`` then the environment.

    Returns ``None`` when the secret cannot be found in either location.
    """
    # Try st.secrets first (only when Streamlit is importable and running)
    try:
        import streamlit as st  # optional import

        value = st.secrets.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    except Exception:
        pass

    value = os.environ.get(name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


# ---------------------------------------------------------------------------
# Resolve key input (raw key or env-var name)
# ---------------------------------------------------------------------------


def resolve_api_key_input(raw_input: str | None) -> str | None:
    """Resolve *raw_input* to an API key value.

    If *raw_input* looks like an environment variable name (all uppercase,
    underscores) it is treated as an env-var reference and resolved via
    :func:`read_secret`.  Otherwise it is returned as-is (the literal key).

    Returns ``None`` when no value can be determined.
    """
    if not raw_input or not raw_input.strip():
        return None

    raw_input = raw_input.strip()

    # If it looks like an env-var name, resolve it
    if _ENV_VAR_PATTERN.match(raw_input):
        return read_secret(raw_input)

    # Treat as a literal key
    return raw_input


# ---------------------------------------------------------------------------
# Default API key resolution
# ---------------------------------------------------------------------------


def default_api_key(provider: str | None = None) -> str | None:
    """Return the best available API key for *provider*.

    Checks, in order:

    1. ``PA_STREAMLIT_API_KEY``
    2. The provider-specific env var (``OPENAI_API_KEY`` for openai,
       ``CLAUDE_API_STRANSKE`` for anthropic).
    3. ``LANGSMITH_API_KEY`` (only when no provider key found – useful
       for tracing configuration).

    Returns ``None`` when no key is available.
    """
    # 1. Explicit streamlit key
    key = default_streamlit_api_key()
    if key:
        return key

    # 2. Provider-specific fallback
    provider = (provider or default_provider()).strip().lower()
    env_name = _PROVIDER_KEY_ENV.get(provider)
    if env_name:
        key = read_secret(env_name)
        if key:
            return key

    # 3. LangSmith fallback (tracing use-case)
    return read_secret(ENV_LANGSMITH_API_KEY)


# ---------------------------------------------------------------------------
# Full provider config resolution
# ---------------------------------------------------------------------------


def resolve_llm_provider_config(
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    org: str | None = None,
) -> LLMProviderConfig:
    """Build a :class:`LLMProviderConfig` from explicit args + env defaults.

    Parameters
    ----------
    provider
        Provider name override.  Falls back to ``PA_LLM_PROVIDER`` then
        ``"openai"``.
    model
        Model name override.  Falls back to ``PA_LLM_MODEL`` (or
        provider default when ``None``).
    api_key
        Raw API key or env-var name.  Falls back to :func:`default_api_key`.
    base_url
        Custom base URL override.  Falls back to ``PA_LLM_BASE_URL``.
    org
        Organisation id.  Falls back to ``PA_LLM_ORG``.

    Returns
    -------
    LLMProviderConfig
        Ready to pass to ``pa_core.llm.create_llm()``.

    Raises
    ------
    ValueError
        When no API key can be resolved.  The error message includes
        guidance on which env var to set but **never** includes the key
        value itself.
    """
    resolved_provider = (provider or default_provider()).strip().lower()

    resolved_model = model or default_model()

    # Key resolution: explicit → resolve input → default
    resolved_key = resolve_api_key_input(api_key) or default_api_key(resolved_provider)

    if not resolved_key:
        env_hint = _PROVIDER_KEY_ENV.get(resolved_provider, ENV_STREAMLIT_API_KEY)
        raise ValueError(
            f"No API key found for provider '{resolved_provider}'. "
            f"Set the {env_hint} environment variable or provide a key "
            f"in the dashboard settings."
        )

    resolved_base_url = base_url or default_base_url()
    resolved_org = org or default_org()

    client_kwargs: dict[str, Any] = {}
    if resolved_base_url:
        client_kwargs["base_url"] = resolved_base_url
    if resolved_org:
        client_kwargs["organization"] = resolved_org

    credentials: dict[str, str] = {"api_key": resolved_key}

    return LLMProviderConfig(
        provider_name=resolved_provider,
        credentials=credentials,
        model_name=resolved_model,
        client_kwargs=client_kwargs,
    )


__all__ = [
    "ENV_BASE_URL",
    "ENV_LANGSMITH_API_KEY",
    "ENV_MODEL",
    "ENV_ORG",
    "ENV_PROVIDER",
    "ENV_STREAMLIT_API_KEY",
    "DEFAULT_PROVIDER",
    "default_api_key",
    "default_base_url",
    "default_model",
    "default_org",
    "default_provider",
    "default_streamlit_api_key",
    "read_secret",
    "resolve_api_key_input",
    "resolve_llm_provider_config",
    "sanitize_api_key",
]
