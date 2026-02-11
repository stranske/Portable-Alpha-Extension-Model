"""Tests for dashboard.components.llm_settings."""

from __future__ import annotations

import os
import re
import sys
import types
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Ensure project root importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dashboard.components.llm_settings import (  # noqa: E402
    default_api_key,
    default_base_url,
    default_model,
    default_org,
    default_provider,
    default_streamlit_api_key,
    read_secret,
    resolve_api_key_input,
    resolve_llm_provider_config,
    sanitize_api_key,
)

# Pattern that matches common API key prefixes – used to verify no secrets
# leak into captured output.
_SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9]{10,}|ghp_[A-Za-z0-9]{10,}")


# ===================================================================
# env default readers
# ===================================================================


class TestEnvDefaultReaders:
    """Verify PA_* env var defaults are read with safe fallback behavior."""

    def _clean_env(self) -> dict[str, str]:
        remove = {
            "PA_LLM_PROVIDER",
            "PA_LLM_MODEL",
            "PA_LLM_BASE_URL",
            "PA_LLM_ORG",
            "PA_STREAMLIT_API_KEY",
        }
        return {k: v for k, v in os.environ.items() if k not in remove}

    def test_provider_default_openai_when_unset(self) -> None:
        with mock.patch.dict(os.environ, self._clean_env(), clear=True):
            assert default_provider() == "openai"

    def test_provider_from_env(self) -> None:
        env = {**self._clean_env(), "PA_LLM_PROVIDER": "Anthropic"}
        with mock.patch.dict(os.environ, env, clear=True):
            assert default_provider() == "anthropic"

    def test_model_from_env(self) -> None:
        env = {**self._clean_env(), "PA_LLM_MODEL": "gpt-4o-mini"}
        with mock.patch.dict(os.environ, env, clear=True):
            assert default_model() == "gpt-4o-mini"

    def test_base_url_from_env(self) -> None:
        env = {**self._clean_env(), "PA_LLM_BASE_URL": "https://example.test"}
        with mock.patch.dict(os.environ, env, clear=True):
            assert default_base_url() == "https://example.test"

    def test_org_from_env(self) -> None:
        env = {**self._clean_env(), "PA_LLM_ORG": "org-abc"}
        with mock.patch.dict(os.environ, env, clear=True):
            assert default_org() == "org-abc"

    def test_streamlit_key_from_env(self) -> None:
        env = {**self._clean_env(), "PA_STREAMLIT_API_KEY": "streamlit-key"}
        with mock.patch.dict(os.environ, env, clear=True):
            assert default_streamlit_api_key() == "streamlit-key"


# ===================================================================
# sanitize_api_key
# ===================================================================


class TestSanitizeApiKey:
    """Verify keys are masked so they never appear verbatim."""

    def test_none(self) -> None:
        assert sanitize_api_key(None) == "(not set)"

    def test_empty(self) -> None:
        assert sanitize_api_key("") == "(not set)"

    def test_whitespace_only(self) -> None:
        assert sanitize_api_key("   ") == "(not set)"

    def test_short_key_fully_masked(self) -> None:
        result = sanitize_api_key("abcd1234")
        assert "*" in result
        # Full key should NOT appear
        assert "abcd1234" not in result

    def test_long_key_partial_reveal(self) -> None:
        key = "sk-abc123xyz7890longkey"
        result = sanitize_api_key(key)
        # First 4 chars visible, last 3 visible, stars in middle
        assert result.startswith("sk-a")
        assert result.endswith("key")
        assert "***" in result
        # Full key never appears
        assert key not in result

    def test_key_not_in_output(self) -> None:
        """No complete key value should survive sanitisation."""
        key = "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz"
        result = sanitize_api_key(key)
        assert key not in result

    def test_special_characters_are_masked_without_leak(self) -> None:
        key = "  sk-!@#$_+[]{}()|:;,./?<>  "
        result = sanitize_api_key(key)
        assert result.startswith("sk-!")
        assert result.endswith("<>")
        assert "***" in result
        assert key.strip() not in result

    @pytest.mark.parametrize("key", ["abcdefgh", "abcd1234"])
    def test_boundary_length_fully_masked(self, key: str) -> None:
        assert sanitize_api_key(key) == "*" * len(key)


# ===================================================================
# read_secret
# ===================================================================


class TestReadSecret:
    """Test st.secrets + environment resolution order."""

    def test_prefers_streamlit_secrets(self) -> None:
        fake_streamlit = types.SimpleNamespace(secrets={"OPENAI_API_KEY": "from-secrets"})
        with mock.patch.dict(sys.modules, {"streamlit": fake_streamlit}):
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "from-env"}, clear=True):
                assert read_secret("OPENAI_API_KEY") == "from-secrets"

    def test_falls_back_to_environment(self) -> None:
        with mock.patch.dict(sys.modules, {"streamlit": None}):
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "from-env"}, clear=True):
                assert read_secret("OPENAI_API_KEY") == "from-env"

    def test_returns_none_for_missing_or_blank(self) -> None:
        fake_streamlit = types.SimpleNamespace(secrets={"OPENAI_API_KEY": "   "})
        with mock.patch.dict(sys.modules, {"streamlit": fake_streamlit}):
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": " "}, clear=True):
                assert read_secret("OPENAI_API_KEY") is None


# ===================================================================
# resolve_api_key_input
# ===================================================================


class TestResolveApiKeyInput:
    """Test direct key vs env-var name resolution."""

    def test_none(self) -> None:
        assert resolve_api_key_input(None) is None

    def test_empty(self) -> None:
        assert resolve_api_key_input("") is None

    def test_literal_key(self) -> None:
        # A value that does NOT look like an env-var name
        assert resolve_api_key_input("sk-abc123") == "sk-abc123"

    def test_env_var_name_resolves(self) -> None:
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "resolved-value"}):
            assert resolve_api_key_input("OPENAI_API_KEY") == "resolved-value"

    def test_env_var_name_missing(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            # Patch st.secrets away
            with mock.patch.dict("sys.modules", {"streamlit": None}):
                assert resolve_api_key_input("MISSING_VAR") is None

    def test_strips_whitespace(self) -> None:
        assert resolve_api_key_input("  sk-abc123  ") == "sk-abc123"

    @pytest.mark.parametrize(
        "raw_input",
        [
            "openai_api_key",  # lowercase should not be treated as env var
            "OPENAI-API-KEY",  # disallowed character
            "AB",  # too short for env-var pattern
        ],
    )
    def test_invalid_env_var_formats_are_treated_as_literals(self, raw_input: str) -> None:
        assert resolve_api_key_input(raw_input) == raw_input


# ===================================================================
# default_api_key
# ===================================================================


class TestDefaultApiKey:
    """Test fallback chain for key resolution."""

    def _clean_env(self) -> dict[str, str]:
        """Return env dict with all LLM key vars removed."""
        remove = {
            "PA_STREAMLIT_API_KEY",
            "OPENAI_API_KEY",
            "CLAUDE_API_STRANSKE",
            "LANGSMITH_API_KEY",
        }
        return {k: v for k, v in os.environ.items() if k not in remove}

    def test_explicit_streamlit_key(self) -> None:
        env = {**self._clean_env(), "PA_STREAMLIT_API_KEY": "streamlit-key"}
        with mock.patch.dict(os.environ, env, clear=True):
            assert default_api_key("openai") == "streamlit-key"

    def test_openai_fallback(self) -> None:
        env = {**self._clean_env(), "OPENAI_API_KEY": "openai-key"}
        with mock.patch.dict(os.environ, env, clear=True):
            assert default_api_key("openai") == "openai-key"

    def test_anthropic_fallback(self) -> None:
        env = {**self._clean_env(), "CLAUDE_API_STRANSKE": "claude-key"}
        with mock.patch.dict(os.environ, env, clear=True):
            assert default_api_key("anthropic") == "claude-key"

    def test_langsmith_fallback(self) -> None:
        env = {**self._clean_env(), "LANGSMITH_API_KEY": "langsmith-key"}
        with mock.patch.dict(os.environ, env, clear=True):
            assert default_api_key("openai") == "langsmith-key"

    def test_no_key_returns_none(self) -> None:
        with mock.patch.dict(os.environ, self._clean_env(), clear=True):
            assert default_api_key("openai") is None

    def test_streamlit_key_preferred_over_provider(self) -> None:
        env = {
            **self._clean_env(),
            "PA_STREAMLIT_API_KEY": "preferred",
            "OPENAI_API_KEY": "fallback",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            assert default_api_key("openai") == "preferred"

    def test_none_provider_defaults_to_openai(self) -> None:
        env = {**self._clean_env(), "OPENAI_API_KEY": "openai-key"}
        with mock.patch.dict(os.environ, env, clear=True):
            assert default_api_key(None) == "openai-key"


# ===================================================================
# resolve_llm_provider_config
# ===================================================================


class TestResolveLlmProviderConfig:
    """Test full provider config resolution."""

    def _clean_env(self) -> dict[str, str]:
        remove = {
            "PA_LLM_PROVIDER",
            "PA_LLM_MODEL",
            "PA_LLM_BASE_URL",
            "PA_LLM_ORG",
            "PA_STREAMLIT_API_KEY",
            "OPENAI_API_KEY",
            "CLAUDE_API_STRANSKE",
            "LANGSMITH_API_KEY",
        }
        return {k: v for k, v in os.environ.items() if k not in remove}

    def test_explicit_args(self) -> None:
        config = resolve_llm_provider_config(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="test-key",
        )
        assert config.provider_name == "anthropic"
        assert config.model_name == "claude-sonnet-4-20250514"
        assert config.credentials["api_key"] == "test-key"

    def test_env_defaults(self) -> None:
        env = {
            **self._clean_env(),
            "PA_LLM_PROVIDER": "anthropic",
            "PA_LLM_MODEL": "claude-haiku",
            "CLAUDE_API_STRANSKE": "env-key",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = resolve_llm_provider_config()
            assert config.provider_name == "anthropic"
            assert config.model_name == "claude-haiku"
            assert config.credentials["api_key"] == "env-key"

    def test_missing_key_raises(self) -> None:
        with mock.patch.dict(os.environ, self._clean_env(), clear=True):
            with pytest.raises(ValueError, match="No API key found"):
                resolve_llm_provider_config(provider="openai")

    def test_error_message_no_secrets(self) -> None:
        """Error message must NOT contain any key values."""
        with mock.patch.dict(os.environ, self._clean_env(), clear=True):
            with pytest.raises(ValueError) as exc_info:
                resolve_llm_provider_config(provider="openai")
            msg = str(exc_info.value)
            assert "OPENAI_API_KEY" in msg  # env-var hint present
            assert not _SECRET_PATTERN.search(msg)

    def test_base_url_and_org(self) -> None:
        config = resolve_llm_provider_config(
            provider="openai",
            api_key="test-key",
            base_url="https://custom.api.com",
            org="org-123",
        )
        assert config.client_kwargs["base_url"] == "https://custom.api.com"
        assert config.client_kwargs["organization"] == "org-123"

    def test_default_provider_is_openai(self) -> None:
        env = {**self._clean_env(), "OPENAI_API_KEY": "key"}
        with mock.patch.dict(os.environ, env, clear=True):
            config = resolve_llm_provider_config()
            assert config.provider_name == "openai"

    def test_env_var_as_api_key_input(self) -> None:
        env = {**self._clean_env(), "OPENAI_API_KEY": "resolved-via-env"}
        with mock.patch.dict(os.environ, env, clear=True):
            config = resolve_llm_provider_config(
                provider="openai",
                api_key="OPENAI_API_KEY",
            )
            assert config.credentials["api_key"] == "resolved-via-env"

    def test_unset_env_vars_no_exception(self) -> None:
        """Importing and resolving with a key must not raise even when
        PA_LLM_PROVIDER, PA_LLM_MODEL, etc. are unset."""
        with mock.patch.dict(os.environ, self._clean_env(), clear=True):
            config = resolve_llm_provider_config(api_key="fallback-key")
            assert config.provider_name == "openai"
            assert config.credentials["api_key"] == "fallback-key"


# ===================================================================
# No secrets in test output
# ===================================================================


class TestNoSecretsLeaked:
    """Verify that test execution produces no secret-like output."""

    def test_sanitize_never_leaks(self, capsys: pytest.CaptureFixture[str]) -> None:
        key = "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz01234"
        result = sanitize_api_key(key)
        print(result)
        captured = capsys.readouterr()
        assert key not in captured.out
        assert key not in captured.err
        assert not _SECRET_PATTERN.search(captured.out)
