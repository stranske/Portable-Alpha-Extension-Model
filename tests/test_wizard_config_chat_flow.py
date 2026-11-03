from __future__ import annotations

import runpy

import streamlit as st

from pa_core.wizard_schema import AnalysisMode, get_default_config


def _mute_streamlit_ui(module: dict, monkeypatch) -> None:
    for name in ("success", "warning", "error", "caption", "markdown", "code", "info"):
        monkeypatch.setattr(module["st"], name, lambda *args, **kwargs: None)


def test_preview_generates_diff_without_mutating_live_config(monkeypatch) -> None:
    st.session_state.clear()
    try:
        module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
        _mute_streamlit_ui(module, monkeypatch)

        config = get_default_config(AnalysisMode.RETURNS)
        config.n_simulations = 1000
        config.sleeve_max_breach = 0.2
        st.session_state["wizard_config"] = config

        module["_preview_config_chat_change"](
            "increase simulations to 5000 and reduce breach tolerance"
        )

        preview = st.session_state.get(module["_CONFIG_CHAT_PREVIEW_KEY"])
        assert isinstance(preview, dict)
        assert st.session_state["wizard_config"].n_simulations == 1000
        assert "n_simulations" in preview["unified_diff"]
        assert preview["patch"]["set"]["n_simulations"] == 5000
    finally:
        st.session_state.clear()


def test_apply_validate_blocks_invalid_patch_without_mutation(monkeypatch) -> None:
    st.session_state.clear()
    try:
        module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
        _mute_streamlit_ui(module, monkeypatch)

        config = get_default_config(AnalysisMode.RETURNS)
        config.n_months = 12
        st.session_state["wizard_config"] = config
        st.session_state[module["_CONFIG_CHAT_PREVIEW_KEY"]] = {
            "patch": {"set": {"n_months": 0}, "merge": {}, "remove": []},
            "instruction": "set months to 0",
        }

        module["_apply_preview_patch"](validate_first=True)

        assert st.session_state["wizard_config"].n_months == 12
        assert st.session_state.get(module["_CONFIG_CHAT_HISTORY_KEY"], []) == []
    finally:
        st.session_state.clear()


def test_revert_restores_previous_config_and_session_mirrors(monkeypatch) -> None:
    st.session_state.clear()
    try:
        module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
        _mute_streamlit_ui(module, monkeypatch)

        config = get_default_config(AnalysisMode.RETURNS)
        config.total_fund_capital = 700.0
        st.session_state["wizard_config"] = config
        st.session_state["wizard_total_fund_capital"] = 700.0
        st.session_state[module["_CONFIG_CHAT_PREVIEW_KEY"]] = {
            "patch": {"set": {"total_fund_capital": 900.0}, "merge": {}, "remove": []},
            "instruction": "set total capital to 900",
        }

        module["_apply_preview_patch"](validate_first=False)
        assert st.session_state["wizard_config"].total_fund_capital == 900.0
        assert st.session_state["wizard_total_fund_capital"] == 900.0

        module["_revert_config_chat_change"]()
        assert st.session_state["wizard_config"].total_fund_capital == 700.0
        assert st.session_state["wizard_total_fund_capital"] == 700.0
    finally:
        st.session_state.clear()


def test_wizard_initial_provider_honors_environment_and_tracing_only_has_no_client(
    monkeypatch, socket_connect_guard
):
    import sys
    from types import SimpleNamespace

    from dashboard.components import llm_settings

    class Prompt:
        @staticmethod
        def from_messages(messages):
            return Prompt()

        def __or__(self, client):
            return client

    monkeypatch.setitem(
        sys.modules, "langchain_core.prompts", SimpleNamespace(ChatPromptTemplate=Prompt)
    )

    st.session_state.clear()
    env = {"PA_LLM_PROVIDER": "anthropic", "CLAUDE_API_STRANSKE": "synthetic-anthropic"}
    monkeypatch.setattr(llm_settings, "read_secret", env.get)
    try:
        module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
        captured = []

        def factory(config):
            captured.append(config)
            return SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content="safe"))

        # runpy returns the globals mapping used by the builder.
        module["_build_config_chat_llm_invoke"].__globals__["create_llm"] = factory
        invoke, provider, _ = module["_build_config_chat_llm_invoke"]()
        assert invoke is not None
        assert provider == "anthropic"
        assert captured[-1].credentials["api_key"] == "synthetic-anthropic"
        env.clear()
        env["LANGSMITH_API_KEY"] = "synthetic-tracing"
        invoke, _, _ = module["_build_config_chat_llm_invoke"]()
        assert invoke is None
        assert len(captured) == 1
        assert socket_connect_guard[0] == []
    finally:
        st.session_state.clear()


def test_config_chat_provider_scoped_credentials_with_fake_client_captures(
    monkeypatch, socket_connect_guard
):
    """Test that provider-scoped credentials are used correctly."""
    import sys
    from types import SimpleNamespace
    from dashboard.components import llm_settings

    monkeypatch.setitem(
        sys.modules, "langchain_core.prompts", SimpleNamespace(ChatPromptTemplate=object)
    )
    monkeypatch.setattr(llm_settings, "read_secret", lambda name: None)
    st.session_state.clear()
    try:
        module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
        captured = []
        module["_build_config_chat_llm_invoke"].__globals__["create_llm"] = (
            lambda config: captured.append(config)
        )

        # Set provider-scoped API keys for OpenAI and Anthropic
        openai_key = f'{module["_CONFIG_CHAT_API_KEY_KEY"]}::openai'
        anthropic_key = f'{module["_CONFIG_CHAT_API_KEY_KEY"]}::anthropic'
        st.session_state[openai_key] = "synthetic-openai-key"
        st.session_state[anthropic_key] = "synthetic-anthropic-key"

        # Test OpenAI
        st.session_state[module["_CONFIG_CHAT_PROVIDER_KEY"]] = "openai"
        module["_build_config_chat_llm_invoke"]()

        # Test Anthropic
        st.session_state[module["_CONFIG_CHAT_PROVIDER_KEY"]] = "anthropic"
        module["_build_config_chat_llm_invoke"]()

        # Verify each provider uses its own key
        assert [c.provider_name for c in captured] == ["openai", "anthropic"]
        assert [c.credentials["api_key"] for c in captured] == [
            "synthetic-openai-key",
            "synthetic-anthropic-key",
        ]
        assert socket_connect_guard[0] == []
    finally:
        st.session_state.clear()


def test_config_chat_openai_anthropic_azure_switch_uses_scoped_keys_and_env_endpoint(
    monkeypatch, socket_connect_guard
):
    """OpenAI -> Anthropic -> Azure must not reuse the previous provider's key."""
    import sys
    from types import SimpleNamespace

    from dashboard.components import llm_settings

    env = {
        "PA_LLM_BASE_URL": "https://example.openai.azure.com",
        "PA_LLM_API_VERSION": "2025-01-01",
    }
    monkeypatch.setitem(
        sys.modules, "langchain_core.prompts", SimpleNamespace(ChatPromptTemplate=object)
    )
    monkeypatch.setattr(llm_settings, "read_secret", env.get)
    st.session_state.clear()
    try:
        module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
        captured = []
        module["_build_config_chat_llm_invoke"].__globals__["create_llm"] = (
            lambda config: captured.append(config)
        )

        base = module["_CONFIG_CHAT_API_KEY_KEY"]
        st.session_state[f"{base}::openai"] = "synthetic-openai-key"
        st.session_state[f"{base}::anthropic"] = "synthetic-anthropic-key"
        st.session_state[f"{base}::azure_openai"] = "synthetic-azure-key"

        for provider in ("openai", "anthropic", "azure_openai"):
            st.session_state[module["_CONFIG_CHAT_PROVIDER_KEY"]] = provider
            module["_build_config_chat_llm_invoke"]()

        assert [c.provider_name for c in captured] == [
            "openai",
            "anthropic",
            "azure_openai",
        ]
        assert [c.credentials["api_key"] for c in captured] == [
            "synthetic-openai-key",
            "synthetic-anthropic-key",
            "synthetic-azure-key",
        ]
        azure_config = captured[-1]
        assert azure_config.credentials["azure_endpoint"] == env["PA_LLM_BASE_URL"]
        assert azure_config.credentials["api_version"] == env["PA_LLM_API_VERSION"]
        assert socket_connect_guard[0] == []
    finally:
        st.session_state.clear()


def test_config_chat_missing_destination_key_constructs_zero_clients(
    monkeypatch, socket_connect_guard
):
    """Test that missing destination credentials construct zero clients."""
    import sys
    from types import SimpleNamespace
    from dashboard.components import llm_settings

    monkeypatch.setitem(
        sys.modules, "langchain_core.prompts", SimpleNamespace(ChatPromptTemplate=object)
    )
    monkeypatch.setattr(llm_settings, "read_secret", lambda name: None)
    st.session_state.clear()
    try:
        module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
        captured = []
        module["_build_config_chat_llm_invoke"].__globals__["create_llm"] = (
            lambda config: captured.append(config)
        )

        # Set provider-scoped API key for OpenAI but not for Anthropic
        openai_key = f'{module["_CONFIG_CHAT_API_KEY_KEY"]}::openai'
        st.session_state[openai_key] = "synthetic-openai-key"

        # Test OpenAI - should construct a client
        st.session_state[module["_CONFIG_CHAT_PROVIDER_KEY"]] = "openai"
        result = module["_build_config_chat_llm_invoke"]()
        assert result[0] is not None  # invoke_llm is not None
        assert len(captured) == 1

        # Test Anthropic - should NOT construct a client (no key)
        st.session_state[module["_CONFIG_CHAT_PROVIDER_KEY"]] = "anthropic"
        result = module["_build_config_chat_llm_invoke"]()
        assert result[0] is None  # invoke_llm is None
        assert len(captured) == 1  # No new client captured

        assert socket_connect_guard[0] == []
    finally:
        st.session_state.clear()


def test_config_chat_return_to_provider_restores_override(monkeypatch, socket_connect_guard):
    """Test that returning to a provider restores its explicit override."""
    import sys
    from types import SimpleNamespace
    from dashboard.components import llm_settings

    monkeypatch.setitem(
        sys.modules, "langchain_core.prompts", SimpleNamespace(ChatPromptTemplate=object)
    )
    monkeypatch.setattr(llm_settings, "read_secret", lambda name: None)
    st.session_state.clear()
    try:
        module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
        captured = []
        module["_build_config_chat_llm_invoke"].__globals__["create_llm"] = (
            lambda config: captured.append(config)
        )

        # Set provider-scoped API keys
        openai_key = f'{module["_CONFIG_CHAT_API_KEY_KEY"]}::openai'
        anthropic_key = f'{module["_CONFIG_CHAT_API_KEY_KEY"]}::anthropic'
        st.session_state[openai_key] = "synthetic-openai-key"
        st.session_state[anthropic_key] = "synthetic-anthropic-key"

        # OpenAI -> Anthropic -> OpenAI
        for provider in ("openai", "anthropic", "openai"):
            st.session_state[module["_CONFIG_CHAT_PROVIDER_KEY"]] = provider
            module["_build_config_chat_llm_invoke"]()

        # Verify the keys are used correctly
        assert [c.provider_name for c in captured] == ["openai", "anthropic", "openai"]
        assert [c.credentials["api_key"] for c in captured] == [
            "synthetic-openai-key",
            "synthetic-anthropic-key",
            "synthetic-openai-key",  # Returning to OpenAI restores its key
        ]
        assert socket_connect_guard[0] == []
    finally:
        st.session_state.clear()


def test_audit_reproduces_config_chat_unscoped_override_without_network(
    monkeypatch, socket_connect_guard
):
    """Characterize the reported, unfixed gap; now confirms the fix is in place.

    The legacy unscoped key is discarded and without provider-scoped keys,
    no LLM clients are constructed (both return None).
    """
    import sys
    from types import SimpleNamespace
    from dashboard.components import llm_settings

    monkeypatch.setitem(
        sys.modules, "langchain_core.prompts", SimpleNamespace(ChatPromptTemplate=object)
    )
    monkeypatch.setattr(llm_settings, "read_secret", lambda name: None)
    st.session_state.clear()
    try:
        module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
        captured = []
        module["_build_config_chat_llm_invoke"].__globals__["create_llm"] = (
            lambda config: captured.append(config)
        )
        st.session_state[module["_CONFIG_CHAT_API_KEY_KEY"]] = "synthetic-audit-openai"

        # Both providers should return None (no client) because the unscoped key is discarded
        # and there are no provider-scoped keys
        for provider in ("openai", "anthropic"):
            st.session_state[module["_CONFIG_CHAT_PROVIDER_KEY"]] = provider
            result = module["_build_config_chat_llm_invoke"]()
            assert result[0] is None  # invoke_llm is None for both

        # No clients should have been created
        assert len(captured) == 0
        assert socket_connect_guard[0] == []
    finally:
        st.session_state.clear()
