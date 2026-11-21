from __future__ import annotations

import runpy
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import streamlit as st

import dashboard.components.config_chat as config_chat_module
from pa_core.llm.config_patch import validate_patch_dict
from pa_core.llm.config_patch_chain import ConfigPatchChainResult
from pa_core.wizard_schema import AnalysisMode, get_default_config

_CONFIG_CHAT_INSTRUCTION = "increase simulations to 5000"
_CONFIG_CHAT_INSTALL_NOTICE = "LLM features unavailable. Install .[llm] to enable Config Chat."
# Pre-fix readback at e1ec34c8: configured credentials + missing optional deps still
# applied heuristics but omitted missing_extra labeling and the install notice.
_LEGACY_MISSING_EXTRA_READBACK = {"risk_flags": [], "install_notice": None}
# Recorded keepalive verification (input, observed missing-extra output, provider control).
_CONFIG_CHAT_MISSING_EXTRA_VERIFICATION: dict[str, Any] = {
    "instruction": _CONFIG_CHAT_INSTRUCTION,
    "missing_extra_observed": {
        "risk_flags": ["missing_extra"],
        "install_notice": _CONFIG_CHAT_INSTALL_NOTICE,
        "summary": "Applied heuristic instruction parsing.",
        "n_simulations": 5000,
    },
    "configured_provider_control": {
        "risk_flags": [],
        "install_notice": None,
        "summary": "Provider patch accepted.",
        "n_simulations": 8000,
    },
}


class _ConfigChatPanelColumnContext:
    def __enter__(self) -> "_ConfigChatPanelColumnContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _ConfigChatPanelFakeStreamlit:
    """Minimal Streamlit stub for exercising render_config_chat_panel."""

    def __init__(
        self, *, button_values: dict[str, bool], session_state: dict[str, Any] | None = None
    ):
        self.button_values = dict(button_values)
        self.session_state: dict[str, Any] = session_state or {}
        self.messages: list[tuple[str, str]] = []

    def subheader(self, message: str) -> None:
        self.messages.append(("subheader", message))

    def caption(self, message: str) -> None:
        self.messages.append(("caption", message))

    def text_area(self, label: str, value: str, key: str, help: str) -> str:
        self.session_state.setdefault(key, value)
        return str(self.session_state[key])

    def columns(self, n: int) -> list[_ConfigChatPanelColumnContext]:
        return [_ConfigChatPanelColumnContext() for _ in range(n)]

    def button(self, label: str, key: str) -> bool:
        return bool(self.button_values.get(key, False))

    def warning(self, message: str) -> None:
        self.messages.append(("warning", message))

    def error(self, message: str) -> None:
        self.messages.append(("error", message))

    def success(self, message: str) -> None:
        self.messages.append(("success", message))

    def info(self, message: str) -> None:
        self.messages.append(("info", message))

    def markdown(self, message: str) -> None:
        self.messages.append(("markdown", message))

    def code(self, body: str, language: str | None = None) -> None:
        self.messages.append(("code", body))


def _heuristic_preview_dict(result: ConfigPatchChainResult) -> dict[str, Any]:
    """Preview-shaped dict from a heuristic ConfigPatchChainResult (wizard install_notice rule)."""
    risk_flags = list(result.risk_flags)
    install_notice = _CONFIG_CHAT_INSTALL_NOTICE if "missing_extra" in risk_flags else None
    return {
        "patch": {
            "set": dict(result.patch.set),
            "merge": dict(result.patch.merge),
            "remove": list(result.patch.remove),
        },
        "summary": result.summary,
        "risk_flags": risk_flags,
        "install_notice": install_notice,
    }


def _matches_legacy_missing_extra_readback(preview: dict[str, Any]) -> bool:
    """Silent heuristic fallback before missing_extra labeling (e1ec34c8 and earlier)."""
    set_ops = preview.get("patch", {}).get("set", {})
    return (
        preview.get("install_notice") is None
        and "missing_extra" not in preview.get("risk_flags", [])
        and set_ops.get("n_simulations") == 5000
        and preview.get("summary") == "Applied heuristic instruction parsing."
    )


def _assert_missing_extra_install_contract(preview: dict[str, Any]) -> None:
    """Post-fix contract: offline heuristics plus actionable install notice."""
    observed = _CONFIG_CHAT_MISSING_EXTRA_VERIFICATION["missing_extra_observed"]
    assert preview["patch"]["set"]["n_simulations"] == observed["n_simulations"]
    assert preview["summary"] == observed["summary"]
    assert preview["risk_flags"] == observed["risk_flags"]
    assert preview["install_notice"] == observed["install_notice"]
    assert "llm_fallback_heuristic" not in preview["risk_flags"]
    assert not _matches_legacy_missing_extra_readback(preview)


def _assert_configured_provider_control(preview: dict[str, Any]) -> None:
    control = _CONFIG_CHAT_MISSING_EXTRA_VERIFICATION["configured_provider_control"]
    assert preview.get("install_notice") == control["install_notice"]
    assert preview["risk_flags"] == control["risk_flags"]
    assert preview["patch"]["set"]["n_simulations"] == control["n_simulations"]
    assert preview["summary"] == control["summary"]


def _assert_missing_extra_vs_configured_provider_contrast(
    missing_extra_preview: dict[str, Any],
    configured_preview: dict[str, Any],
) -> None:
    """Same verification input: label missing extras without breaking healthy provider previews."""
    assert missing_extra_preview["risk_flags"] != configured_preview["risk_flags"]
    assert "missing_extra" in missing_extra_preview["risk_flags"]
    assert "missing_extra" not in configured_preview["risk_flags"]
    assert missing_extra_preview.get("install_notice") == _CONFIG_CHAT_INSTALL_NOTICE
    assert configured_preview.get("install_notice") is None
    assert not _matches_legacy_missing_extra_readback(missing_extra_preview)


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
        invoke, provider, _, missing_extra = module["_build_config_chat_llm_invoke"]()
        assert invoke is not None
        assert provider == "anthropic"
        assert missing_extra is False
        assert captured[-1].credentials["api_key"] == "synthetic-anthropic"
        env.clear()
        env["LANGSMITH_API_KEY"] = "synthetic-tracing"
        invoke, _, _, missing_extra = module["_build_config_chat_llm_invoke"]()
        assert missing_extra is False
        assert invoke is None
        assert len(captured) == 1
        assert socket_connect_guard[0] == []
    finally:
        st.session_state.clear()


def test_config_chat_missing_extra_explains_offline_fallback(monkeypatch) -> None:
    """Missing optional LLM deps keep heuristics and surface the install notice."""
    st.session_state.clear()
    try:
        module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
        config = get_default_config(AnalysisMode.RETURNS)
        config.n_simulations = 1000
        st.session_state["wizard_config"] = config
        st.session_state[module["_CONFIG_CHAT_API_KEY_KEY"]] = "synthetic-openai"

        globals_ = module["_build_config_chat_llm_invoke"].__globals__

        def _raise_missing_extra(*_args, **_kwargs) -> None:
            raise ModuleNotFoundError("langchain_openai")

        globals_["create_llm"] = _raise_missing_extra
        globals_["resolve_llm_provider_config"] = lambda **_kwargs: SimpleNamespace(
            provider_name="openai",
            model_name="gpt-4o-mini",
        )

        preview_fn = module["_preview_config_chat_instruction"]
        preview = preview_fn(_CONFIG_CHAT_INSTRUCTION)
        _assert_missing_extra_install_contract(preview)

        panel_fake = _ConfigChatPanelFakeStreamlit(
            button_values={"wizard_config_chat_preview": True},
            session_state={"wizard_config_chat_instruction": _CONFIG_CHAT_INSTRUCTION},
        )
        monkeypatch.setattr(config_chat_module, "st", panel_fake)
        config_chat_module.render_config_chat_panel(
            on_preview=module["_preview_config_chat_instruction"],
            key_prefix="wizard",
        )
        assert any(
            kind == "info" and _CONFIG_CHAT_INSTALL_NOTICE in message
            for kind, message in panel_fake.messages
        )
        panel_preview = panel_fake.session_state.get("wizard_config_chat_preview")
        assert isinstance(panel_preview, dict)
        assert panel_preview.get("install_notice") == _CONFIG_CHAT_INSTALL_NOTICE

        wrapped = module["_preview_config_chat_change"](_CONFIG_CHAT_INSTRUCTION)
        assert wrapped["install_notice"] == preview["install_notice"]
        assert st.session_state[module["_CONFIG_CHAT_PREVIEW_KEY"]]["install_notice"] == (
            _CONFIG_CHAT_INSTALL_NOTICE
        )

        st.session_state.clear()
        st.session_state["wizard_config"] = config

        class Prompt:
            @staticmethod
            def from_messages(messages):
                return Prompt()

            def __or__(self, client):
                return client

        monkeypatch.setitem(
            sys.modules, "langchain_core.prompts", SimpleNamespace(ChatPromptTemplate=Prompt)
        )

        run_globals = module["_run_config_chat_instruction"].__globals__
        run_globals["create_llm"] = lambda _config: SimpleNamespace(
            invoke=lambda _prompt: SimpleNamespace(content="{}")
        )
        patched = validate_patch_dict({"set": {"n_simulations": 8000}, "merge": {}, "remove": []})
        run_globals["run_config_patch_chain"] = lambda **_kwargs: ConfigPatchChainResult(
            patch=patched,
            summary="Provider patch accepted.",
            risk_flags=[],
            unknown_output_keys=[],
            trace_url=None,
        )
        st.session_state[module["_CONFIG_CHAT_API_KEY_KEY"]] = "synthetic-openai"

        configured = module["_preview_config_chat_instruction"](
            _CONFIG_CHAT_MISSING_EXTRA_VERIFICATION["instruction"]
        )
        _assert_configured_provider_control(configured)
        _assert_missing_extra_vs_configured_provider_contrast(preview, configured)

        assert (
            module["_CONFIG_CHAT_INSTALL_NOTICE"] == _CONFIG_CHAT_INSTALL_NOTICE
        ), "wizard install notice must match verification record"
    finally:
        st.session_state.clear()


def test_config_chat_heuristic_boundary_pins_missing_extra_labeling() -> None:
    """Pin wizard heuristic boundary: only missing_optional_dependency adds missing_extra."""
    st.session_state.clear()
    try:
        module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
        config = get_default_config(AnalysisMode.RETURNS)
        instruction = _CONFIG_CHAT_MISSING_EXTRA_VERIFICATION["instruction"]
        with_missing = module["_heuristic_config_patch_result"](
            instruction,
            config=config,
            missing_optional_dependency=True,
        )
        without_missing = module["_heuristic_config_patch_result"](
            instruction,
            config=config,
            missing_optional_dependency=False,
        )
        assert "missing_extra" in with_missing.risk_flags
        assert "missing_extra" not in without_missing.risk_flags
        assert with_missing.patch.set.get("n_simulations") == without_missing.patch.set.get(
            "n_simulations"
        )
        assert with_missing.summary == without_missing.summary

        st.session_state["wizard_config"] = config
        st.session_state[module["_CONFIG_CHAT_API_KEY_KEY"]] = "synthetic-openai"
        globals_ = module["_build_config_chat_llm_invoke"].__globals__

        def _raise_missing_extra(*_args, **_kwargs) -> None:
            raise ModuleNotFoundError("langchain_openai")

        globals_["create_llm"] = _raise_missing_extra
        globals_["resolve_llm_provider_config"] = lambda **_kwargs: SimpleNamespace(
            provider_name="openai",
            model_name="gpt-4o-mini",
        )
        wizard_preview = module["_preview_config_chat_instruction"](instruction)
        assert wizard_preview["patch"]["set"]["n_simulations"] == 5000
        assert wizard_preview["summary"] == with_missing.summary
        assert "missing_extra" in wizard_preview["risk_flags"]
        assert wizard_preview["install_notice"] == _CONFIG_CHAT_INSTALL_NOTICE
        _assert_missing_extra_install_contract(wizard_preview)
    finally:
        st.session_state.clear()


def test_config_chat_missing_extra_legacy_readback_fails_install_contract() -> None:
    """Pre-fix (e1ec34c8) readback for the same instruction input violates the new contract."""
    observed = _CONFIG_CHAT_MISSING_EXTRA_VERIFICATION["missing_extra_observed"]
    legacy_preview = {
        "patch": {"set": {"n_simulations": observed["n_simulations"]}, "merge": {}, "remove": []},
        "summary": observed["summary"],
        "risk_flags": list(_LEGACY_MISSING_EXTRA_READBACK["risk_flags"]),
        "install_notice": _LEGACY_MISSING_EXTRA_READBACK["install_notice"],
    }
    assert _matches_legacy_missing_extra_readback(legacy_preview)
    with pytest.raises(AssertionError):
        _assert_missing_extra_install_contract(legacy_preview)


def test_config_chat_missing_extra_same_fixture_deliberate_break_readback(monkeypatch) -> None:
    """Same verification input: pre-fix heuristic readback fails; live wizard path passes."""
    instruction = _CONFIG_CHAT_MISSING_EXTRA_VERIFICATION["instruction"]
    st.session_state.clear()
    try:
        module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
        config = get_default_config(AnalysisMode.RETURNS)
        st.session_state["wizard_config"] = config
        st.session_state[module["_CONFIG_CHAT_API_KEY_KEY"]] = "synthetic-openai"

        globals_ = module["_build_config_chat_llm_invoke"].__globals__

        def _raise_missing_extra(*_args, **_kwargs) -> None:
            raise ModuleNotFoundError("langchain_openai")

        globals_["create_llm"] = _raise_missing_extra
        globals_["resolve_llm_provider_config"] = lambda **_kwargs: SimpleNamespace(
            provider_name="openai",
            model_name="gpt-4o-mini",
        )

        live_preview = module["_preview_config_chat_instruction"](instruction)
        _assert_missing_extra_install_contract(live_preview)

        pre_fix = module["_heuristic_config_patch_result"](
            instruction,
            config=config,
            missing_optional_dependency=False,
        )
        legacy_preview = _heuristic_preview_dict(pre_fix)
        assert _matches_legacy_missing_extra_readback(legacy_preview)
        with pytest.raises(AssertionError):
            _assert_missing_extra_install_contract(legacy_preview)
    finally:
        st.session_state.clear()


def test_audit_reproduces_config_chat_unscoped_override_without_network(
    monkeypatch, socket_connect_guard
):
    """Characterize the reported, unfixed gap; do not declare provider isolation."""
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
        module["_build_config_chat_llm_invoke"].__globals__["create_llm"] = lambda config: (
            captured.append(config)
        )
        st.session_state[module["_CONFIG_CHAT_API_KEY_KEY"]] = "synthetic-audit-openai"
        for provider in ("openai", "anthropic"):
            st.session_state[module["_CONFIG_CHAT_PROVIDER_KEY"]] = provider
            module["_build_config_chat_llm_invoke"]()
        assert [c.provider_name for c in captured] == ["openai", "anthropic"]
        assert [c.credentials["api_key"] for c in captured] == ["synthetic-audit-openai"] * 2
        assert socket_connect_guard[0] == []
    finally:
        st.session_state.clear()
