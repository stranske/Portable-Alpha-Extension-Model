from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import streamlit as st

from pa_core.wizard_schema import AnalysisMode, get_default_config


def _wizard_module() -> dict[str, Any]:
    return runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")


def test_wizard_validation_panel_uses_existing_validation_ui(monkeypatch) -> None:
    module = _wizard_module()
    validation_ui = module["validation_ui"]
    calls: dict[str, Any] = {}

    settings = {
        "validate_on_change": True,
        "show_details": False,
        "show_warnings": True,
    }
    results = [object()]

    def fake_validate(payload: dict[str, Any], received_settings: dict[str, Any]) -> list[Any]:
        calls["payload"] = payload
        calls["settings"] = received_settings
        return results

    def fake_display(received_results: list[Any], title: str) -> None:
        calls["display"] = (received_results, title)

    monkeypatch.setattr(validation_ui, "create_validation_sidebar", lambda: settings)
    monkeypatch.setattr(validation_ui, "validate_scenario_config", fake_validate)
    monkeypatch.setattr(validation_ui, "display_validation_results", fake_display)

    st.session_state.clear()
    try:
        config = get_default_config(AnalysisMode.RETURNS)
        config.rho_idx_h = 0.1
        config.rho_idx_e = 0.2
        config.rho_idx_m = 0.3
        config.rho_h_e = 0.4
        config.rho_h_m = 0.5
        config.rho_e_m = 0.6
        config.external_pa_capital = 10.0
        config.active_ext_capital = 20.0
        config.internal_pa_capital = 30.0
        config.total_fund_capital = 100.0
        config.n_simulations = 5000
        st.session_state["financing_settings"] = {
            "financing_model": "schedule",
            "reference_sigma": 0.02,
            "volatility_multiple": 4.0,
            "term_months": 12.0,
            "schedule_path": "margin.csv",
        }

        assert module["_render_validation_panel"](config) == results
        assert calls["settings"] == settings
        assert calls["payload"]["rho_idx_H"] == 0.1
        assert calls["payload"]["rho_E_M"] == 0.6
        assert calls["payload"]["reference_sigma"] == 0.02
        assert calls["payload"]["volatility_multiple"] == 4.0
        assert calls["payload"]["financing_model"] == "schedule"
        assert calls["payload"]["financing_term_months"] == 12.0
        assert calls["payload"]["financing_schedule_path"] == "margin.csv"
        assert calls["payload"]["N_SIMULATIONS"] == 5000
        assert calls["display"] == (results, "Wizard Validation")
    finally:
        st.session_state.clear()


def test_scenario_wizard_no_unreachable_double_return_after_correlations() -> None:
    source = Path("dashboard/pages/3_Scenario_Wizard.py").read_text()

    assert "_render_validation_panel(config)\n    return config\n\n    return config" not in source


def test_results_page_uses_modern_rerun_api() -> None:
    source = Path("dashboard/pages/4_Results.py").read_text()

    assert "st.experimental_rerun" not in source
    assert "st.rerun()" in source
