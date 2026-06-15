from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any


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
    results = [SimpleNamespace(is_valid=True, severity="info")]

    def fake_validate(payload: dict[str, Any], received_settings: dict[str, Any]) -> list[Any]:
        calls["payload"] = payload
        calls["settings"] = received_settings
        return results

    def fake_display(received_results: list[Any], title: str) -> None:
        calls["display"] = (received_results, title)

    monkeypatch.setattr(validation_ui, "create_validation_sidebar", lambda: settings)
    monkeypatch.setattr(validation_ui, "validate_scenario_config", fake_validate)
    monkeypatch.setattr(validation_ui, "display_validation_results", fake_display)

    config = SimpleNamespace(
        rho_idx_h=0.1,
        rho_idx_e=0.2,
        rho_idx_m=0.3,
        rho_h_e=0.4,
        rho_h_m=0.5,
        rho_e_m=0.6,
        external_pa_capital=10.0,
        active_ext_capital=20.0,
        internal_pa_capital=30.0,
        total_fund_capital=100.0,
        reference_sigma=0.01,
        volatility_multiple=3.0,
        financing_model="simple_proxy",
        financing_term_months=12.0,
        financing_schedule_path=None,
        n_simulations=5000,
    )

    assert module["_render_validation_panel"](config) == results
    assert calls["settings"] == settings
    assert calls["payload"]["rho_idx_H"] == 0.1
    assert calls["payload"]["rho_E_M"] == 0.6
    assert calls["payload"]["N_SIMULATIONS"] == 5000
    assert calls["display"] == (results, "Wizard Validation")


def test_scenario_wizard_no_unreachable_double_return_after_correlations() -> None:
    source = Path("dashboard/pages/3_Scenario_Wizard.py").read_text()

    assert "_render_validation_panel(config)\n    return config\n\n    return config" not in source


def test_results_page_uses_modern_rerun_api() -> None:
    source = Path("dashboard/pages/4_Results.py").read_text()

    assert "st.experimental_rerun" not in source
    assert "st.rerun()" in source
