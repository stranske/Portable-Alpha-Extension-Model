from __future__ import annotations

from dashboard.utils import (
    DASHBOARD_ASSET_LIBRARY_KEY,
    DASHBOARD_PORTFOLIO_KEY,
    DASHBOARD_RESULT_KEY,
    DASHBOARD_SCENARIO_KEY,
    get_dashboard_asset_library_yaml,
    get_dashboard_capital_defaults,
    get_dashboard_result_path,
    store_dashboard_asset_library,
    store_dashboard_portfolio,
    store_dashboard_result,
    store_dashboard_scenario,
)
from pa_core.wizard_schema import AnalysisMode, get_default_config


def test_wizard_result_path_defaults_results_page_to_latest_session_output() -> None:
    state: dict[str, object] = {}
    config = get_default_config(AnalysisMode.RETURNS)
    config.total_fund_capital = 750.0
    config.external_pa_capital = 125.0
    config.active_ext_capital = 175.0
    config.internal_pa_capital = 450.0

    scenario = store_dashboard_scenario(
        state,
        config,
        source="scenario_wizard",
        yaml_data={"total_fund_capital": 750.0},
    )
    result = store_dashboard_result(
        state,
        "ScenarioRun.xlsx",
        source="scenario_wizard",
        scenario=scenario,
        seed=123,
    )

    assert state[DASHBOARD_SCENARIO_KEY] == scenario
    assert state[DASHBOARD_RESULT_KEY] == result
    assert get_dashboard_result_path(state, "Outputs.xlsx") == "ScenarioRun.xlsx"
    assert result["seed"] == 123


def test_stress_lab_capital_defaults_reuse_latest_scenario_values() -> None:
    state: dict[str, object] = {}
    config = get_default_config(AnalysisMode.RETURNS)
    config.total_fund_capital = 900.0
    config.external_pa_capital = 100.0
    config.active_ext_capital = 250.0
    config.internal_pa_capital = 550.0

    store_dashboard_scenario(state, config, source="scenario_wizard")

    assert get_dashboard_capital_defaults(state) == {
        "total_fund_capital": 900.0,
        "external_pa_capital": 100.0,
        "active_ext_capital": 250.0,
        "internal_pa_capital": 550.0,
    }


def test_capital_defaults_can_fall_back_to_result_embedded_scenario() -> None:
    state: dict[str, object] = {}
    scenario = {
        "values": {
            "total_fund_capital": "600",
            "external_pa_capital": "120",
            "active_ext_capital": "80",
            "internal_pa_capital": "400",
        }
    }

    store_dashboard_result(
        state,
        "Results.xlsx",
        source="scenario_wizard",
        scenario=scenario,
    )
    state.pop(DASHBOARD_SCENARIO_KEY, None)

    assert get_dashboard_capital_defaults(state)["total_fund_capital"] == 600.0
    assert get_dashboard_capital_defaults(state)["internal_pa_capital"] == 400.0


def test_asset_library_and_portfolio_yaml_are_available_without_download_roundtrip() -> None:
    state: dict[str, object] = {}

    asset_payload = store_dashboard_asset_library(
        state,
        "assets:\n  - id: Index\n",
        source_name="uploaded.csv",
    )
    portfolio_payload = store_dashboard_portfolio(
        state,
        "portfolios:\n  - id: portfolio1\n",
        source_name="session asset library",
    )

    assert state[DASHBOARD_ASSET_LIBRARY_KEY] == asset_payload
    assert state[DASHBOARD_PORTFOLIO_KEY] == portfolio_payload
    assert get_dashboard_asset_library_yaml(state) == "assets:\n  - id: Index\n"
