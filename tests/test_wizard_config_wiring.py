import runpy
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import streamlit as st

from pa_core.backend import SUPPORTED_BACKENDS
from pa_core.config import ModelConfig
from pa_core.wizard_schema import AnalysisMode, RiskMetric, get_default_config


def _load_build_yaml() -> tuple[callable, dict]:
    module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
    return module["_build_yaml_from_config"], module


class _Column:
    def __enter__(self) -> "_Column":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


class _SessionState(dict[str, Any]):
    def __getattr__(self, key: str) -> Any:
        return self[key]

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


class _RerunRequested(BaseException):
    pass


def _stale_step_2_state() -> _SessionState:
    return _SessionState(
        {
            "wizard_total_fund_capital": 777.0,
            "wizard_external_pa_capital": 7.0,
            "wizard_active_ext_capital": 8.0,
            "wizard_internal_pa_capital": 9.0,
            "wizard_w_beta_h": 0.9,
            "wizard_theta_extpa": 0.8,
            "wizard_active_share": 0.7,
        }
    )


def test_step_1_render_clamps_default_simulations_to_widget_min(monkeypatch) -> None:
    module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
    config = get_default_config(AnalysisMode.RETURNS)
    config.n_simulations = 1

    monkeypatch.setattr(module["st"], "subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr(module["st"], "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(module["st"], "info", lambda *args, **kwargs: None)

    def columns(spec: int | list[Any], *args: Any, **kwargs: Any) -> list[_Column]:
        count = spec if isinstance(spec, int) else len(spec)
        return [_Column() for _ in range(count)]

    monkeypatch.setattr(module["st"], "columns", columns)
    monkeypatch.setattr(
        module["st"],
        "selectbox",
        lambda _label, options, index=0, **_kwargs: options[index],
    )

    def number_input(label: str, *args: Any, **kwargs: Any) -> Any:
        min_value = kwargs.get("min_value")
        value = kwargs.get("value")
        if label == "Number of Simulations":
            assert min_value == 100
            assert value >= min_value
        return value

    monkeypatch.setattr(module["st"], "number_input", number_input)

    rendered = module["_render_step_1_analysis_mode"](config)

    assert rendered.n_simulations == 100


def test_loaded_config_replaces_stale_widget_state_and_survives_step_2(monkeypatch) -> None:
    module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
    widget_values: dict[str, Any] = {}

    loaded_values = {
        "wizard_total_fund_capital": 1000.0,
        "wizard_external_pa_capital": 125.0,
        "wizard_active_ext_capital": 175.0,
        "wizard_internal_pa_capital": 650.0,
        "wizard_w_beta_h": 0.35,
        "wizard_theta_extpa": 0.42,
        "wizard_active_share": 0.73,
    }
    session_state = _stale_step_2_state()
    uploaded = MagicMock()
    uploaded.getvalue.return_value = (
        module["yaml"]
        .safe_dump(
            {
                "Number of simulations": 1000,
                "Number of months": 12,
                "financing_mode": "broadcast",
                "total_fund_capital": loaded_values["wizard_total_fund_capital"],
                "external_pa_capital": loaded_values["wizard_external_pa_capital"],
                "active_ext_capital": loaded_values["wizard_active_ext_capital"],
                "internal_pa_capital": loaded_values["wizard_internal_pa_capital"],
                "w_beta_H": loaded_values["wizard_w_beta_h"],
                "w_alpha_H": 1.0 - loaded_values["wizard_w_beta_h"],
                "theta_extpa": loaded_values["wizard_theta_extpa"],
                "active_share": loaded_values["wizard_active_share"],
            }
        )
        .encode()
    )

    fake_st = MagicMock()
    fake_st.session_state = session_state
    fake_st.sidebar.button.side_effect = (
        lambda label, *args, **kwargs: label == "Load Configuration"
    )
    fake_st.sidebar.file_uploader.return_value = uploaded
    fake_st.rerun.side_effect = _RerunRequested
    fake_st.columns.side_effect = lambda spec, *args, **kwargs: [
        _Column() for _ in range(spec if isinstance(spec, int) else len(spec))
    ]
    fake_st.expander.return_value = _Column()
    fake_st.button.return_value = False

    def _record_value(label: str, *args: Any, **kwargs: Any) -> Any:
        value = kwargs.get("value")
        widget_values[label] = value
        if key := kwargs.get("key"):
            session_state[key] = value
        return value

    fake_st.number_input.side_effect = _record_value
    fake_st.slider.side_effect = _record_value
    fake_st.selectbox.side_effect = lambda label, options, index=0, **kwargs: options[index]
    globals_ = module["main"].__globals__
    monkeypatch.setitem(globals_, "st", fake_st)
    monkeypatch.setitem(globals_, "render_settings_sidebar", lambda: (None, None))
    monkeypatch.setitem(globals_, "apply_theme", lambda _path: None)

    try:
        module["main"]()
    except _RerunRequested:
        pass
    else:
        raise AssertionError("loading a configuration must request a Streamlit rerun")

    assert {key: session_state[key] for key in loaded_values} == loaded_values

    rendered = module["_render_step_2_capital"](session_state.wizard_config)

    assert rendered.total_fund_capital == 1000.0
    assert rendered.external_pa_capital == 125.0
    assert rendered.active_ext_capital == 175.0
    assert rendered.internal_pa_capital == 650.0
    assert rendered.w_beta_h == 0.35
    assert rendered.theta_extpa == 0.42
    assert rendered.active_share == 0.73
    assert widget_values["Total Fund Capital"] == 1000.0
    assert widget_values["External PA Capital [$M]"] == 125.0
    assert widget_values["Active Extension Capital [$M]"] == 175.0
    assert widget_values["Internal PA Capital [$M]"] == 650.0
    assert widget_values["Internal Beta Weight"] == 0.35
    assert widget_values["External PA Alpha Fraction"] == 0.42
    assert widget_values["Active Extension Share"] == 0.73


def test_sidebar_reset_replaces_stale_step_2_widget_state(monkeypatch) -> None:
    module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
    config = get_default_config(AnalysisMode.RETURNS)
    session_state = _stale_step_2_state()
    session_state.wizard_config = config
    session_state.wizard_step = 5

    fake_st = MagicMock()
    fake_st.session_state = session_state
    fake_st.sidebar.button.side_effect = (
        lambda label, *args, **kwargs: label == "🔄 Reset All Defaults"
    )
    fake_st.rerun.side_effect = _RerunRequested
    globals_ = module["main"].__globals__
    monkeypatch.setitem(globals_, "st", fake_st)
    monkeypatch.setitem(globals_, "render_settings_sidebar", lambda: (None, None))
    monkeypatch.setitem(globals_, "apply_theme", lambda _path: None)

    try:
        module["main"]()
    except _RerunRequested:
        pass
    else:
        raise AssertionError("resetting defaults must request a Streamlit rerun")

    reset = session_state.wizard_config
    assert session_state["wizard_total_fund_capital"] == reset.total_fund_capital
    assert session_state["wizard_external_pa_capital"] == reset.external_pa_capital
    assert session_state["wizard_active_ext_capital"] == reset.active_ext_capital
    assert session_state["wizard_internal_pa_capital"] == reset.internal_pa_capital
    assert session_state["wizard_w_beta_h"] == reset.w_beta_h
    assert session_state["wizard_theta_extpa"] == reset.theta_extpa
    assert session_state["wizard_active_share"] == reset.active_share


def test_review_reset_replaces_stale_step_2_widget_state(monkeypatch) -> None:
    module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
    config = get_default_config(AnalysisMode.RETURNS)
    session_state = _stale_step_2_state()
    session_state.wizard_config = config

    fake_st = MagicMock()
    fake_st.session_state = session_state
    fake_st.columns.side_effect = lambda spec, *args, **kwargs: [
        _Column() for _ in range(spec if isinstance(spec, int) else len(spec))
    ]
    fake_st.button.side_effect = lambda label, *args, **kwargs: label == "🔄 Reset to Defaults"
    fake_st.checkbox.return_value = True
    fake_st.rerun.side_effect = _RerunRequested
    monkeypatch.setitem(module["_render_step_5_review"].__globals__, "st", fake_st)

    try:
        module["_render_step_5_review"](config)
    except _RerunRequested:
        pass
    else:
        raise AssertionError("resetting defaults must request a Streamlit rerun")

    reset = session_state.wizard_config
    assert session_state["wizard_total_fund_capital"] == reset.total_fund_capital
    assert session_state["wizard_external_pa_capital"] == reset.external_pa_capital
    assert session_state["wizard_active_ext_capital"] == reset.active_ext_capital
    assert session_state["wizard_internal_pa_capital"] == reset.internal_pa_capital
    assert session_state["wizard_w_beta_h"] == reset.w_beta_h
    assert session_state["wizard_theta_extpa"] == reset.theta_extpa
    assert session_state["wizard_active_share"] == reset.active_share


def test_build_yaml_maps_all_fields() -> None:
    st.session_state.clear()
    try:
        build_yaml, module = _load_build_yaml()
        config = get_default_config(AnalysisMode.RETURNS)

        config.analysis_mode = AnalysisMode.CAPITAL
        config.n_simulations = 1234
        config.n_months = 17
        config.financing_mode = "broadcast"

        config.external_pa_capital = 10.0
        config.active_ext_capital = 20.0
        config.internal_pa_capital = 30.0
        config.total_fund_capital = 60.0

        st.session_state[module["_TOTAL_CAPITAL_KEY"]] = 99.0
        st.session_state[module["_EXTERNAL_CAPITAL_KEY"]] = 11.0
        st.session_state[module["_ACTIVE_CAPITAL_KEY"]] = 22.0
        st.session_state[module["_INTERNAL_CAPITAL_KEY"]] = 66.0

        config.w_beta_h = 0.11
        config.w_alpha_h = 0.89
        config.theta_extpa = 0.22
        config.active_share = 0.33

        config.mu_h = 0.01
        config.mu_e = 0.02
        config.mu_m = 0.03
        config.sigma_h = 0.04
        config.sigma_e = 0.05
        config.sigma_m = 0.06

        config.rho_idx_h = 0.1
        config.rho_idx_e = 0.2
        config.rho_idx_m = 0.3
        config.rho_h_e = 0.4
        config.rho_h_m = 0.5
        config.rho_e_m = 0.6

        config.risk_metrics = ["Return", "Risk", "terminal_ShortfallProb"]

        st.session_state["sleeve_max_te"] = 0.02
        st.session_state["sleeve_max_breach"] = 0.25
        st.session_state["sleeve_max_cvar"] = 0.05
        st.session_state["sleeve_max_shortfall"] = 0.1
        st.session_state["sleeve_constraint_scope"] = "sleeves"
        st.session_state["sleeve_validate_on_run"] = True

        st.session_state["financing_settings"] = {
            "financing_model": "simple_proxy",
            "reference_sigma": 0.02,
            "volatility_multiple": 4.0,
            "term_months": 2.5,
            "schedule_path": "ignored.csv",
        }

        yaml_dict = build_yaml(config)

        assert yaml_dict["N_SIMULATIONS"] == 1234
        assert yaml_dict["N_MONTHS"] == 17
        assert yaml_dict["analysis_mode"] == "capital"
        assert yaml_dict["financing_mode"] == "broadcast"

        assert yaml_dict["total_fund_capital"] == 99.0
        assert yaml_dict["external_pa_capital"] == 11.0
        assert yaml_dict["active_ext_capital"] == 22.0
        assert yaml_dict["internal_pa_capital"] == 66.0

        assert yaml_dict["w_beta_H"] == 0.11
        assert yaml_dict["w_alpha_H"] == 0.89
        assert yaml_dict["theta_extpa"] == 0.22
        assert yaml_dict["active_share"] == 0.33

        assert yaml_dict["mu_H"] == 0.01
        assert yaml_dict["mu_E"] == 0.02
        assert yaml_dict["mu_M"] == 0.03
        assert yaml_dict["sigma_H"] == 0.04
        assert yaml_dict["sigma_E"] == 0.05
        assert yaml_dict["sigma_M"] == 0.06

        assert yaml_dict["rho_idx_H"] == 0.1
        assert yaml_dict["rho_idx_E"] == 0.2
        assert yaml_dict["rho_idx_M"] == 0.3
        assert yaml_dict["rho_H_E"] == 0.4
        assert yaml_dict["rho_H_M"] == 0.5
        assert yaml_dict["rho_E_M"] == 0.6

        assert yaml_dict["risk_metrics"] == ["Return", "Risk", "terminal_ShortfallProb"]
        assert yaml_dict["sleeve_max_te"] == 0.02
        assert yaml_dict["sleeve_max_breach"] == 0.25
        assert yaml_dict["sleeve_max_cvar"] == 0.05
        assert yaml_dict["sleeve_max_shortfall"] == 0.1
        assert yaml_dict["sleeve_constraint_scope"] == "per_sleeve"
        assert yaml_dict["sleeve_validate_on_run"] is True
        assert yaml_dict["reference_sigma"] == 0.02
        assert yaml_dict["volatility_multiple"] == 4.0
        assert yaml_dict["financing_model"] == "simple_proxy"
        assert yaml_dict["financing_schedule_path"] is None
        assert yaml_dict["financing_term_months"] == 2.5

        model_config = ModelConfig.model_validate(yaml_dict)
        assert model_config.sleeve_max_te == 0.02
        assert model_config.sleeve_max_breach == 0.25
        assert model_config.sleeve_max_cvar == 0.05
        assert model_config.sleeve_max_shortfall == 0.1
        assert model_config.sleeve_constraint_scope == "per_sleeve"
        assert model_config.sleeve_validate_on_run is True
    finally:
        st.session_state.clear()


def test_build_yaml_includes_schedule_path() -> None:
    st.session_state.clear()
    try:
        build_yaml, _module = _load_build_yaml()
        config = get_default_config(AnalysisMode.RETURNS)

        st.session_state["financing_settings"] = {
            "financing_model": "schedule",
            "reference_sigma": 0.03,
            "volatility_multiple": 5.5,
            "term_months": 4.0,
            "schedule_path": Path("schedule.csv"),
        }

        yaml_dict = build_yaml(config)

        assert yaml_dict["reference_sigma"] == 0.03
        assert yaml_dict["volatility_multiple"] == 5.5
        assert yaml_dict["financing_model"] == "schedule"
        assert yaml_dict["financing_schedule_path"] == "schedule.csv"
        assert yaml_dict["financing_term_months"] == 4.0
    finally:
        st.session_state.clear()


def test_build_yaml_dict_alias_matches_from_config() -> None:
    st.session_state.clear()
    try:
        module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
        build_yaml = module["_build_yaml_from_config"]
        build_yaml_dict = module["_build_yaml_dict"]
        config = get_default_config(AnalysisMode.RETURNS)

        assert build_yaml_dict(config) == build_yaml(config)
    finally:
        st.session_state.clear()


def test_default_config_yaml_validates() -> None:
    st.session_state.clear()
    try:
        build_yaml, _module = _load_build_yaml()
        config = get_default_config(AnalysisMode.RETURNS)

        yaml_dict = build_yaml(config)
        model_config = ModelConfig.model_validate(yaml_dict)

        assert model_config.return_distribution == config.return_distribution
        assert model_config.return_copula == config.return_copula
        assert model_config.vol_regime == config.vol_regime
        assert model_config.covariance_shrinkage == config.covariance_shrinkage
    finally:
        st.session_state.clear()


def test_build_yaml_serializes_risk_metric_enums() -> None:
    st.session_state.clear()
    try:
        build_yaml, _module = _load_build_yaml()
        config = get_default_config(AnalysisMode.RETURNS)

        config.risk_metrics = [RiskMetric.RETURN, RiskMetric.RISK, RiskMetric.SHORTFALL_PROB]

        yaml_dict = build_yaml(config)

        assert yaml_dict["risk_metrics"] == ["Return", "Risk", "terminal_ShortfallProb"]
    finally:
        st.session_state.clear()


def test_build_yaml_includes_advanced_simulation_settings() -> None:
    st.session_state.clear()
    try:
        build_yaml, _module = _load_build_yaml()
        config = get_default_config(AnalysisMode.RETURNS)

        config.return_distribution = "student_t"
        config.return_t_df = 5.5
        config.return_copula = "t"
        config.vol_regime = "two_state"
        config.vol_regime_window = 12
        config.covariance_shrinkage = "ledoit_wolf"
        config.correlation_repair_mode = "warn_fix"
        config.correlation_repair_shrinkage = 0.2
        config.correlation_repair_max_abs_delta = 0.25
        config.backend = list(SUPPORTED_BACKENDS)[0]

        yaml_dict = build_yaml(config)

        assert yaml_dict["return_distribution"] == "student_t"
        assert yaml_dict["return_t_df"] == 5.5
        assert yaml_dict["return_copula"] == "t"
        assert yaml_dict["vol_regime"] == "two_state"
        assert yaml_dict["vol_regime_window"] == 12
        assert yaml_dict["covariance_shrinkage"] == "ledoit_wolf"
        assert yaml_dict["correlation_repair_mode"] == "warn_fix"
        assert yaml_dict["correlation_repair_shrinkage"] == 0.2
        assert yaml_dict["correlation_repair_max_abs_delta"] == 0.25
        assert yaml_dict["backend"] == config.backend

        model_config = ModelConfig.model_validate(yaml_dict)
        assert model_config.return_distribution == "student_t"
        assert model_config.return_t_df == 5.5
        assert model_config.return_copula == "t"
        assert model_config.vol_regime == "two_state"
        assert model_config.vol_regime_window == 12
        assert model_config.covariance_shrinkage == "ledoit_wolf"
        assert model_config.correlation_repair_mode == "warn_fix"
        assert model_config.correlation_repair_shrinkage == 0.2
        assert model_config.correlation_repair_max_abs_delta == 0.25
        assert model_config.backend == config.backend
    finally:
        st.session_state.clear()


def test_build_yaml_includes_regime_switching() -> None:
    st.session_state.clear()
    try:
        build_yaml, _module = _load_build_yaml()
        config = get_default_config(AnalysisMode.RETURNS)

        config.regimes = [
            {"name": "Calm", "idx_sigma_multiplier": 0.8},
            {"name": "Stressed", "idx_sigma_multiplier": 1.3},
        ]
        config.regime_transition = [[0.9, 0.1], [0.2, 0.8]]
        config.regime_start = "Calm"

        yaml_dict = build_yaml(config)

        assert yaml_dict["regimes"] == config.regimes
        assert yaml_dict["regime_transition"] == config.regime_transition
        assert yaml_dict["regime_start"] == "Calm"

        model_config = ModelConfig.model_validate(yaml_dict)
        assert model_config.regimes is not None
        assert [regime.name for regime in model_config.regimes] == ["Calm", "Stressed"]
        assert model_config.regime_transition == [[0.9, 0.1], [0.2, 0.8]]
        assert model_config.regime_start == "Calm"
    finally:
        st.session_state.clear()


def test_step_3_render_preserves_advanced_regime_and_sleeve_settings(monkeypatch) -> None:
    st.session_state.clear()
    try:
        module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
        config = get_default_config(AnalysisMode.RETURNS)
        config.risk_metrics = [RiskMetric.RETURN, RiskMetric.SHORTFALL_PROB]
        config.return_distribution = "student_t"
        config.return_t_df = 7.5
        config.return_copula = "t"
        config.vol_regime = "two_state"
        config.vol_regime_window = 12
        config.covariance_shrinkage = "ledoit_wolf"
        config.correlation_repair_mode = "warn_fix"
        config.correlation_repair_shrinkage = 0.2
        config.correlation_repair_max_abs_delta = 0.25
        expected_regimes = [
            {"name": "Calm", "idx_sigma_multiplier": 0.8},
            {"name": "Stressed", "idx_sigma_multiplier": 1.3},
        ]
        expected_transition = [[0.9, 0.1], [0.2, 0.8]]
        config.regimes = expected_regimes
        config.regime_transition = expected_transition
        config.regime_start = "Stressed"
        config.sleeve_max_te = 0.0
        config.sleeve_max_breach = 0.4
        config.sleeve_max_cvar = 0.06
        config.sleeve_max_shortfall = 0.08
        config.sleeve_constraint_scope = "per_sleeve"
        config.sleeve_validate_on_run = True

        rendered_widgets: dict[str, dict[str, Any]] = {}

        def _record_value(label: str, *args: Any, **kwargs: Any) -> Any:
            rendered_widgets[label] = kwargs
            value = kwargs.get("value")
            if key := kwargs.get("key"):
                st.session_state[key] = value
            return value

        def _selectbox(
            label: str,
            options: list[Any],
            index: int = 0,
            **kwargs: Any,
        ) -> Any:
            rendered_widgets[label] = {
                **kwargs,
                "options": list(options),
                "index": index,
            }
            value = options[index]
            if key := kwargs.get("key"):
                st.session_state[key] = value
            return value

        def _multiselect(
            label: str,
            options: list[Any],
            default: list[Any],
            **kwargs: Any,
        ) -> list[Any]:
            rendered_widgets[label] = {
                **kwargs,
                "options": list(options),
                "default": list(default),
            }
            return list(default)

        def _checkbox(label: str, *args: Any, **kwargs: Any) -> bool:
            rendered_widgets[label] = kwargs
            value = bool(kwargs.get("value", False))
            if key := kwargs.get("key"):
                st.session_state[key] = value
            return value

        def _text_area(label: str, *args: Any, **kwargs: Any) -> str:
            rendered_widgets[label] = kwargs
            value = str(kwargs.get("value", ""))
            if key := kwargs.get("key"):
                st.session_state[key] = value
            return value

        monkeypatch.setattr(
            module["st"], "columns", lambda count: [_Column() for _ in range(count)]
        )
        monkeypatch.setattr(module["st"], "expander", lambda *args, **kwargs: _Column())
        monkeypatch.setattr(module["st"], "number_input", _record_value)
        monkeypatch.setattr(module["st"], "slider", _record_value)
        monkeypatch.setattr(module["st"], "selectbox", _selectbox)
        monkeypatch.setattr(module["st"], "multiselect", _multiselect)
        monkeypatch.setattr(module["st"], "checkbox", _checkbox)
        monkeypatch.setattr(module["st"], "text_area", _text_area)
        monkeypatch.setattr(module["st"], "button", lambda *args, **kwargs: False)
        for method in ("subheader", "markdown", "write"):
            monkeypatch.setattr(module["st"], method, lambda *args, **kwargs: None)

        rendered = module["_render_step_3_returns_risk"](config)

        assert rendered is config
        assert config.risk_metrics == ["Return", "terminal_ShortfallProb"]
        assert config.return_distribution == "student_t"
        assert config.return_t_df == 7.5
        assert config.return_copula == "t"
        assert config.vol_regime == "two_state"
        assert config.vol_regime_window == 12
        assert config.covariance_shrinkage == "ledoit_wolf"
        assert config.correlation_repair_mode == "warn_fix"
        assert config.correlation_repair_shrinkage == 0.2
        assert config.correlation_repair_max_abs_delta == 0.25
        assert config.regime_start == "Stressed"
        assert config.regimes == expected_regimes
        assert config.regime_transition == expected_transition
        assert config.sleeve_max_te == 0.0
        assert config.sleeve_max_breach == 0.4
        assert config.sleeve_max_cvar == 0.06
        assert config.sleeve_max_shortfall == 0.08
        assert config.sleeve_constraint_scope == "per_sleeve"
        assert config.sleeve_validate_on_run is True

        assert rendered_widgets["Select Risk Metrics"]["options"] == list(RiskMetric)
        assert rendered_widgets["Select Risk Metrics"]["default"] == [
            RiskMetric.RETURN,
            RiskMetric.SHORTFALL_PROB,
        ]
        assert rendered_widgets["Return distribution"]["options"] == ["normal", "student_t"]
        assert rendered_widgets["Return distribution"]["index"] == 1
        assert rendered_widgets["Return copula"]["options"] == ["gaussian", "t"]
        assert rendered_widgets["Return copula"]["index"] == 1
        assert rendered_widgets["Student-t degrees of freedom"]["disabled"] is False
        assert rendered_widgets["Student-t degrees of freedom"]["value"] == 7.5
        assert rendered_widgets["Volatility regime"]["index"] == 1
        assert rendered_widgets["Volatility regime window (months)"]["value"] == 12
        assert rendered_widgets["Covariance shrinkage"]["index"] == 1
        assert rendered_widgets["Correlation repair mode"]["index"] == 1
        assert rendered_widgets["Correlation repair shrinkage"]["value"] == 0.2
        assert rendered_widgets["Enforce max correlation repair delta"]["value"] is True
        assert rendered_widgets["Correlation repair max abs delta"]["value"] == 0.25
        assert module["yaml"].safe_load(rendered_widgets["Regimes (YAML/JSON)"]["value"]) == (
            expected_regimes
        )
        assert (
            module["yaml"].safe_load(
                rendered_widgets["Regime transition matrix (YAML/JSON)"]["value"]
            )
            == expected_transition
        )
        assert rendered_widgets["Starting regime (optional)"]["options"] == [
            "(auto)",
            "Calm",
            "Stressed",
        ]
        assert rendered_widgets["Starting regime (optional)"]["index"] == 2
        assert rendered_widgets["Max Tracking Error"]["value"] == 0.0
        assert rendered_widgets["Max Breach Probability"]["value"] == 0.4
        assert rendered_widgets["Max monthly_CVaR"]["value"] == 0.06
        assert rendered_widgets["Max Terminal Shortfall Probability"]["value"] == 0.08
        assert rendered_widgets["Constraint Scope"]["options"] == ["sleeves", "total", "both"]
        assert rendered_widgets["Constraint Scope"]["index"] == 0
        assert rendered_widgets["Validate constraints on run"]["value"] is True
    finally:
        st.session_state.clear()
