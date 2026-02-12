"""5-Step Wizard for guided scenario creation with validation."""

from __future__ import annotations

import math
import os
import re
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

import pandas as pd
import streamlit as st
import yaml

from dashboard import cli as dashboard_cli
from dashboard.app import _DEF_THEME, _DEF_XLSX, apply_theme
from dashboard.components.config_chat import render_config_chat_panel
from dashboard.components.llm_settings import (
    default_api_key,
    resolve_api_key_input,
    resolve_llm_provider_config,
)
from dashboard.glossary import tooltip
from dashboard.utils import run_sleeve_frontier, run_sleeve_suggestions
from pa_core import cli as pa_cli
from pa_core.backend import SUPPORTED_BACKENDS
from pa_core.config import load_config
from pa_core.data import load_index_returns
from pa_core.llm import ConfigPatchChainResult, create_llm, run_config_patch_chain
from pa_core.llm.config_chat import (
    apply_config_chat_preview as apply_config_chat_preview_core,
)
from pa_core.llm.config_chat import (
    snapshot_wizard_session_state,
)
from pa_core.llm.config_patch import (
    ALLOWED_WIZARD_FIELDS,
    empty_patch,
    round_trip_validate_config,
    validate_patch_dict,
)
from pa_core.llm.config_patch import (
    apply_patch as apply_config_patch,
)
from pa_core.llm.config_patch import (
    diff_config as diff_wizard_config,
)
from pa_core.validators import calculate_margin_requirement, load_margin_schedule
from pa_core.viz import frontier as frontier_viz
from pa_core.wizard.session_state import restore_wizard_session_snapshot
from pa_core.wizard_schema import (
    AnalysisMode,
    DefaultConfigView,
    RiskMetric,
    get_default_config,
    make_view_from_model,
)

_TOTAL_CAPITAL_KEY = "wizard_total_fund_capital"
_EXTERNAL_CAPITAL_KEY = "wizard_external_pa_capital"
_ACTIVE_CAPITAL_KEY = "wizard_active_ext_capital"
_INTERNAL_CAPITAL_KEY = "wizard_internal_pa_capital"
_W_BETA_KEY = "wizard_w_beta_h"
_THETA_EXTPA_KEY = "wizard_theta_extpa"
_ACTIVE_SHARE_KEY = "wizard_active_share"
_REGIME_ENABLED_KEY = "wizard_regime_enabled"
_REGIMES_KEY = "wizard_regimes_yaml"
_REGIME_TRANSITION_KEY = "wizard_regime_transition_yaml"
_REGIME_START_KEY = "wizard_regime_start"
_CONFIG_CHAT_HISTORY_KEY = "wizard_config_chat_history"
_CONFIG_CHAT_PROVIDER_KEY = "wizard_config_chat_provider"
_CONFIG_CHAT_MODEL_KEY = "wizard_config_chat_model"
_CONFIG_CHAT_API_KEY_KEY = "wizard_config_chat_api_key"
_CONFIG_CHAT_PREVIEW_KEY = "wizard_config_chat_preview"


def _config_chat_history() -> list[dict[str, Any]]:
    history = st.session_state.get(_CONFIG_CHAT_HISTORY_KEY)
    if not isinstance(history, list):
        history = []
        st.session_state[_CONFIG_CHAT_HISTORY_KEY] = history
    return history


def _normalize_config_chat_value(value: Any) -> Any:
    if isinstance(value, (AnalysisMode, RiskMetric)):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _normalize_config_chat_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_normalize_config_chat_value(child) for child in value]
    if isinstance(value, tuple):
        return [_normalize_config_chat_value(child) for child in value]
    return value


def _stable_unique_text_items(values: Any) -> list[str]:
    """Normalize arbitrary values into a deterministic, de-duplicated string list."""

    if values is None:
        return []
    if isinstance(values, str):
        raw_values: list[Any] = [values]
    elif isinstance(values, list):
        raw_values = values
    else:
        raw_values = [values]

    normalized: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def _config_chat_snapshot(config: DefaultConfigView) -> dict[str, Any]:
    return {
        key: _normalize_config_chat_value(getattr(config, key, None))
        for key in ALLOWED_WIZARD_FIELDS
    }


def _heuristic_config_patch_result(
    instruction: str,
    *,
    config: DefaultConfigView,
    rejected_patch_keys: list[str] | None = None,
    rejected_patch_paths: list[str] | None = None,
) -> ConfigPatchChainResult:
    text = instruction.strip().lower()
    set_ops: dict[str, Any] = {}
    risk_flags: list[str] = []

    simulations_match = re.search(r"(?:simulations?|n_simulations)\D+(\d+)", text)
    if simulations_match:
        set_ops["n_simulations"] = int(simulations_match.group(1))

    breach_to_match = re.search(
        r"(?:breach tolerance|max breach|breach probability)\D+to\D*([0-9]*\.?[0-9]+)",
        text,
    )
    if breach_to_match:
        set_ops["sleeve_max_breach"] = float(breach_to_match.group(1))
    elif "breach tolerance" in text and any(
        token in text for token in ("reduce", "lower", "decrease")
    ):
        current = getattr(config, "sleeve_max_breach", None)
        if isinstance(current, (int, float)):
            set_ops["sleeve_max_breach"] = max(0.0, float(current) * 0.8)
            risk_flags.append("heuristic_inferred_breach_target")

    patch = validate_patch_dict({"set": set_ops}) if set_ops else empty_patch()
    summary = "No recognized config changes in instruction."
    if set_ops:
        summary = "Applied heuristic instruction parsing."
    else:
        risk_flags.append("heuristic_noop")
    return ConfigPatchChainResult(
        patch=patch,
        summary=summary,
        risk_flags=risk_flags,
        unknown_output_keys=[],
        rejected_patch_keys=list(rejected_patch_keys or []),
        rejected_patch_paths=list(rejected_patch_paths or []),
        trace_url=None,
    )


def _build_config_chat_llm_invoke() -> tuple[Callable[[str], str] | None, str | None, str | None]:
    provider = str(st.session_state.get(_CONFIG_CHAT_PROVIDER_KEY, "openai")).strip().lower()
    model = str(st.session_state.get(_CONFIG_CHAT_MODEL_KEY, "")).strip() or None
    raw_key = st.session_state.get(_CONFIG_CHAT_API_KEY_KEY)
    resolved_key = resolve_api_key_input(raw_key) or default_api_key(provider)
    if not resolved_key:
        return None, provider, model

    try:
        provider_config = resolve_llm_provider_config(
            provider=provider,
            model=model,
            api_key=resolved_key,
        )
        llm = create_llm(provider_config)
        from langchain_core.prompts import ChatPromptTemplate
    except Exception:
        return None, provider, model

    def _invoke(prompt: str) -> str:
        chain = ChatPromptTemplate.from_messages([("system", "{prompt}")]) | llm
        response = chain.invoke({"prompt": prompt})
        content = getattr(response, "content", None)
        return str(content or response)

    return _invoke, provider_config.provider_name, provider_config.model_name


def _run_config_chat_instruction(
    instruction: str,
    *,
    config: DefaultConfigView,
) -> ConfigPatchChainResult:
    invoke_llm, provider, model = _build_config_chat_llm_invoke()
    if invoke_llm is None:
        return _heuristic_config_patch_result(instruction, config=config)

    try:
        return run_config_patch_chain(
            current_config=_config_chat_snapshot(config),
            instruction=instruction,
            invoke_llm=invoke_llm,
            provider_name=provider,
            model_name=model,
        )
    except Exception as exc:
        rejected_patch_keys: list[str] = []
        rejected_patch_paths: list[str] = []
        unknown_keys = getattr(exc, "unknown_keys", None)
        if isinstance(unknown_keys, list):
            rejected_patch_keys = [str(key) for key in unknown_keys if str(key)]
        unknown_paths = getattr(exc, "unknown_paths", None)
        if isinstance(unknown_paths, list):
            rejected_patch_paths = [str(path) for path in unknown_paths if str(path)]

        fallback = _heuristic_config_patch_result(
            instruction,
            config=config,
            rejected_patch_keys=rejected_patch_keys,
            rejected_patch_paths=rejected_patch_paths,
        )
        has_unknown_patch_fields = bool(rejected_patch_paths)
        if has_unknown_patch_fields and "rejected_unknown_patch_fields" not in fallback.risk_flags:
            fallback.risk_flags.append("rejected_unknown_patch_fields")
        if "llm_fallback_heuristic" not in fallback.risk_flags:
            fallback.risk_flags.append("llm_fallback_heuristic")
        return fallback


def _preview_config_chat_instruction(instruction: str) -> dict[str, Any]:
    config = st.session_state.get("wizard_config")
    if not isinstance(config, DefaultConfigView):
        raise ValueError("wizard_config is not initialized")

    before_config = deepcopy(config)
    result = _run_config_chat_instruction(instruction, config=config)
    after_config = apply_config_patch(config, result.patch)
    unified_diff, before_text, after_text = diff_wizard_config(before_config, after_config)
    validation_result = round_trip_validate_config(
        after_config,
        build_yaml_from_config=_build_yaml_from_config,
    )

    risk_flags = _stable_unique_text_items(result.risk_flags)
    unknown_output_keys = _stable_unique_text_items(result.unknown_output_keys)
    rejected_patch_keys = _stable_unique_text_items(result.rejected_patch_keys)
    rejected_patch_paths = _stable_unique_text_items(result.rejected_patch_paths)
    if unknown_output_keys and "stripped_unknown_output_keys" not in risk_flags:
        risk_flags.append("stripped_unknown_output_keys")
    if rejected_patch_paths and "rejected_unknown_patch_fields" not in risk_flags:
        risk_flags.append("rejected_unknown_patch_fields")
    if not validation_result.is_valid and "round_trip_validation_failed" not in risk_flags:
        risk_flags.append("round_trip_validation_failed")

    return {
        "patch": {
            "set": dict(result.patch.set),
            "merge": dict(result.patch.merge),
            "remove": list(result.patch.remove),
        },
        "summary": result.summary,
        "risk_flags": risk_flags,
        "unknown_output_keys": unknown_output_keys,
        "rejected_patch_keys": rejected_patch_keys,
        "rejected_patch_paths": rejected_patch_paths,
        "trace_url": result.trace_url,
        "validation_errors": list(validation_result.errors),
        "unified_diff": unified_diff,
        "before_text": before_text,
        "after_text": after_text,
    }


def _preview_config_chat_change(instruction: str) -> dict[str, Any]:
    """Backward-compatible wrapper used by legacy tests/callers."""
    preview = dict(_preview_config_chat_instruction(instruction))
    st.session_state[_CONFIG_CHAT_PREVIEW_KEY] = preview
    return preview


def _config_chat_state_snapshot(config: DefaultConfigView) -> dict[str, Any]:
    return snapshot_wizard_session_state(config, session_state=st.session_state)


def _apply_config_chat_preview(preview: Mapping[str, Any], validate: bool) -> tuple[bool, str]:
    config = st.session_state.get("wizard_config")
    if not isinstance(config, DefaultConfigView):
        return False, "wizard_config is not initialized."

    snapshot = _config_chat_state_snapshot(config)
    action = "Apply+Validate" if validate else "Apply"
    ok, message = apply_config_chat_preview_core(
        preview,
        action=action,
        session_state=st.session_state,
        build_yaml_from_config=_build_yaml_from_config,
        wizard_config_key="wizard_config",
        validate_config=round_trip_validate_config,
    )
    if not ok:
        return ok, message

    _config_chat_history().append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **snapshot,
        }
    )
    return ok, message


def _revert_last_config_chat_change() -> tuple[bool, str]:
    history = _config_chat_history()
    if not history:
        return False, "No applied config-chat changes to revert."

    last = history.pop()
    if not restore_wizard_session_snapshot(last, session_state=st.session_state):
        return False, "Cannot revert because stored snapshot is invalid."

    return True, "Reverted to previous config snapshot."


def _apply_preview_patch(validate_first: bool) -> tuple[bool, str]:
    """Backward-compatible wrapper that applies the stored preview payload."""
    preview = st.session_state.get(_CONFIG_CHAT_PREVIEW_KEY)
    if not isinstance(preview, Mapping):
        return False, "No preview available to apply."
    return _apply_config_chat_preview(preview, validate_first)


def _revert_config_chat_change() -> tuple[bool, str]:
    """Backward-compatible wrapper used by legacy tests/callers."""
    return _revert_last_config_chat_change()


def _render_config_chat_sidebar() -> None:
    render_config_chat_panel(
        on_preview=_preview_config_chat_instruction,
        on_apply=_apply_config_chat_preview,
        on_revert=_revert_last_config_chat_change,
        key_prefix="wizard",
    )


def _normalize_risk_metric_defaults(metrics: list[Any]) -> list[RiskMetric]:
    """Coerce serialized metrics into RiskMetric enums for Streamlit defaults."""
    normalized: list[RiskMetric] = []
    for metric in metrics:
        if isinstance(metric, RiskMetric):
            normalized.append(metric)
            continue
        try:
            normalized.append(RiskMetric(str(metric)))
        except ValueError:
            continue
    return normalized or list(RiskMetric)


def _serialize_risk_metrics(metrics: list[Any]) -> list[str]:
    """Serialize RiskMetric enums back into ModelConfig-friendly strings."""
    serialized: list[str] = []
    for metric in metrics:
        if isinstance(metric, RiskMetric):
            serialized.append(metric.value)
        else:
            serialized.append(str(metric))
    return serialized


def _serialize_regimes(regimes: list[Any]) -> list[dict[str, Any]]:
    """Serialize regime configs into YAML-friendly dicts."""
    serialized: list[dict[str, Any]] = []
    for regime in regimes:
        if hasattr(regime, "model_dump"):
            serialized.append(regime.model_dump())
        elif isinstance(regime, dict):
            serialized.append(regime)
        else:
            raise ValueError("Regime entries must be dicts or model instances.")
    return serialized


def _parse_yaml_or_json(raw: str, label: str) -> Any:
    """Parse YAML/JSON text and raise a user-friendly error on failure."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"{label} must be valid YAML/JSON. {exc}") from exc


def _validate_regime_inputs(
    regimes: Any, transition: Any
) -> tuple[list[dict[str, Any]], list[list[float]], list[str]]:
    """Validate parsed regime inputs and return regime names."""
    normalized_regimes: list[dict[str, Any]] = []
    if isinstance(regimes, dict):
        if not regimes:
            raise ValueError("Regimes must be a non-empty YAML/JSON list or mapping.")
        for name, regime in regimes.items():
            name_str = str(name).strip()
            if not name_str:
                raise ValueError("Regime name keys must be non-empty.")
            if not isinstance(regime, dict):
                raise ValueError(f"Regime '{name}' must be a mapping of fields.")
            if "name" in regime:
                declared_name = str(regime["name"]).strip()
                if not declared_name:
                    raise ValueError(f"Regime '{name_str}' has an empty name field.")
                if declared_name != name_str:
                    raise ValueError(
                        f"Regime name '{declared_name}' must match mapping key '{name_str}'."
                    )
            regime_dict = dict(regime)
            regime_dict["name"] = name_str
            normalized_regimes.append(regime_dict)
    elif isinstance(regimes, list) and regimes:
        for idx, regime in enumerate(regimes, start=1):
            if not isinstance(regime, dict):
                raise ValueError(f"Regime #{idx} must be a mapping with a name field.")
            name = regime.get("name")
            if name is None:
                raise ValueError(f"Regime #{idx} is missing a name.")
            name_str = str(name).strip()
            if not name_str:
                raise ValueError(f"Regime #{idx} is missing a name.")
            regime_dict = dict(regime)
            regime_dict["name"] = name_str
            normalized_regimes.append(regime_dict)
    else:
        raise ValueError("Regimes must be a non-empty YAML/JSON list or mapping.")

    regime_names = [regime["name"] for regime in normalized_regimes]

    if len(set(regime_names)) != len(regime_names):
        raise ValueError("Regime names must be unique.")

    if not isinstance(transition, (list, tuple)) or not transition:
        raise ValueError("Transition matrix must be a non-empty list of lists.")

    normalized_transition: list[list[float]] = []
    for row in transition:
        if isinstance(row, tuple):
            row = list(row)
        if not isinstance(row, list):
            raise ValueError("Transition matrix must be a non-empty list of lists.")
        normalized_transition.append(row)

    if len(normalized_transition) != len(regime_names) or any(
        len(row) != len(regime_names) for row in normalized_transition
    ):
        raise ValueError("Transition matrix must be square and match the number of regimes.")

    coerced_transition: list[list[float]] = []
    for row_idx, row in enumerate(normalized_transition, start=1):
        if any(not isinstance(value, (int, float)) for value in row):
            raise ValueError(f"Transition matrix row {row_idx} must contain numeric values.")
        coerced_row = [float(value) for value in row]
        if any(value < 0 or value > 1 for value in coerced_row):
            raise ValueError(f"Transition matrix row {row_idx} values must be between 0 and 1.")
        if not math.isclose(sum(coerced_row), 1.0, abs_tol=1e-6):
            raise ValueError(f"Transition matrix row {row_idx} must sum to 1.0.")
        coerced_transition.append(coerced_row)

    return normalized_regimes, coerced_transition, regime_names


def _normalize_sleeve_constraint_scope(scope: str | None) -> str:
    if scope in ("sleeves", "per_sleeve"):
        return "per_sleeve"
    return "total"


def _build_yaml_from_config(config: DefaultConfigView) -> Dict[str, Any]:
    """Construct a YAML-compatible dict for ModelConfig from the wizard state.

    Includes optional Financing & Margin settings stored in session state.

    Args:
        config: DefaultConfigView object with all required configuration attributes.
    """
    ss = st.session_state
    fs = ss.get("financing_settings", {})

    # Direct attribute access - all attributes are guaranteed to exist on DefaultConfigView
    analysis_mode = config.analysis_mode
    n_simulations = int(config.n_simulations)
    n_months = int(config.n_months)
    financing_mode = config.financing_mode

    # Prefer session-state overrides so applied suggestions persist across steps.
    total_capital = float(ss.get(_TOTAL_CAPITAL_KEY, config.total_fund_capital))
    external_pa_capital = float(ss.get(_EXTERNAL_CAPITAL_KEY, config.external_pa_capital))
    active_ext_capital = float(ss.get(_ACTIVE_CAPITAL_KEY, config.active_ext_capital))
    internal_pa_capital = float(ss.get(_INTERNAL_CAPITAL_KEY, config.internal_pa_capital))

    w_beta_h = float(config.w_beta_h)
    w_alpha_h = float(config.w_alpha_h)
    theta_extpa = float(config.theta_extpa)
    active_share = float(config.active_share)

    mu_h = float(config.mu_h)
    mu_e = float(config.mu_e)
    mu_m = float(config.mu_m)
    sigma_h = float(config.sigma_h)
    sigma_e = float(config.sigma_e)
    sigma_m = float(config.sigma_m)

    rho_idx_h = float(config.rho_idx_h)
    rho_idx_e = float(config.rho_idx_e)
    rho_idx_m = float(config.rho_idx_m)
    rho_h_e = float(config.rho_h_e)
    rho_h_m = float(config.rho_h_m)
    rho_e_m = float(config.rho_e_m)

    risk_metrics = _serialize_risk_metrics(config.risk_metrics)
    sleeve_max_te = ss.get("sleeve_max_te", config.sleeve_max_te)
    sleeve_max_breach = ss.get("sleeve_max_breach", config.sleeve_max_breach)
    sleeve_max_cvar = ss.get("sleeve_max_cvar", config.sleeve_max_cvar)
    sleeve_max_shortfall = ss.get("sleeve_max_shortfall", config.sleeve_max_shortfall)
    sleeve_constraint_scope = _normalize_sleeve_constraint_scope(
        ss.get("sleeve_constraint_scope", config.sleeve_constraint_scope)
    )
    sleeve_validate_on_run = bool(ss.get("sleeve_validate_on_run", config.sleeve_validate_on_run))

    return_distribution = str(config.return_distribution)
    return_t_df = float(config.return_t_df)
    return_copula = str(config.return_copula)
    vol_regime = str(config.vol_regime)
    vol_regime_window = int(config.vol_regime_window)
    covariance_shrinkage = str(config.covariance_shrinkage)
    correlation_repair_mode = str(config.correlation_repair_mode)
    correlation_repair_shrinkage = float(config.correlation_repair_shrinkage)
    correlation_repair_max_abs_delta = config.correlation_repair_max_abs_delta
    backend = str(config.backend)
    regimes = config.regimes
    regime_transition = config.regime_transition
    regime_start = config.regime_start

    fm = fs.get("financing_model", "simple_proxy")
    ref_sigma = float(fs.get("reference_sigma", 0.01))
    vol_mult = float(fs.get("volatility_multiple", 3.0))
    term_m = float(fs.get("term_months", 1.0))
    sched_path = fs.get("schedule_path")

    yaml_dict: Dict[str, Any] = {
        "N_SIMULATIONS": n_simulations,
        "N_MONTHS": n_months,
        "analysis_mode": (
            analysis_mode.value if hasattr(analysis_mode, "value") else str(analysis_mode)
        ),
        "financing_mode": financing_mode,
        "total_fund_capital": total_capital,
        "external_pa_capital": external_pa_capital,
        "active_ext_capital": active_ext_capital,
        "internal_pa_capital": internal_pa_capital,
        "w_beta_H": w_beta_h,
        "w_alpha_H": w_alpha_h,
        "theta_extpa": theta_extpa,
        "active_share": active_share,
        "mu_H": mu_h,
        "mu_E": mu_e,
        "mu_M": mu_m,
        "sigma_H": sigma_h,
        "sigma_E": sigma_e,
        "sigma_M": sigma_m,
        "rho_idx_H": rho_idx_h,
        "rho_idx_E": rho_idx_e,
        "rho_idx_M": rho_idx_m,
        "rho_H_E": rho_h_e,
        "rho_H_M": rho_h_m,
        "rho_E_M": rho_e_m,
        "risk_metrics": risk_metrics,
        "sleeve_max_te": sleeve_max_te,
        "sleeve_max_breach": sleeve_max_breach,
        "sleeve_max_cvar": sleeve_max_cvar,
        "sleeve_max_shortfall": sleeve_max_shortfall,
        "sleeve_constraint_scope": sleeve_constraint_scope,
        "sleeve_validate_on_run": sleeve_validate_on_run,
        "return_distribution": return_distribution,
        "return_t_df": return_t_df,
        "return_copula": return_copula,
        "vol_regime": vol_regime,
        "vol_regime_window": vol_regime_window,
        "covariance_shrinkage": covariance_shrinkage,
        "correlation_repair_mode": correlation_repair_mode,
        "correlation_repair_shrinkage": correlation_repair_shrinkage,
        "correlation_repair_max_abs_delta": correlation_repair_max_abs_delta,
        "backend": backend,
        "reference_sigma": ref_sigma,
        "volatility_multiple": vol_mult,
        "financing_model": fm,
        "financing_schedule_path": (str(sched_path) if (fm == "schedule" and sched_path) else None),
        "financing_term_months": term_m,
    }

    if regimes is not None:
        if regime_transition is None:
            raise ValueError("regime_transition is required when regimes are specified")
        if isinstance(regimes, list):
            normalized_input = _serialize_regimes(regimes)
        else:
            normalized_input = regimes
        validated_regimes, validated_transition, regime_names = _validate_regime_inputs(
            normalized_input, regime_transition
        )
        if regime_start is not None:
            regime_start_value = str(regime_start).strip()
            if not regime_start_value or regime_start_value not in regime_names:
                raise ValueError("regime_start must match one of the regime names")
            yaml_dict["regime_start"] = regime_start_value
        yaml_dict["regimes"] = validated_regimes
        yaml_dict["regime_transition"] = validated_transition

    return yaml_dict


def _build_yaml_dict(config: DefaultConfigView) -> Dict[str, Any]:
    """Backward-compatible alias for _build_yaml_from_config()."""
    return _build_yaml_from_config(config)


def _validate_yaml_dict(yaml_dict: Dict[str, Any]) -> None:
    """Validate using core ModelConfig; raises on error."""
    load_config(yaml_dict)


@contextmanager
def _temp_yaml_file(data: Dict[str, Any]):
    """Context manager for temporary YAML file that ensures cleanup."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml") as tmp:
        yaml.safe_dump(data, tmp, default_flow_style=False)
        tmp.flush()  # Ensure data is written to disk
        yield tmp.name


def _render_progress_bar(current_step: int, total_steps: int = 5) -> None:
    """Render step progress indicator."""
    st.markdown("### 📋 Scenario Configuration Wizard")

    progress = current_step / total_steps
    st.progress(progress)

    # Step indicators
    steps = [
        "Analysis Mode",
        "Capital Allocation",
        "Return & Risk",
        "Correlations",
        "Review & Run",
    ]
    cols = st.columns(5)

    for i, (col, step_name) in enumerate(zip(cols, steps)):
        with col:
            if i + 1 < current_step:
                st.success(f"✅ {i + 1}. {step_name}")
            elif i + 1 == current_step:
                st.info(f"📍 {i + 1}. {step_name}")
            else:
                st.write(f"⭕ {i + 1}. {step_name}")


def _render_step_1_analysis_mode(config: Any) -> Any:
    """Step 1: Analysis Mode Selection."""
    st.subheader("Step 1: Analysis Mode & Basic Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Choose Your Analysis Focus:**")

        mode_options = {
            AnalysisMode.RETURNS: "📈 Returns Analysis - Compare different return assumptions",
            AnalysisMode.CAPITAL: "💰 Capital Allocation - Optimize capital distribution",
            AnalysisMode.ALPHA_SHARES: "🎯 Alpha Shares - Analyze alpha source allocation",
            AnalysisMode.VOL_MULT: "📊 Volatility Stress - Test volatility scenarios",
        }

        selected_mode = st.selectbox(
            "Analysis Mode",
            options=list(mode_options.keys()),
            format_func=lambda x: mode_options[x],
            index=list(mode_options.keys()).index(config.analysis_mode),
            help="Choose the primary focus of your analysis",
        )

        # Update config if mode changed
        if selected_mode != config.analysis_mode:
            config = get_default_config(selected_mode)

    with col2:
        st.markdown("**Simulation Settings:**")

        config.n_simulations = st.number_input(
            "Number of Simulations",
            min_value=100,
            max_value=10000,
            value=config.n_simulations,
            step=100,
            help="More simulations provide more accurate results but take longer",
        )

        config.n_months = st.number_input(
            "Simulation Horizon [months]",
            min_value=1,
            max_value=60,
            value=config.n_months,
            help="Length of each simulation run in months",
        )

    # Mode-specific guidance
    mode_descriptions = {
        AnalysisMode.RETURNS: "This mode focuses on testing different return assumptions while keeping capital allocation fixed. Ideal for sensitivity analysis on alpha expectations.",
        AnalysisMode.CAPITAL: "This mode optimizes capital allocation across sleeves. Use when determining the optimal mix of internal PA, external PA, and active extension.",
        AnalysisMode.ALPHA_SHARES: "This mode analyzes the allocation of alpha sources across different strategies. Perfect for optimizing alpha capture efficiency.",
        AnalysisMode.VOL_MULT: "This mode stress-tests your portfolio under different volatility scenarios. Essential for risk management and extreme event preparation.",
    }

    st.info(f"**{selected_mode.value.title()} Mode:** {mode_descriptions[selected_mode]}")

    return config


def _render_step_2_capital(config: Any) -> Any:
    """Step 2: Capital Allocation Settings."""
    st.subheader("Step 2: Capital Allocation")

    ss = st.session_state
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Fund Capital ($ millions):**")

        config.total_fund_capital = st.number_input(
            "Total Fund Capital",
            min_value=1.0,
            value=ss.get(_TOTAL_CAPITAL_KEY, config.total_fund_capital),
            step=10.0,
            format="%.1f",
            help="Total capital available for allocation",
            key=_TOTAL_CAPITAL_KEY,
        )

        config.external_pa_capital = st.number_input(
            "External PA Capital [$M]",
            min_value=0.0,
            max_value=config.total_fund_capital,
            value=ss.get(_EXTERNAL_CAPITAL_KEY, config.external_pa_capital),
            step=5.0,
            format="%.1f",
            help="Capital allocated to external portable alpha managers",
            key=_EXTERNAL_CAPITAL_KEY,
        )

        config.active_ext_capital = st.number_input(
            "Active Extension Capital [$M]",
            min_value=0.0,
            max_value=config.total_fund_capital,
            value=ss.get(_ACTIVE_CAPITAL_KEY, config.active_ext_capital),
            step=5.0,
            format="%.1f",
            help="Capital for active equity overlay strategies",
            key=_ACTIVE_CAPITAL_KEY,
        )

        # Calculate remaining capital
        remaining = (
            config.total_fund_capital - config.external_pa_capital - config.active_ext_capital
        )
        config.internal_pa_capital = st.number_input(
            "Internal PA Capital [$M]",
            min_value=0.0,
            value=ss.get(_INTERNAL_CAPITAL_KEY, max(0.0, remaining)),
            step=5.0,
            format="%.1f",
            help="Capital managed internally for portable alpha",
            key=_INTERNAL_CAPITAL_KEY,
        )

    with col2:
        st.markdown("**Portfolio Weights & Shares:**")

        config.w_beta_h = st.slider(
            "Internal Beta Weight",
            min_value=0.0,
            max_value=1.0,
            value=ss.get(_W_BETA_KEY, config.w_beta_h),
            step=0.05,
            help="Beta component weight in internal sleeve",
            key=_W_BETA_KEY,
        )

        config.w_alpha_h = 1.0 - config.w_beta_h
        st.write(f"Internal Alpha Weight: {config.w_alpha_h:.2f} (auto-calculated)")

        config.theta_extpa = st.slider(
            "External PA Alpha Fraction",
            min_value=0.0,
            max_value=1.0,
            value=ss.get(_THETA_EXTPA_KEY, config.theta_extpa),
            step=0.05,
            help="Fraction of alpha from external PA manager",
            key=_THETA_EXTPA_KEY,
        )

        config.active_share = st.slider(
            "Active Extension Share",
            min_value=0.0,
            max_value=1.0,
            value=ss.get(_ACTIVE_SHARE_KEY, config.active_share),
            step=0.05,
            help=tooltip("active share"),
            key=_ACTIVE_SHARE_KEY,
        )

    # Validation and visualization
    total_allocated = (
        config.external_pa_capital + config.active_ext_capital + config.internal_pa_capital
    )

    if abs(total_allocated - config.total_fund_capital) > 0.01:
        st.error(
            f"❌ Capital allocation mismatch! Allocated: ${total_allocated:.1f}M, Total: ${config.total_fund_capital:.1f}M"
        )
    else:
        st.success(f"✅ Capital allocation balanced: ${total_allocated:.1f}M")

        # Allocation pie chart
        if total_allocated > 0:
            allocation_data = {
                "External PA": config.external_pa_capital,
                "Active Extension": config.active_ext_capital,
                "Internal PA": config.internal_pa_capital,
            }
            # Filter out zero allocations
            allocation_data = {k: v for k, v in allocation_data.items() if v > 0}

            if allocation_data:
                st.plotly_chart(
                    {
                        "data": [
                            {
                                "type": "pie",
                                "labels": list(allocation_data.keys()),
                                "values": list(allocation_data.values()),
                                "textinfo": "label+percent",
                            }
                        ],
                        "layout": {"title": "Capital Allocation", "showlegend": True},
                    },
                    use_container_width=True,
                )

    # Financing & Margin (optional enhancement)
    st.markdown("---")
    with st.expander("⚙️ Financing & Margin (optional)", expanded=False):
        # Defaults in session state
        ss = st.session_state
        if "financing_settings" not in ss:
            ss.financing_settings = {
                "financing_model": "simple_proxy",
                "reference_sigma": 0.01,
                "volatility_multiple": 3.0,
                "term_months": 1.0,
                "schedule_path": None,
            }

        fm = st.selectbox(
            "Financing model",
            options=["simple_proxy", "schedule"],
            index=["simple_proxy", "schedule"].index(ss.financing_settings["financing_model"]),
            help="Choose how to compute margin: proxy multiple or broker schedule",
        )
        ss.financing_settings["financing_model"] = fm

        colA, colB = st.columns(2)
        with colA:
            ref_sigma = st.number_input(
                "Reference sigma (monthly)",
                min_value=0.0,
                value=float(ss.financing_settings["reference_sigma"]),
                step=0.001,
                format="%.4f",
                help="Monthly reference volatility used for margin",
            )
            ss.financing_settings["reference_sigma"] = ref_sigma

        if fm == "simple_proxy":
            with colB:
                k = st.number_input(
                    "Volatility multiple (k)",
                    min_value=0.1,
                    value=float(ss.financing_settings["volatility_multiple"]),
                    step=0.1,
                    format="%.2f",
                    help="Proxy multiple used to size margin",
                )
                ss.financing_settings["volatility_multiple"] = k

            margin = calculate_margin_requirement(
                reference_sigma=ref_sigma,
                volatility_multiple=k,
                total_capital=(
                    config.total_fund_capital if hasattr(config, "total_fund_capital") else 1000.0
                ),
                financing_model="simple_proxy",
            )
            st.metric("Estimated Margin Requirement", f"${margin:.1f}M")

        else:
            with colB:
                term = st.number_input(
                    "Term (months)",
                    min_value=0.0,
                    value=float(ss.financing_settings["term_months"]),
                    step=0.5,
                    format="%.1f",
                    help="Interpolate schedule multiplier at this tenor",
                )
                ss.financing_settings["term_months"] = term

            uploaded = st.file_uploader(
                "Upload margin schedule CSV (columns: term,multiplier)",
                type=["csv"],
                accept_multiple_files=False,
                help="Terms must be non-negative, unique, strictly increasing; multipliers positive",
            )

            schedule_df = None
            if uploaded is not None:
                try:
                    # Persist to a secure temp path for validated loader
                    fd, tpath = tempfile.mkstemp(suffix=".csv")
                    try:
                        os.close(fd)  # Close file descriptor before writing to path
                        Path(tpath).write_bytes(uploaded.getvalue())
                    except Exception:
                        os.unlink(tpath)
                        raise
                    tmp_path = Path(tpath)
                    ss.financing_settings["schedule_path"] = str(tmp_path)
                    schedule_df = load_margin_schedule(tmp_path)
                    st.success("Schedule validated ✓")
                    st.dataframe(schedule_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Schedule error: {e}")
                    ss.financing_settings["schedule_path"] = None
            elif ss.financing_settings.get("schedule_path"):
                schedule_path = Path(str(ss.financing_settings["schedule_path"]))
                if schedule_path.exists():
                    try:
                        schedule_df = load_margin_schedule(schedule_path)
                        st.info(f"Using schedule at {schedule_path}")
                        st.dataframe(schedule_df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Schedule error: {e}")
                        ss.financing_settings["schedule_path"] = None
                else:
                    st.warning(f"Schedule file not found: {schedule_path}")
                    ss.financing_settings["schedule_path"] = None

            if schedule_df is not None:
                # Interpolate multiplier and compute margin
                margin = calculate_margin_requirement(
                    reference_sigma=ref_sigma,
                    total_capital=(
                        config.total_fund_capital
                        if hasattr(config, "total_fund_capital")
                        else 1000.0
                    ),
                    financing_model="schedule",
                    margin_schedule=schedule_df,
                    term_months=term,
                )
                # Interpolated k = margin / (ref_sigma * total_capital)
                total_cap = (
                    config.total_fund_capital if hasattr(config, "total_fund_capital") else 1000.0
                )
                k_interp = margin / max(ref_sigma * total_cap, 1e-12)
                c1, c2 = st.columns(2)
                c1.metric("Interpolated k (multiplier)", f"{k_interp:.2f}")
                c2.metric("Estimated Margin Requirement", f"${margin:.1f}M")

    return config


def _render_sleeve_suggestor(config: DefaultConfigView) -> None:
    st.subheader("Sleeve Suggestor")
    st.markdown("Provide risk constraints and generate ranked sleeve allocations.")

    col1, col2 = st.columns(2)
    with col1:
        st.number_input(
            "Max Tracking Error",
            min_value=0.0,
            value=st.session_state.get("sleeve_max_te", 0.02),
            step=0.01,
            format="%.2f",
            key="sleeve_max_te",
        )
        st.number_input(
            "Max Breach Probability",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("sleeve_max_breach", 0.5),
            step=0.05,
            format="%.2f",
            key="sleeve_max_breach",
        )
        st.number_input(
            "Max monthly_CVaR",
            min_value=0.0,
            value=st.session_state.get("sleeve_max_cvar", 0.05),
            step=0.01,
            format="%.2f",
            key="sleeve_max_cvar",
        )
        st.number_input(
            "Max Terminal Shortfall Probability",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("sleeve_max_shortfall", 0.05),
            step=0.05,
            format="%.2f",
            key="sleeve_max_shortfall",
        )

    with col2:
        st.number_input(
            "Grid Step (fraction of total capital)",
            min_value=0.01,
            max_value=1.0,
            value=st.session_state.get("sleeve_step", 0.25),
            step=0.05,
            format="%.2f",
            key="sleeve_step",
        )
        st.number_input(
            "Max Evaluations",
            min_value=50,
            max_value=5000,
            value=st.session_state.get("sleeve_max_evals", 500),
            step=50,
            key="sleeve_max_evals",
            help="Cap the number of grid points to evaluate.",
        )
        scope_labels = {
            "sleeves": "Per-sleeve constraints",
            "total": "Total portfolio constraints",
            "both": "Sleeves + total constraints",
        }
        current_scope = st.session_state.get("sleeve_constraint_scope")
        if current_scope is None:
            current_scope = "sleeves" if config.sleeve_constraint_scope == "per_sleeve" else "total"
        if current_scope not in scope_labels:
            current_scope = "sleeves"
        scope_index = list(scope_labels.keys()).index(current_scope)
        st.selectbox(
            "Constraint Scope",
            options=list(scope_labels.keys()),
            index=scope_index,
            format_func=lambda x: scope_labels[x],
            key="sleeve_constraint_scope",
        )
        st.checkbox(
            "Validate constraints on run",
            value=st.session_state.get("sleeve_validate_on_run", config.sleeve_validate_on_run),
            key="sleeve_validate_on_run",
            help="Raise an error if run results violate the sleeve constraints.",
        )

    constraints = {
        "max_te": float(st.session_state["sleeve_max_te"]),
        "max_breach": float(st.session_state["sleeve_max_breach"]),
        "max_cvar": float(st.session_state["sleeve_max_cvar"]),
        "max_shortfall": float(st.session_state["sleeve_max_shortfall"]),
        "step": float(st.session_state["sleeve_step"]),
        "max_evals": int(st.session_state["sleeve_max_evals"]),
        "constraint_scope": st.session_state["sleeve_constraint_scope"],
    }

    config.sleeve_max_te = float(st.session_state["sleeve_max_te"])
    config.sleeve_max_breach = float(st.session_state["sleeve_max_breach"])
    config.sleeve_max_cvar = float(st.session_state["sleeve_max_cvar"])
    config.sleeve_max_shortfall = float(st.session_state["sleeve_max_shortfall"])
    config.sleeve_constraint_scope = _normalize_sleeve_constraint_scope(
        st.session_state["sleeve_constraint_scope"]
    )
    config.sleeve_validate_on_run = bool(
        st.session_state.get("sleeve_validate_on_run", config.sleeve_validate_on_run)
    )

    if st.button("Run Suggestor"):
        yaml_dict = _build_yaml_from_config(config)
        cfg = load_config(yaml_dict)
        idx_path = Path(__file__).resolve().parents[2] / "data" / "sp500tr_fred_divyield.csv"
        if not idx_path.exists():
            st.error(f"Default index file missing: {idx_path}")
            return
        idx_series = load_index_returns(idx_path)
        use_seed = st.session_state.get("wizard_use_seed", True)
        seed_value = st.session_state.get("wizard_seed", 42)
        suggest_seed = int(seed_value) if use_seed else None
        df = run_sleeve_suggestions(
            cfg,
            idx_series,
            max_te=constraints["max_te"],
            max_breach=constraints["max_breach"],
            max_cvar=constraints["max_cvar"],
            max_shortfall=constraints["max_shortfall"],
            step=constraints["step"],
            max_evals=constraints["max_evals"],
            constraint_scope=constraints["constraint_scope"],
            seed=suggest_seed,
        )
        st.session_state["sleeve_suggestions"] = df
        st.session_state["sleeve_suggestion_constraints"] = constraints
        try:
            frontier_df = run_sleeve_frontier(
                cfg,
                idx_series,
                max_te=constraints["max_te"],
                max_breach=constraints["max_breach"],
                max_cvar=constraints["max_cvar"],
                max_shortfall=constraints["max_shortfall"],
                step=constraints["step"],
                max_evals=constraints["max_evals"],
                constraint_scope=constraints["constraint_scope"],
                seed=suggest_seed,
            )
            st.session_state["sleeve_frontier"] = frontier_df
        except Exception as exc:
            st.session_state["sleeve_frontier"] = pd.DataFrame()
            st.error(f"Frontier computation failed: {exc}")

    constraints_used = st.session_state.get("sleeve_suggestion_constraints", constraints)
    st.markdown("**Constraint Summary:**")
    st.write(
        f"Max TE: {constraints_used['max_te']:.2%} | "
        f"Max Breach: {constraints_used['max_breach']:.2%} | "
        f"Max monthly_CVaR: {constraints_used['max_cvar']:.2%} | "
        f"Max Shortfall: {constraints_used['max_shortfall']:.2%} | "
        f"Scope: {scope_labels.get(constraints_used['constraint_scope'], constraints_used['constraint_scope'])}"
    )

    suggestions = st.session_state.get("sleeve_suggestions")
    if suggestions is None:
        return
    if suggestions.empty:
        st.warning("No feasible sleeve allocations found.")
        return

    ranked = suggestions.sort_values("risk_score", ascending=True).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))

    top_n = st.number_input(
        "Show top results",
        min_value=1,
        max_value=len(ranked),
        value=min(10, len(ranked)),
        step=1,
    )

    preferred_cols = [
        "rank",
        "external_pa_capital",
        "active_ext_capital",
        "internal_pa_capital",
        "risk_score",
        "ExternalPA_monthly_TE",
        "ExternalPA_monthly_BreachProb",
        "ExternalPA_monthly_CVaR",
        "ExternalPA_terminal_ShortfallProb",
        "ActiveExt_monthly_TE",
        "ActiveExt_monthly_BreachProb",
        "ActiveExt_monthly_CVaR",
        "ActiveExt_terminal_ShortfallProb",
        "InternalPA_monthly_TE",
        "InternalPA_monthly_BreachProb",
        "InternalPA_monthly_CVaR",
        "InternalPA_terminal_ShortfallProb",
        "Total_monthly_TE",
        "Total_monthly_BreachProb",
        "Total_monthly_CVaR",
        "Total_terminal_ShortfallProb",
    ]
    display_cols = [col for col in preferred_cols if col in ranked.columns]
    tradeoff_table = ranked.loc[:, display_cols].head(int(top_n))
    st.dataframe(tradeoff_table, use_container_width=True)

    frontier_df = st.session_state.get("sleeve_frontier")
    if isinstance(frontier_df, pd.DataFrame) and not frontier_df.empty:
        st.subheader("Efficient Frontier")
        st.caption("Return vs tracking error with CVaR shading; infeasible points are marked.")
        fig = frontier_viz.make(
            frontier_df,
            max_te=constraints_used["max_te"],
            max_cvar=constraints_used["max_cvar"],
            max_breach=constraints_used["max_breach"],
            max_shortfall=constraints_used["max_shortfall"],
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run the suggestor to populate the frontier visualization.")

    selected_rank = st.number_input(
        "Select rank to apply",
        min_value=1,
        max_value=len(ranked),
        value=1,
        step=1,
    )
    confirm_apply = st.checkbox("Confirm apply selected suggestion")
    if st.button("Apply suggestion", disabled=not confirm_apply):
        row = ranked.iloc[int(selected_rank) - 1]
        config.external_pa_capital = float(row["external_pa_capital"])
        config.active_ext_capital = float(row["active_ext_capital"])
        config.internal_pa_capital = float(row["internal_pa_capital"])
        st.session_state[_TOTAL_CAPITAL_KEY] = float(config.total_fund_capital)
        st.session_state[_EXTERNAL_CAPITAL_KEY] = float(config.external_pa_capital)
        st.session_state[_ACTIVE_CAPITAL_KEY] = float(config.active_ext_capital)
        st.session_state[_INTERNAL_CAPITAL_KEY] = float(config.internal_pa_capital)
        st.session_state["suggestion_applied"] = True
        st.session_state["suggestion_confirmed"] = True
        st.success("Suggested allocation applied. Review before running.")
    elif not confirm_apply:
        st.info("Confirm the selection to enable Apply suggestion.")


def _render_step_3_returns_risk(config: Any) -> Any:
    """Step 3: Return & Risk Parameters."""
    st.subheader("Step 3: Return & Risk Parameters")

    st.markdown("*All parameters are annualized*")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Expected Returns (Annual %):**")

        config.mu_h = (
            st.number_input(
                "In-House Alpha Return [annual %]",
                value=config.mu_h * 100,
                step=0.5,
                format="%.2f",
                help="Expected annual return from in-house alpha generation",
            )
            / 100
        )

        config.mu_e = (
            st.number_input(
                "Extension Alpha Return [annual %]",
                value=config.mu_e * 100,
                step=0.5,
                format="%.2f",
                help="Expected annual return from active extension strategies",
            )
            / 100
        )

        config.mu_m = (
            st.number_input(
                "External PA Alpha Return [annual %]",
                value=config.mu_m * 100,
                step=0.5,
                format="%.2f",
                help="Expected annual return from external PA managers",
            )
            / 100
        )

    with col2:
        st.markdown("**Volatility (Annual %):**")

        config.sigma_h = (
            st.number_input(
                "In-House Alpha Volatility [annual %]",
                min_value=0.01,
                value=config.sigma_h * 100,
                step=0.1,
                format="%.2f",
                help="Annual volatility of in-house alpha",
            )
            / 100
        )

        config.sigma_e = (
            st.number_input(
                "Extension Alpha Volatility [annual %]",
                min_value=0.01,
                value=config.sigma_e * 100,
                step=0.1,
                format="%.2f",
                help="Annual volatility of active extension alpha",
            )
            / 100
        )

        config.sigma_m = (
            st.number_input(
                "External PA Alpha Volatility [annual %]",
                min_value=0.01,
                value=config.sigma_m * 100,
                step=0.1,
                format="%.2f",
                help="Annual volatility of external PA alpha",
            )
            / 100
        )

    # Risk metrics selection
    st.markdown("**Risk Metrics to Calculate:**")

    metric_options = {
        RiskMetric.RETURN: "Return - Expected portfolio return",
        RiskMetric.RISK: "Risk - Portfolio volatility",
        RiskMetric.SHORTFALL_PROB: "Shortfall Probability - Risk of underperformance",
    }

    default_metrics = _normalize_risk_metric_defaults(config.risk_metrics)
    selected_metrics = st.multiselect(
        "Select Risk Metrics",
        options=list(metric_options.keys()),
        default=default_metrics,
        format_func=lambda x: metric_options[x],
        help="Choose which risk metrics to calculate and display",
    )

    config.risk_metrics = _serialize_risk_metrics(selected_metrics)

    with st.expander("Advanced Simulation Settings", expanded=False):
        st.markdown("**Return Distribution:**")
        distribution_options = ["normal", "student_t"]
        dist_index = (
            distribution_options.index(config.return_distribution)
            if config.return_distribution in distribution_options
            else 0
        )
        config.return_distribution = st.selectbox(
            "Return distribution",
            options=distribution_options,
            index=dist_index,
            help="Controls the base distribution used for return draws.",
        )

        copula_options = (
            ["gaussian"] if config.return_distribution == "normal" else ["gaussian", "t"]
        )
        copula_index = (
            copula_options.index(config.return_copula)
            if config.return_copula in copula_options
            else 0
        )
        config.return_copula = st.selectbox(
            "Return copula",
            options=copula_options,
            index=copula_index,
            help="Joint return dependence structure for simulated draws.",
        )

        config.return_t_df = st.number_input(
            "Student-t degrees of freedom",
            min_value=2.01,
            value=float(config.return_t_df),
            step=0.5,
            format="%.2f",
            help="Used only when return distribution is Student-t.",
            disabled=config.return_distribution != "student_t",
        )

        st.markdown("**Volatility Regime:**")
        vol_regime_options = ["single", "two_state"]
        vol_regime_index = (
            vol_regime_options.index(config.vol_regime)
            if config.vol_regime in vol_regime_options
            else 0
        )
        config.vol_regime = st.selectbox(
            "Volatility regime",
            options=vol_regime_options,
            index=vol_regime_index,
            help="Switch between a single volatility state or a two-state regime.",
        )

        vol_window_min = 2 if config.vol_regime == "two_state" else 1
        vol_window_value = max(int(config.vol_regime_window), vol_window_min)
        config.vol_regime_window = st.number_input(
            "Volatility regime window (months)",
            min_value=vol_window_min,
            value=vol_window_value,
            step=1,
            help="Window length for regime estimation; must be > 1 for two-state regimes.",
        )

        st.markdown("**Correlation Repair:**")
        shrinkage_options = ["none", "ledoit_wolf"]
        shrinkage_index = (
            shrinkage_options.index(config.covariance_shrinkage)
            if config.covariance_shrinkage in shrinkage_options
            else 0
        )
        config.covariance_shrinkage = st.selectbox(
            "Covariance shrinkage",
            options=shrinkage_options,
            index=shrinkage_index,
            help="Optional shrinkage estimator applied before correlation repair.",
        )

        repair_mode_options = ["error", "warn_fix"]
        repair_mode_index = (
            repair_mode_options.index(config.correlation_repair_mode)
            if config.correlation_repair_mode in repair_mode_options
            else 0
        )
        config.correlation_repair_mode = st.selectbox(
            "Correlation repair mode",
            options=repair_mode_options,
            index=repair_mode_index,
            help="Choose whether invalid correlations raise an error or are repaired.",
        )

        config.correlation_repair_shrinkage = st.slider(
            "Correlation repair shrinkage",
            min_value=0.0,
            max_value=1.0,
            value=float(config.correlation_repair_shrinkage),
            step=0.05,
            help="Shrinkage toward identity before repairing correlations.",
        )

        max_delta_enabled = st.checkbox(
            "Enforce max correlation repair delta",
            value=config.correlation_repair_max_abs_delta is not None,
            help="Fail validation if repaired correlations move too far from the original matrix.",
        )
        if max_delta_enabled:
            max_delta_default = (
                float(config.correlation_repair_max_abs_delta)
                if config.correlation_repair_max_abs_delta is not None
                else 0.0
            )
            config.correlation_repair_max_abs_delta = st.number_input(
                "Correlation repair max abs delta",
                min_value=0.0,
                value=max_delta_default,
                step=0.01,
                format="%.2f",
                help="Maximum absolute delta allowed after correlation repair.",
            )
        else:
            config.correlation_repair_max_abs_delta = None

        st.markdown("**Backend:**")
        backend_options = list(SUPPORTED_BACKENDS)
        backend_index = (
            backend_options.index(config.backend) if config.backend in backend_options else 0
        )
        config.backend = st.selectbox(
            "Simulation backend",
            options=backend_options,
            index=backend_index,
            help="Select the numerical backend used for simulations.",
        )

    with st.expander("Regime Switching", expanded=False):
        st.markdown("Enable a Markov regime-switching configuration for the simulation.")

        regime_enabled = st.checkbox(
            "Enable regime switching",
            value=bool(config.regimes),
            key=_REGIME_ENABLED_KEY,
        )

        if not regime_enabled:
            config.regimes = None
            config.regime_transition = None
            config.regime_start = None
        else:
            regimes_default = ""
            if config.regimes:
                regimes_default = yaml.safe_dump(
                    _serialize_regimes(config.regimes),
                    default_flow_style=False,
                )
            regimes_raw = st.text_area(
                "Regimes (YAML/JSON)",
                value=regimes_default,
                key=_REGIMES_KEY,
                height=180,
                help="Provide a list of regime dicts with a unique name per regime.",
                placeholder=(
                    "- name: Calm\n  idx_sigma_multiplier: 0.8\n"
                    "- name: Stressed\n  idx_sigma_multiplier: 1.3"
                ),
            )

            transition_default = ""
            if config.regime_transition:
                transition_default = yaml.safe_dump(
                    config.regime_transition,
                    default_flow_style=False,
                )
            transition_raw = st.text_area(
                "Regime transition matrix (YAML/JSON)",
                value=transition_default,
                key=_REGIME_TRANSITION_KEY,
                height=140,
                help="Square matrix with rows summing to 1.0.",
                placeholder="- [0.9, 0.1]\n- [0.2, 0.8]",
            )

            if not regimes_raw.strip():
                st.error("Regimes are required when regime switching is enabled.")
                st.stop()
            if not transition_raw.strip():
                st.error("Transition matrix is required when regime switching is enabled.")
                st.stop()

            try:
                regimes = _parse_yaml_or_json(regimes_raw, "Regimes")
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

            try:
                transition = _parse_yaml_or_json(transition_raw, "Transition matrix")
            except ValueError as exc:
                st.error(str(exc))
                st.stop()
            try:
                regimes, transition, regime_names = _validate_regime_inputs(regimes, transition)
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

            start_options = ["(auto)"] + regime_names
            default_start = config.regime_start if config.regime_start in regime_names else "(auto)"
            start_index = start_options.index(default_start)
            start_selection = st.selectbox(
                "Starting regime (optional)",
                options=start_options,
                index=start_index,
                key=_REGIME_START_KEY,
                help="Choose the initial regime; auto uses the first regime.",
            )

            config.regimes = regimes
            config.regime_transition = transition
            config.regime_start = None if start_selection == "(auto)" else start_selection

    st.markdown("---")
    _render_sleeve_suggestor(config)

    return config


def _render_step_4_correlations(config: Any) -> Any:
    """Step 4: Correlation Parameters."""
    st.subheader("Step 4: Correlation Parameters")

    st.markdown("*Set correlations between different alpha sources and the market index*")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Index Correlations:**")

        config.rho_idx_h = st.slider(
            "Index ↔ In-House Alpha",
            min_value=-0.99,
            max_value=0.99,
            value=config.rho_idx_h,
            step=0.05,
            help="Correlation between market index and in-house alpha",
        )

        config.rho_idx_e = st.slider(
            "Index ↔ Extension Alpha",
            min_value=-0.99,
            max_value=0.99,
            value=config.rho_idx_e,
            step=0.05,
            help="Correlation between market index and extension alpha",
        )

        config.rho_idx_m = st.slider(
            "Index ↔ External PA Alpha",
            min_value=-0.99,
            max_value=0.99,
            value=config.rho_idx_m,
            step=0.05,
            help="Correlation between market index and external PA alpha",
        )

    with col2:
        st.markdown("**Cross-Alpha Correlations:**")

        config.rho_h_e = st.slider(
            "In-House ↔ Extension Alpha",
            min_value=-0.99,
            max_value=0.99,
            value=config.rho_h_e,
            step=0.05,
            help="Correlation between in-house and extension alpha",
        )

        config.rho_h_m = st.slider(
            "In-House ↔ External PA Alpha",
            min_value=-0.99,
            max_value=0.99,
            value=config.rho_h_m,
            step=0.05,
            help="Correlation between in-house and external PA alpha",
        )

        config.rho_e_m = st.slider(
            "Extension ↔ External PA Alpha",
            min_value=-0.99,
            max_value=0.99,
            value=config.rho_e_m,
            step=0.05,
            help="Correlation between extension and external PA alpha",
        )

    # Correlation matrix visualization
    import numpy as np
    import pandas as pd

    corr_matrix = pd.DataFrame(
        [
            [1.0, config.rho_idx_h, config.rho_idx_e, config.rho_idx_m],
            [config.rho_idx_h, 1.0, config.rho_h_e, config.rho_h_m],
            [config.rho_idx_e, config.rho_h_e, 1.0, config.rho_e_m],
            [config.rho_idx_m, config.rho_h_m, config.rho_e_m, 1.0],
        ],
        index=["Index", "In-House", "Extension", "External PA"],
        columns=["Index", "In-House", "Extension", "External PA"],
    )

    st.markdown("**Correlation Matrix:**")
    st.dataframe(corr_matrix.round(3), use_container_width=True)

    # Check for potential issues
    eigenvalues = np.linalg.eigvals(corr_matrix.values)
    min_eigenvalue = np.min(eigenvalues)

    if min_eigenvalue < -1e-6:
        st.warning(
            f"⚠️ Correlation matrix may not be positive definite (min eigenvalue: {min_eigenvalue:.6f}). Consider adjusting correlations."
        )
    else:
        st.success("✅ Correlation matrix is valid")
    return config

    return config


def _render_step_5_review(config: DefaultConfigView) -> bool:
    """Step 5: Review configuration and optionally run simulation."""
    st.subheader("Step 5: Review & Run")

    # Configuration summary
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Configuration Summary:**")

        st.write(f"**Analysis Mode:** {config.analysis_mode.value.title()}")
        st.write(f"**Simulations:** {config.n_simulations:,}")
        st.write(f"**Time Horizon:** {config.n_months} months")

        st.markdown("**Capital Allocation:**")
        total_capital = config.total_fund_capital
        st.write(f"• Total Fund: ${total_capital:.1f}M")
        st.write(
            f"• External PA: ${config.external_pa_capital:.1f}M ({config.external_pa_capital / total_capital:.1%})"
        )
        st.write(
            f"• Active Extension: ${config.active_ext_capital:.1f}M ({config.active_ext_capital / total_capital:.1%})"
        )
        st.write(
            f"• Internal PA: ${config.internal_pa_capital:.1f}M ({config.internal_pa_capital / total_capital:.1%})"
        )

    with col2:
        st.markdown("**Expected Returns & Risk:**")
        st.write(f"• In-House Alpha: {config.mu_h:.2%} ± {config.sigma_h:.2%}")
        st.write(f"• Extension Alpha: {config.mu_e:.2%} ± {config.sigma_e:.2%}")
        st.write(f"• External PA Alpha: {config.mu_m:.2%} ± {config.sigma_m:.2%}")

        st.markdown("**Key Correlations:**")
        st.write(f"• Index ↔ In-House: {config.rho_idx_h:.2f}")
        st.write(f"• In-House ↔ Extension: {config.rho_h_e:.2f}")
        st.write(f"• Extension ↔ External PA: {config.rho_e_m:.2f}")

    # Financing summary
    fs = st.session_state.get("financing_settings", {})
    if fs:
        st.markdown("**Financing & Margin:**")
        fm = fs.get("financing_model", "simple_proxy")
        st.write(f"• Model: {fm}")
        st.write(f"• Reference sigma (monthly): {float(fs.get('reference_sigma', 0.01)):.4f}")
        if fm == "simple_proxy":
            st.write(f"• Volatility multiple (k): {float(fs.get('volatility_multiple', 3.0)):.2f}")
        else:
            st.write(f"• Term (months): {float(fs.get('term_months', 1.0)):.1f}")
            st.write(f"• Schedule file: {'set' if fs.get('schedule_path') else 'not set'}")

    # Diff vs last run
    if "last_wizard_config" in st.session_state:
        last_config = st.session_state.last_wizard_config
        if last_config != config:
            with st.expander("🔍 Changes vs Last Run", expanded=False):
                changes = []

                # Compare key parameters
                comparisons = [
                    (
                        "Analysis Mode",
                        last_config.analysis_mode.value,
                        config.analysis_mode.value,
                    ),
                    ("Simulations", last_config.n_simulations, config.n_simulations),
                    (
                        "External PA Capital",
                        f"{last_config.external_pa_capital:.1f}M",
                        f"{config.external_pa_capital:.1f}M",
                    ),
                    (
                        "Active Extension Capital",
                        f"{last_config.active_ext_capital:.1f}M",
                        f"{config.active_ext_capital:.1f}M",
                    ),
                    (
                        "Internal PA Capital",
                        f"{last_config.internal_pa_capital:.1f}M",
                        f"{config.internal_pa_capital:.1f}M",
                    ),
                    (
                        "In-House Return",
                        f"{last_config.mu_h:.2%}",
                        f"{config.mu_h:.2%}",
                    ),
                    (
                        "Extension Return",
                        f"{last_config.mu_e:.2%}",
                        f"{config.mu_e:.2%}",
                    ),
                    (
                        "External PA Return",
                        f"{last_config.mu_m:.2%}",
                        f"{config.mu_m:.2%}",
                    ),
                ]

                for param, old_val, new_val in comparisons:
                    if old_val != new_val:
                        changes.append(f"• **{param}**: {old_val} → {new_val}")

                if changes:
                    st.markdown("**Changed Parameters:**")
                    for change in changes:
                        st.markdown(change)
                else:
                    st.info("No changes from last configuration")
        else:
            st.info("💡 Configuration matches last run")
    else:
        st.info("💡 This is your first configuration - no previous run to compare")

    # Validation
    try:
        yaml_preview = _build_yaml_from_config(config)
        _validate_yaml_dict(yaml_preview)
        validation_status = "✅"
        validation_msg = "Configuration is valid"
        can_run = True
    except Exception as e:
        validation_status = "❌"
        validation_msg = f"Validation errors: {str(e)}"
        can_run = False

    if can_run:
        st.success(f"{validation_status} {validation_msg}")
    else:
        st.error(f"{validation_status} {validation_msg}")

    st.markdown("**Reproducibility:**")
    use_seed = st.checkbox(
        "Use deterministic seed",
        value=st.session_state.get("wizard_use_seed", True),
        key="wizard_use_seed",
    )
    st.number_input(
        "Random seed",
        min_value=0,
        step=1,
        value=int(st.session_state.get("wizard_seed", 42)),
        key="wizard_seed",
        disabled=not use_seed,
        help="Fix the random seed to make runs reproducible.",
    )

    confirmed = True
    if st.session_state.get("suggestion_applied"):
        confirmed = st.checkbox(
            "I confirm the suggested sleeve allocation",
            key="suggestion_confirmed",
        )
        if not confirmed:
            st.warning("Sleeve suggestion applied - please confirm before running.")
    can_run = can_run and confirmed

    # Action buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Reset to Defaults", help="Reset all parameters to sensible defaults"):
            mode = config.analysis_mode
            st.session_state.wizard_config = get_default_config(mode)
            st.rerun()

    with col2:
        if st.button("💾 Download YAML", help="Download configuration as YAML file"):
            yaml_data = _build_yaml_from_config(config)
            yaml_str = yaml.safe_dump(yaml_data, default_flow_style=False)
            st.download_button(
                "Download Configuration",
                yaml_str,
                file_name=f"scenario_{config.analysis_mode.value}.yml",
                mime="application/x-yaml",
            )

    with col3:
        run_simulation = st.button(
            f"{validation_status} Run Simulation",
            disabled=not can_run,
            help="Start the Monte Carlo simulation with current parameters",
        )

    return run_simulation


def main() -> None:
    """Main wizard interface with 5-step stepper."""

    # Initialize session state
    if "wizard_step" not in st.session_state:
        st.session_state.wizard_step = 1

    if "wizard_config" not in st.session_state:
        st.session_state.wizard_config = get_default_config(AnalysisMode.RETURNS)

    # Theme and validation settings in sidebar
    theme_path = st.sidebar.text_input("Theme file", _DEF_THEME)
    apply_theme(theme_path)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Quick Actions")

    if st.sidebar.button("🏠 Reset to Step 1"):
        st.session_state.wizard_step = 1
        st.rerun()

    if st.sidebar.button("🔄 Reset All Defaults"):
        current = st.session_state.wizard_config
        mode = current.analysis_mode
        st.session_state.wizard_config = get_default_config(mode)
        st.rerun()

    # Alternative: File upload mode
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Alternative: File Upload")

    cfg = st.sidebar.file_uploader("Upload Scenario YAML", type=["yaml", "yml"])
    if cfg is not None:
        try:
            config_data = yaml.safe_load(cfg.getvalue())
            if st.sidebar.button("Load Configuration"):
                model_config = load_config(config_data)
                wizard_config = make_view_from_model(model_config)
                financing_model = model_config.financing_model
                st.session_state.financing_settings = {
                    "financing_model": financing_model,
                    "reference_sigma": model_config.reference_sigma,
                    "volatility_multiple": model_config.volatility_multiple,
                    "term_months": model_config.financing_term_months,
                    "schedule_path": (
                        str(model_config.financing_schedule_path)
                        if model_config.financing_schedule_path
                        else None
                    ),
                }
                st.session_state.wizard_config = wizard_config
                st.session_state.wizard_step = 5  # Go to review step
                st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error loading config: {e}")

    # Main wizard interface
    current_step = st.session_state.wizard_step
    config = st.session_state.wizard_config

    with st.sidebar.expander("💬 Config Chat", expanded=False):
        _render_config_chat_sidebar()

    _render_progress_bar(current_step)

    # Render current step
    if current_step == 1:
        config = _render_step_1_analysis_mode(config)
        st.session_state.wizard_config = config

        if st.button("➡️ Next: Capital Allocation", use_container_width=True):
            st.session_state.wizard_step = 2
            st.rerun()

    elif current_step == 2:
        config = _render_step_2_capital(config)
        st.session_state.wizard_config = config

        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back: Analysis Mode"):
                st.session_state.wizard_step = 1
                st.rerun()
        with col2:
            if st.button("➡️ Next: Return & Risk"):
                st.session_state.wizard_step = 3
                st.rerun()

    elif current_step == 3:
        config = _render_step_3_returns_risk(config)
        st.session_state.wizard_config = config

        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back: Capital Allocation"):
                st.session_state.wizard_step = 2
                st.rerun()
        with col2:
            if st.button("➡️ Next: Correlations"):
                st.session_state.wizard_step = 4
                st.rerun()

    elif current_step == 4:
        config = _render_step_4_correlations(config)
        st.session_state.wizard_config = config

        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back: Return & Risk"):
                st.session_state.wizard_step = 3
                st.rerun()
        with col2:
            if st.button("➡️ Next: Review & Run"):
                st.session_state.wizard_step = 5
                st.rerun()

    elif current_step == 5:
        run_simulation = _render_step_5_review(config)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back: Correlations"):
                st.session_state.wizard_step = 4
                st.rerun()

        # Handle simulation execution
        if run_simulation:
            # Store config for diff functionality
            st.session_state.last_wizard_config = config

            # Need index data for simulation
            st.subheader("📊 Index Data Required")
            idx = st.file_uploader(
                "Index CSV", type=["csv"], help="Upload market index returns data"
            )
            output = st.text_input("Output workbook", _DEF_XLSX)

            if idx is not None:
                # Convert config to YAML and run simulation
                yaml_data = _build_yaml_from_config(config)
                with _temp_yaml_file(yaml_data) as cfg_path:
                    # Write index data to a secure temp file
                    fd, idx_path = tempfile.mkstemp(suffix=".csv")
                    sim_ok = False
                    try:
                        os.close(fd)  # Close file descriptor before writing to path
                        Path(idx_path).write_bytes(idx.getvalue())

                        use_seed = st.session_state.get("wizard_use_seed", True)
                        seed_value = st.session_state.get("wizard_seed", 42)
                        args = dashboard_cli.build_pa_core_args(
                            cfg_path,
                            idx_path,
                            output,
                            use_seed=use_seed,
                            seed_value=int(seed_value) if use_seed else None,
                        )

                        with st.spinner("🔄 Running simulation..."):
                            pa_cli.main(args)
                        sim_ok = True
                    except Exception as exc:
                        st.error(f"❌ Simulation failed: {exc}")
                    finally:
                        # Cleanup index temp file
                        try:
                            os.unlink(idx_path)
                        except Exception:
                            pass

                    if sim_ok:
                        st.success(f"✅ Simulation complete! Results written to {output}")
                        st.balloons()

                        # Show quick margin summary using current financing settings if available
                        fs = st.session_state.get("financing_settings", {})
                        fm = fs.get("financing_model", "simple_proxy")
                        ref_sigma = float(fs.get("reference_sigma", 0.01))
                        vol_mult = float(fs.get("volatility_multiple", 3.0))
                        term_m = float(fs.get("term_months", 1.0))
                        sched_path = fs.get("schedule_path")

                        try:
                            margin_requirement = calculate_margin_requirement(
                                reference_sigma=ref_sigma,
                                volatility_multiple=vol_mult,
                                total_capital=float(yaml_data.get("total_fund_capital", 1000.0)),
                                financing_model=fm,
                                schedule_path=Path(sched_path) if sched_path else None,
                                term_months=term_m,
                            )
                            st.info(f"Margin requirement: ${margin_requirement:.1f}M (model: {fm})")
                        except Exception:
                            # Non-blocking: skip margin summary if misconfigured
                            pass

                        # Show configuration used
                        yaml_str = yaml.safe_dump(yaml_data, default_flow_style=False)
                        with st.expander("Configuration Used", expanded=False):
                            st.code(yaml_str, language="yaml")

                        st.page_link("pages/4_Results.py", label="📈 View Results →")


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
