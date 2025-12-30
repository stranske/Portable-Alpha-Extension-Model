"""5-Step Wizard for guided scenario creation with validation."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

import streamlit as st
import yaml

from dashboard.app import _DEF_THEME, _DEF_XLSX, apply_theme
from dashboard.glossary import tooltip
from pa_core import cli as pa_cli
from pa_core.config import load_config
from pa_core.data import load_index_returns
from pa_core.sleeve_suggestor import suggest_sleeve_sizes
from pa_core.validators import calculate_margin_requirement, load_margin_schedule
from pa_core.wizard_schema import (
    AnalysisMode,
    DefaultConfigView,
    RiskMetric,
    WizardScenarioConfig,
    get_default_config,
)


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

    total_capital = float(config.total_fund_capital)
    external_pa_capital = float(config.external_pa_capital)
    active_ext_capital = float(config.active_ext_capital)
    internal_pa_capital = float(config.internal_pa_capital)

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

    risk_metrics = config.risk_metrics

    fm = fs.get("financing_model", "simple_proxy")
    ref_sigma = float(fs.get("reference_sigma", 0.01))
    vol_mult = float(fs.get("volatility_multiple", 3.0))
    term_m = float(fs.get("term_months", 1.0))
    sched_path = fs.get("schedule_path")

    yaml_dict: Dict[str, Any] = {
        "N_SIMULATIONS": n_simulations,
        "N_MONTHS": n_months,
        "analysis_mode": (
            analysis_mode.value
            if hasattr(analysis_mode, "value")
            else str(analysis_mode)
        ),
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
        "reference_sigma": ref_sigma,
        "volatility_multiple": vol_mult,
        "financing_model": fm,
        "financing_schedule_path": (
            str(sched_path) if (fm == "schedule" and sched_path) else None
        ),
        "financing_term_months": term_m,
    }

    return yaml_dict


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
                st.success(f"✅ {i+1}. {step_name}")
            elif i + 1 == current_step:
                st.info(f"📍 {i+1}. {step_name}")
            else:
                st.write(f"⭕ {i+1}. {step_name}")


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

    st.info(
        f"**{selected_mode.value.title()} Mode:** {mode_descriptions[selected_mode]}"
    )

    return config


def _render_step_2_capital(config: Any) -> Any:
    """Step 2: Capital Allocation Settings."""
    st.subheader("Step 2: Capital Allocation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Fund Capital ($ millions):**")

        config.total_fund_capital = st.number_input(
            "Total Fund Capital",
            min_value=1.0,
            value=config.total_fund_capital,
            step=10.0,
            format="%.1f",
            help="Total capital available for allocation",
        )

        config.external_pa_capital = st.number_input(
            "External PA Capital [$M]",
            min_value=0.0,
            max_value=config.total_fund_capital,
            value=config.external_pa_capital,
            step=5.0,
            format="%.1f",
            help="Capital allocated to external portable alpha managers",
        )

        config.active_ext_capital = st.number_input(
            "Active Extension Capital [$M]",
            min_value=0.0,
            max_value=config.total_fund_capital,
            value=config.active_ext_capital,
            step=5.0,
            format="%.1f",
            help="Capital for active equity overlay strategies",
        )

        # Calculate remaining capital
        remaining = (
            config.total_fund_capital
            - config.external_pa_capital
            - config.active_ext_capital
        )
        config.internal_pa_capital = st.number_input(
            "Internal PA Capital [$M]",
            min_value=0.0,
            value=max(0.0, remaining),
            step=5.0,
            format="%.1f",
            help="Capital managed internally for portable alpha",
        )

    with col2:
        st.markdown("**Portfolio Weights & Shares:**")

        config.w_beta_h = st.slider(
            "Internal Beta Weight",
            min_value=0.0,
            max_value=1.0,
            value=config.w_beta_h,
            step=0.05,
            help="Beta component weight in internal sleeve",
        )

        config.w_alpha_h = 1.0 - config.w_beta_h
        st.write(f"Internal Alpha Weight: {config.w_alpha_h:.2f} (auto-calculated)")

        config.theta_extpa = st.slider(
            "External PA Alpha Fraction",
            min_value=0.0,
            max_value=1.0,
            value=config.theta_extpa,
            step=0.05,
            help="Fraction of alpha from external PA manager",
        )

        config.active_share = st.slider(
            "Active Extension Share",
            min_value=0.0,
            max_value=1.0,
            value=config.active_share,
            step=0.05,
            help=tooltip("active share"),
        )

    # Validation and visualization
    total_allocated = (
        config.external_pa_capital
        + config.active_ext_capital
        + config.internal_pa_capital
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
            index=["simple_proxy", "schedule"].index(
                ss.financing_settings["financing_model"]
            ),
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
                    config.total_fund_capital
                    if hasattr(config, "total_fund_capital")
                    else 1000.0
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
                    config.total_fund_capital
                    if hasattr(config, "total_fund_capital")
                    else 1000.0
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
            "Max CVaR",
            min_value=0.0,
            value=st.session_state.get("sleeve_max_cvar", 0.05),
            step=0.01,
            format="%.2f",
            key="sleeve_max_cvar",
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
        current_scope = st.session_state.get("sleeve_constraint_scope", "sleeves")
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

    constraints = {
        "max_te": float(st.session_state["sleeve_max_te"]),
        "max_breach": float(st.session_state["sleeve_max_breach"]),
        "max_cvar": float(st.session_state["sleeve_max_cvar"]),
        "step": float(st.session_state["sleeve_step"]),
        "max_evals": int(st.session_state["sleeve_max_evals"]),
        "constraint_scope": st.session_state["sleeve_constraint_scope"],
    }

    if st.button("Run Suggestor"):
        yaml_dict = _build_yaml_from_config(config)
        cfg = load_config(yaml_dict)
        idx_path = (
            Path(__file__).resolve().parents[2] / "data" / "sp500tr_fred_divyield.csv"
        )
        if not idx_path.exists():
            st.error(f"Default index file missing: {idx_path}")
            return
        idx_series = load_index_returns(idx_path)
        df = suggest_sleeve_sizes(
            cfg,
            idx_series,
            max_te=constraints["max_te"],
            max_breach=constraints["max_breach"],
            max_cvar=constraints["max_cvar"],
            step=constraints["step"],
            max_evals=constraints["max_evals"],
            constraint_scope=constraints["constraint_scope"],
        )
        st.session_state["sleeve_suggestions"] = df
        st.session_state["sleeve_suggestion_constraints"] = constraints

    constraints_used = st.session_state.get(
        "sleeve_suggestion_constraints", constraints
    )
    st.markdown("**Constraint Summary:**")
    st.write(
        f"Max TE: {constraints_used['max_te']:.2%} | "
        f"Max Breach: {constraints_used['max_breach']:.2%} | "
        f"Max CVaR: {constraints_used['max_cvar']:.2%} | "
        f"Scope: {scope_labels.get(constraints_used['constraint_scope'], constraints_used['constraint_scope'])}"
    )

    suggestions = st.session_state.get("sleeve_suggestions")
    if suggestions is None:
        return
    if suggestions.empty:
        st.warning("No feasible sleeve allocations found.")
        return

    ranked = suggestions.sort_values("risk_score", ascending=True).reset_index(
        drop=True
    )
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
        "ExternalPA_TE",
        "ExternalPA_BreachProb",
        "ExternalPA_CVaR",
        "ActiveExt_TE",
        "ActiveExt_BreachProb",
        "ActiveExt_CVaR",
        "InternalPA_TE",
        "InternalPA_BreachProb",
        "InternalPA_CVaR",
        "Total_TE",
        "Total_BreachProb",
        "Total_CVaR",
    ]
    display_cols = [col for col in preferred_cols if col in ranked.columns]
    tradeoff_table = ranked.loc[:, display_cols].head(int(top_n))
    st.dataframe(tradeoff_table, use_container_width=True)

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

    selected_metrics = st.multiselect(
        "Select Risk Metrics",
        options=list(metric_options.keys()),
        default=config.risk_metrics,
        format_func=lambda x: metric_options[x],
        help="Choose which risk metrics to calculate and display",
    )

    config.risk_metrics = selected_metrics

    st.markdown("---")
    _render_sleeve_suggestor(config)

    return config


def _render_step_4_correlations(config: Any) -> Any:
    """Step 4: Correlation Parameters."""
    st.subheader("Step 4: Correlation Parameters")

    st.markdown(
        "*Set correlations between different alpha sources and the market index*"
    )

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
            f"• External PA: ${config.external_pa_capital:.1f}M ({config.external_pa_capital/total_capital:.1%})"
        )
        st.write(
            f"• Active Extension: ${config.active_ext_capital:.1f}M ({config.active_ext_capital/total_capital:.1%})"
        )
        st.write(
            f"• Internal PA: ${config.internal_pa_capital:.1f}M ({config.internal_pa_capital/total_capital:.1%})"
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
        st.write(
            f"• Reference sigma (monthly): {float(fs.get('reference_sigma', 0.01)):.4f}"
        )
        if fm == "simple_proxy":
            st.write(
                f"• Volatility multiple (k): {float(fs.get('volatility_multiple', 3.0)):.2f}"
            )
        else:
            st.write(f"• Term (months): {float(fs.get('term_months', 1.0)):.1f}")
            st.write(
                f"• Schedule file: {'set' if fs.get('schedule_path') else 'not set'}"
            )

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
        if st.button(
            "🔄 Reset to Defaults", help="Reset all parameters to sensible defaults"
        ):
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
                # Convert uploaded config to wizard format
                wizard_config = WizardScenarioConfig.model_validate(
                    {
                        "analysis_mode": config_data.get("analysis_mode", "returns"),
                        "n_simulations": config_data.get("N_SIMULATIONS", 1000),
                        "n_months": config_data.get("N_MONTHS", 12),
                        "total_fund_capital": config_data.get(
                            "total_fund_capital", 300.0
                        ),
                        "external_pa_capital": config_data.get(
                            "external_pa_capital", 100.0
                        ),
                        "active_ext_capital": config_data.get(
                            "active_ext_capital", 50.0
                        ),
                        "internal_pa_capital": config_data.get(
                            "internal_pa_capital", 150.0
                        ),
                        "w_beta_h": config_data.get("w_beta_H", 0.5),
                        "w_alpha_h": config_data.get("w_alpha_H", 0.5),
                        "theta_extpa": config_data.get("theta_extpa", 0.5),
                        "active_share": config_data.get("active_share", 0.5),
                        "mu_h": config_data.get("mu_H", 0.04),
                        "sigma_h": config_data.get("sigma_H", 0.01),
                        "mu_e": config_data.get("mu_E", 0.05),
                        "sigma_e": config_data.get("sigma_E", 0.02),
                        "mu_m": config_data.get("mu_M", 0.03),
                        "sigma_m": config_data.get("sigma_M", 0.02),
                        "rho_idx_h": config_data.get("rho_idx_H", 0.05),
                        "rho_idx_e": config_data.get("rho_idx_E", 0.0),
                        "rho_idx_m": config_data.get("rho_idx_M", 0.0),
                        "rho_h_e": config_data.get("rho_H_E", 0.1),
                        "rho_h_m": config_data.get("rho_H_M", 0.1),
                        "rho_e_m": config_data.get("rho_E_M", 0.0),
                        "internal_financing_mean_month": config_data.get(
                            "internal_financing_mean_month", 0.0
                        ),
                        "internal_financing_sigma_month": config_data.get(
                            "internal_financing_sigma_month", 0.0
                        ),
                        "internal_spike_prob": config_data.get(
                            "internal_spike_prob", 0.0
                        ),
                        "internal_spike_factor": config_data.get(
                            "internal_spike_factor", 0.0
                        ),
                        "ext_pa_financing_mean_month": config_data.get(
                            "ext_pa_financing_mean_month", 0.0
                        ),
                        "ext_pa_financing_sigma_month": config_data.get(
                            "ext_pa_financing_sigma_month", 0.0
                        ),
                        "ext_pa_spike_prob": config_data.get("ext_pa_spike_prob", 0.0),
                        "ext_pa_spike_factor": config_data.get(
                            "ext_pa_spike_factor", 0.0
                        ),
                        "act_ext_financing_mean_month": config_data.get(
                            "act_ext_financing_mean_month", 0.0
                        ),
                        "act_ext_financing_sigma_month": config_data.get(
                            "act_ext_financing_sigma_month", 0.0
                        ),
                        "act_ext_spike_prob": config_data.get(
                            "act_ext_spike_prob", 0.0
                        ),
                        "act_ext_spike_factor": config_data.get(
                            "act_ext_spike_factor", 0.0
                        ),
                        "risk_metrics": config_data.get(
                            "risk_metrics", ["Return", "Risk", "ShortfallProb"]
                        ),
                    }
                )
                financing_model = config_data.get("financing_model", "simple_proxy")
                st.session_state.financing_settings = {
                    "financing_model": financing_model,
                    "reference_sigma": config_data.get("reference_sigma", 0.01),
                    "volatility_multiple": config_data.get(
                        "volatility_multiple", 3.0
                    ),
                    "term_months": config_data.get(
                        "financing_term_months", config_data.get("term_months", 1.0)
                    ),
                    "schedule_path": config_data.get("financing_schedule_path"),
                }
                st.session_state.wizard_config = wizard_config
                st.session_state.wizard_step = 5  # Go to review step
                st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error loading config: {e}")

    # Main wizard interface
    current_step = st.session_state.wizard_step
    config = st.session_state.wizard_config

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

                        args = [
                            "--config",
                            cfg_path,
                            "--index",
                            idx_path,
                            "--output",
                            output,
                        ]
                        use_seed = st.session_state.get("wizard_use_seed", True)
                        seed_value = st.session_state.get("wizard_seed", 42)
                        if use_seed:
                            args.extend(["--seed", str(int(seed_value))])

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
                        st.success(
                            f"✅ Simulation complete! Results written to {output}"
                        )
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
                                total_capital=float(
                                    yaml_data.get("total_fund_capital", 1000.0)
                                ),
                                financing_model=fm,
                                schedule_path=Path(sched_path) if sched_path else None,
                                term_months=term_m,
                            )
                            st.info(
                                f"Margin requirement: ${margin_requirement:.1f}M (model: {fm})"
                            )
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
