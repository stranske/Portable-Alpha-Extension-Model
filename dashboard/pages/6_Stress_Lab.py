"""Stress Lab: apply built-in stress presets and compare deltas.

This page lets users pick a stress preset (from pa_core.stress.STRESS_PRESETS),
run a base vs. stressed simulation using the same seed, and see a side-by-side
summary plus a delta table (stressed - base). It reuses the orchestrator and
summary_table utilities from pa_core.
"""

from __future__ import annotations

import io
from typing import cast

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from dashboard.app import apply_theme, render_settings_sidebar
from dashboard.utils import (
    build_sample_model_config,
    config_capital_defaults,
    current_index_returns,
    current_scenario_config,
    load_bundled_sample_index,
)
from pa_core.config import ModelConfig
from pa_core.orchestrator import SimulatorOrchestrator
from pa_core.reporting.constraints import build_constraint_report
from pa_core.reporting.stress_delta import (
    build_delta_table,
    build_stress_workbook,
    format_delta_table,
)
from pa_core.stress import STRESS_PRESET_LABELS, STRESS_PRESETS, apply_stress_preset


def _read_index_csv(file) -> pd.Series:
    data = file.getvalue() if hasattr(file, "getvalue") else file.read()
    df = pd.read_csv(io.BytesIO(data))
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError(
            "No numeric columns found in index CSV. Please ensure the file contains at least one column with numeric data."
        )
    return pd.Series(df[num_cols[0]].dropna().to_numpy())


def _config_diff(base: ModelConfig, stressed: ModelConfig) -> pd.DataFrame:
    """Return DataFrame of parameters that differ between two configs."""
    base_d = base.model_dump()
    stress_d = stressed.model_dump()
    ignore_keys = {"agents"}
    rows = []
    # Check all keys from both configurations to catch removed parameters
    for key in sorted(set(base_d.keys()) | set(stress_d.keys())):
        if key in ignore_keys:
            continue
        b_val = base_d.get(key)
        s_val = stress_d.get(key)
        if b_val != s_val:
            rows.append({"Parameter": key, "Base": b_val, "Stressed": s_val})
    return pd.DataFrame(rows)


def main() -> None:
    st.title("Stress Lab (presets)")
    _, theme_path = render_settings_sidebar()
    apply_theme(theme_path)

    st.markdown("Pick a preset, run base vs stressed with identical seeds, and review deltas.")
    session_cfg = current_scenario_config(st.session_state)
    session_index = current_index_returns(st.session_state)
    capital_defaults = config_capital_defaults(session_cfg)

    idx_file = st.file_uploader(
        "Index returns CSV (single numeric column)", type=["csv"], key="stress_idx"
    )
    use_sample = st.checkbox(
        "Use bundled sample data (no upload needed)",
        value=False,
        key="stress_use_sample",
        help="Loads the bundled S&P 500 TR / FRED dividend-yield series for a one-click demo.",
    )
    if idx_file is None and not use_sample and session_index is None:
        st.info(
            "Upload an index CSV, tick 'Use bundled sample data', or run a scenario first to simulate."
        )
        return

    seed = st.number_input("Random seed", value=42, step=1)

    # Base configuration controls (minimal for now)
    st.sidebar.subheader("Base configuration")
    n_sims_default = int(session_cfg.N_SIMULATIONS if session_cfg else 1000)
    n_months_default = int(session_cfg.N_MONTHS if session_cfg else 12)
    n_sims = st.sidebar.number_input("Number of simulations", value=n_sims_default, step=100)
    n_months = st.sidebar.number_input("Number of months", value=n_months_default, step=1)

    # Capital breakout (kept simple but non-zero to activate agents)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Capital ($mm)**")
    total_cap = st.sidebar.number_input(
        "Total fund capital",
        value=capital_defaults["total_fund_capital"],
        step=50.0,
    )
    ext_cap = st.sidebar.number_input(
        "External PA capital",
        value=min(capital_defaults["external_pa_capital"], total_cap),
        step=10.0,
        min_value=0.0,
        max_value=total_cap,
    )
    act_cap = st.sidebar.number_input(
        "Active Extension capital",
        value=min(capital_defaults["active_ext_capital"], total_cap),
        step=10.0,
        min_value=0.0,
        max_value=total_cap,
    )
    int_cap = st.sidebar.number_input(
        "Internal PA capital",
        value=min(capital_defaults["internal_pa_capital"], total_cap),
        step=10.0,
        min_value=0.0,
        max_value=total_cap,
    )

    theta = st.sidebar.slider(
        "External PA alpha fraction θ",
        min_value=0.0,
        max_value=1.0,
        value=float(session_cfg.theta_extpa if session_cfg else 0.5),
        step=0.05,
    )
    active_share = st.sidebar.slider(
        "Active share",
        min_value=0.0,
        max_value=1.0,
        value=float(session_cfg.active_share if session_cfg else 0.5),
        step=0.05,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Constraint limits**")
    max_te = st.sidebar.number_input("Max tracking error", value=0.02, step=0.005, format="%.3f")
    max_breach = st.sidebar.number_input(
        "Max breach probability", value=0.5, step=0.05, format="%.2f"
    )
    max_cvar = st.sidebar.number_input("Max monthly CVaR", value=0.05, step=0.01, format="%.3f")

    preset_names = sorted(STRESS_PRESETS.keys())
    preset = st.selectbox(
        "Stress preset",
        options=preset_names,
        index=(preset_names.index("2008_vol_regime") if "2008_vol_regime" in preset_names else 0),
        format_func=lambda key: STRESS_PRESET_LABELS.get(key, key),
    )

    if st.button("Run stress test"):
        try:
            if idx_file is not None:
                index_series = _read_index_csv(idx_file)
            elif use_sample:
                index_series = load_bundled_sample_index()
            else:
                index_series = session_index
            if index_series is None:
                raise ValueError(
                    "Provide an index CSV, enable bundled sample data, or run a scenario first."
                )

            base_cfg = build_sample_model_config(
                **{
                    "Number of simulations": int(n_sims),
                    "Number of months": int(n_months),
                    "Total fund capital (mm)": float(total_cap),
                    "External PA capital (mm)": float(ext_cap),
                    "Active Extension capital (mm)": float(act_cap),
                    "Internal PA capital (mm)": float(int_cap),
                    "External PA alpha fraction": float(theta),
                    "Active share": float(active_share),
                }
            )

            stressed_cfg = apply_stress_preset(base_cfg, cast(str, preset))

            # Run orchestrations with identical seed
            base_orch = SimulatorOrchestrator(base_cfg, index_series)
            _, base_summary = base_orch.run(seed=int(seed))

            stress_orch = SimulatorOrchestrator(stressed_cfg, index_series)
            _, stress_summary = stress_orch.run(seed=int(seed))

            st.subheader("Summary (Base vs Stressed)")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Base**")
                st.dataframe(base_summary)
            with col2:
                st.markdown("**Stressed**")
                st.dataframe(stress_summary)

            st.subheader("Constraint breaches (Base vs Stressed)")
            base_constraints = build_constraint_report(
                base_summary,
                max_te=float(max_te),
                max_breach=float(max_breach),
                max_cvar=float(max_cvar),
            )
            stress_constraints = build_constraint_report(
                stress_summary,
                max_te=float(max_te),
                max_breach=float(max_breach),
                max_cvar=float(max_cvar),
            )
            breach_col1, breach_col2 = st.columns(2)
            with breach_col1:
                st.markdown("**Base**")
                if base_constraints.empty:
                    st.info("No base constraint breaches.")
                else:
                    st.dataframe(base_constraints)
            with breach_col2:
                st.markdown("**Stressed**")
                if stress_constraints.empty:
                    st.info("No stressed constraint breaches.")
                else:
                    st.dataframe(stress_constraints)

            st.subheader("Delta (Stressed - Base)")
            delta_df = build_delta_table(base_summary, stress_summary)
            if delta_df.empty:
                st.info("No shared numeric metrics to compare.")
            else:
                st.dataframe(format_delta_table(delta_df))
            st.download_button(
                "Download deltas as CSV",
                delta_df.to_csv(index=False).encode("utf-8"),
                "stress_deltas.csv",
                "text/csv",
            )

            st.subheader("Config diff")
            diff_df = _config_diff(base_cfg, stressed_cfg)
            st.dataframe(diff_df)
            st.download_button(
                "Download config diff",
                diff_df.to_csv(index=False).encode("utf-8"),
                "config_diff.csv",
                "text/csv",
            )
            stress_workbook = build_stress_workbook(
                base_summary,
                stress_summary,
                delta_df,
                diff_df,
                base_constraints,
                stress_constraints,
            )
            st.download_button(
                "Download stress results (Excel)",
                stress_workbook,
                "stress_results.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        except ValidationError:  # pragma: no cover - runtime UX
            st.error(
                "The sample stress-test settings could not be validated. "
                "Refresh the defaults or run a scenario first."
            )
        except Exception as exc:  # pragma: no cover - runtime UX
            st.error(str(exc))


if __name__ == "__main__":  # pragma: no cover
    main()
