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

from dashboard.app import _DEF_THEME, apply_theme
from pa_core.config import ModelConfig
from pa_core.orchestrator import SimulatorOrchestrator
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
    theme_path = st.sidebar.text_input("Theme file", _DEF_THEME)
    apply_theme(theme_path)

    st.markdown("Pick a preset, run base vs stressed with identical seeds, and review deltas.")

    idx_file = st.file_uploader(
        "Index returns CSV (single numeric column)", type=["csv"], key="stress_idx"
    )
    if idx_file is None:
        st.info("Provide index CSV to simulate.")
        return

    seed = st.number_input("Random seed", value=42, step=1)

    # Base configuration controls (minimal for now)
    st.sidebar.subheader("Base configuration")
    n_sims = st.sidebar.number_input("Number of simulations", value=1000, step=100)
    n_months = st.sidebar.number_input("Number of months", value=12, step=1)

    # Capital breakout (kept simple but non-zero to activate agents)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Capital ($mm)**")
    total_cap = st.sidebar.number_input("Total fund capital", value=1000.0, step=50.0)
    ext_cap = st.sidebar.number_input(
        "External PA capital",
        value=200.0,
        step=10.0,
        min_value=0.0,
        max_value=total_cap,
    )
    act_cap = st.sidebar.number_input(
        "Active Extension capital",
        value=200.0,
        step=10.0,
        min_value=0.0,
        max_value=total_cap,
    )
    int_cap = st.sidebar.number_input(
        "Internal PA capital",
        value=200.0,
        step=10.0,
        min_value=0.0,
        max_value=total_cap,
    )

    theta = st.sidebar.slider(
        "External PA alpha fraction θ",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )
    active_share = st.sidebar.slider(
        "Active share", min_value=0.0, max_value=1.0, value=0.5, step=0.05
    )

    preset_names = sorted(STRESS_PRESETS.keys())
    preset = st.selectbox(
        "Stress preset",
        options=preset_names,
        index=(preset_names.index("2008_vol_regime") if "2008_vol_regime" in preset_names else 0),
        format_func=lambda key: STRESS_PRESET_LABELS.get(key, key),
    )

    if st.button("Run stress test"):
        try:
            index_series = _read_index_csv(idx_file)

            base_cfg = ModelConfig.model_validate(
                {
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
            stress_workbook = build_stress_workbook(base_summary, stress_summary, delta_df, diff_df)
            st.download_button(
                "Download stress results (Excel)",
                stress_workbook,
                "stress_results.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        except Exception as exc:  # pragma: no cover - runtime UX
            st.error(str(exc))


if __name__ == "__main__":  # pragma: no cover
    main()
