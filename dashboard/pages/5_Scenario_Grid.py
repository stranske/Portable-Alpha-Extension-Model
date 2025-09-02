"""Placeholder for Scenario Grid & Frontier page.

This branch focuses on Stress Lab UI (#382). The full Scenario Grid implementation
is available on its feature branch. This stub exists so dashboard tests that import
pages continue to pass.
"""

from __future__ import annotations

import io

import pandas as pd  # type: ignore[reportMissingImports]
import yaml
import streamlit as st

from dashboard.app import _DEF_THEME, apply_theme
from pa_core.viz import grid_heatmap
from pa_core.config import ModelConfig
from pa_core.sweep import run_parameter_sweep_cached, sweep_results_to_dataframe


def _read_csv(file) -> pd.DataFrame:
    try:
        data = file.getvalue() if hasattr(file, "getvalue") else file.read()
        return pd.read_csv(io.BytesIO(data))
    except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:  # pragma: no cover - user input variability
        st.error(f"Failed to parse CSV: {exc}")
        return pd.DataFrame()


def main() -> None:
    st.title("Scenario Grid & Frontier (beta)")
    theme_path = st.sidebar.text_input("Theme file", _DEF_THEME)
    apply_theme(theme_path)

    tab1, tab2 = st.tabs(["Upload Grid CSV", "Compute from Config"]) 

    with tab1:
        st.markdown(
            "Upload a grid CSV with columns `AE_leverage`, `ExtPA_frac`, `Sharpe` to preview the heatmap."
        )
        up = st.file_uploader("Grid CSV", type=["csv"], key="grid_csv")
        if up is not None:
            df = _read_csv(up)
            required = {"AE_leverage", "ExtPA_frac", "Sharpe"}
            if required.issubset(df.columns):
                fig = grid_heatmap.make(df, x="AE_leverage", y="ExtPA_frac", z="Sharpe")
                # Frontier overlay: for each AE_leverage, pick ExtPA_frac with max Sharpe
                try:
                    pivot = df.pivot(index="ExtPA_frac", columns="AE_leverage", values="Sharpe").sort_index().sort_index(axis=1)
                    xs = []
                    ys = []
                    for col in pivot.columns:
                        col_series = pivot[col]
                        if col_series.notna().any():
                            y_best = float(col_series.idxmax())
                            xs.append(float(col))
                            ys.append(y_best)
                    if xs and ys:
                        fig.add_scatter(x=xs, y=ys, mode="lines+markers", name="Frontier", line=dict(color="white", width=2), marker=dict(color="white"))
                except (ValueError, KeyError) as exc:
                    st.error(f"Failed to compute frontier overlay: {exc}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"CSV must include columns: {sorted(required)}")
        else:
            st.info("Provide a CSV to render the grid heatmap.")

    with tab2:
        st.markdown("Compute an alpha-shares grid via Monte Carlo sweep.")
        idx_file = st.file_uploader("Index returns CSV (single numeric column)", type=["csv"], key="idx_csv")
        seed = st.number_input("Random seed", value=42, step=1)

        # Optional step controls
        st.sidebar.markdown("---")
        st.sidebar.subheader("Sweep steps (alpha shares)")
        ext_min = st.sidebar.number_input("ExtPA alpha min [%]", value=25.0, step=5.0)
        ext_max = st.sidebar.number_input("ExtPA alpha max [%]", value=75.0, step=5.0)
        ext_step = st.sidebar.number_input("ExtPA alpha step [%]", value=5.0, step=1.0)
        act_min = st.sidebar.number_input("Active share min [%]", value=20.0, step=5.0)
        act_max = st.sidebar.number_input("Active share max [%]", value=100.0, step=5.0)
        act_step = st.sidebar.number_input("Active share step [%]", value=5.0, step=1.0)

        if idx_file is not None and st.button("Run sweep"):
            try:
                idx_df = _read_csv(idx_file)
                # Use first numeric column as index series
                num_cols = [c for c in idx_df.columns if pd.api.types.is_numeric_dtype(idx_df[c])]
                if not num_cols:
                    st.error("No numeric column found in index CSV")
                    return
                # Ensure a plain pandas Series for type-checker
                index_series = pd.Series(idx_df[num_cols[0]].dropna().to_numpy())

                base_cfg = ModelConfig.model_validate({
                    "Number of simulations": 1000,
                    "Number of months": 12,
                })
                cfg = base_cfg.model_copy(update={
                    "analysis_mode": "alpha_shares",
                    "external_pa_alpha_min_pct": float(ext_min),
                    "external_pa_alpha_max_pct": float(ext_max),
                    "external_pa_alpha_step_pct": float(ext_step),
                    "active_share_min_pct": float(act_min),
                    "active_share_max_pct": float(act_max),
                    "active_share_step_pct": float(act_step),
                })

                prog = st.progress(0.0)

                def _update(i: int, total: int) -> None:
                    prog.progress(i / total)

                results = run_parameter_sweep_cached(
                    cfg, index_series, int(seed), progress=_update
                )
                prog.empty()
                df_res = sweep_results_to_dataframe(results)
                # Focus on Base agent and compute Sharpe
                base_rows = df_res[df_res["Agent"] == "Base"].copy()
                if base_rows.empty:
                    st.warning("No base agent rows found in results.")
                    return
                base_rows["Sharpe"] = base_rows.apply(
                    lambda r: (r["AnnReturn"] / r["AnnVol"]) if r["AnnVol"] else 0.0, axis=1
                )
                # Use native parameter names for axes
                grid_df = base_rows[["active_share", "theta_extpa", "Sharpe"]].copy()
                # Help static checker understand this is a DataFrame
                from typing import cast
                fig2 = grid_heatmap.make(cast(pd.DataFrame, grid_df), x="active_share", y="theta_extpa", z="Sharpe")  # type: ignore[arg-type]
                # Frontier overlay: best theta per active_share
                try:
                    pv = grid_df.pivot(index="theta_extpa", columns="active_share", values="Sharpe").sort_index().sort_index(axis=1)
                    xs2: list[float] = []
                    ys2: list[float] = []
                    for col in pv.columns:
                        col_series = pv[col]
                        if col_series.notna().any():
                            y_best = float(col_series.idxmax())
                            xs2.append(float(col))
                            ys2.append(y_best)
                    if xs2 and ys2:
                        fig2.add_scatter(x=xs2, y=ys2, mode="lines+markers", name="Frontier", line=dict(color="white", width=2), marker=dict(color="white"))
                except Exception as exc:
                    st.error(f"Frontier overlay failed: {exc}")
                st.plotly_chart(fig2, use_container_width=True)

                # Simple promote: allow choosing a point from available values
                st.subheader("Promote selection")
                x_vals = sorted(grid_df["active_share"].unique().tolist())
                y_vals = sorted(grid_df["theta_extpa"].unique().tolist())
                sel_x = st.selectbox("Active share", options=x_vals)
                sel_y = st.selectbox("External PA alpha fraction", options=y_vals)
                if st.button("Promote to session"):
                    st.session_state["promoted_alpha_shares"] = {
                        "active_share": float(sel_x),
                        "theta_extpa": float(sel_y),
                    }
                    st.success("Selection promoted. Other pages can read session_state['promoted_alpha_shares'].")
                    st.page_link("pages/2_Portfolio_Builder.py", label="→ Go to Portfolio Builder")

                    # Offer a ready-to-run scenario YAML with promoted parameters
                    cfg_yaml = {
                        "Number of simulations": 1000,
                        "Number of months": 12,
                        "analysis_mode": "alpha_shares",
                        "theta_extpa": float(sel_y),
                        "active_share": float(sel_x),
                        "risk_metrics": ["Return", "Risk", "ShortfallProb"],
                    }
                    yaml_str = yaml.safe_dump(cfg_yaml, sort_keys=False)
                    st.download_button(
                        "Download Scenario YAML",
                        yaml_str,
                        file_name="scenario_alpha_shares.yml",
                        mime="application/x-yaml",
                    )
            except ValueError as exc:
                st.error(f"Invalid configuration: {exc}")
            except RuntimeError as exc:
                st.error(f"Simulation failed: {exc}")


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
