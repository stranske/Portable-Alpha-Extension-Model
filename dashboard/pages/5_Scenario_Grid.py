"""Scenario Grid & Frontier interactive page.

Allows exploring alpha-share configurations via CSV upload or Monte Carlo sweep,
renders a frontier overlay and supports promoting selections to other pages.
"""

from __future__ import annotations

import io

import pandas as pd  # type: ignore[reportMissingImports]
import streamlit as st
import yaml

from dashboard.app import _DEF_THEME, apply_theme
from dashboard.utils import (
    build_alpha_shares_payload,
    bump_session_token,
    make_grid_cache_key,
)
from pa_core.config import ModelConfig
from pa_core.sweep import run_parameter_sweep_cached, sweep_results_to_dataframe
from pa_core.viz import grid_heatmap

# Cached empty DataFrame to avoid repeated creation in error cases
_EMPTY_DATAFRAME: pd.DataFrame | None = None
_GRID_CACHE_KEY = "scenario_grid_cache"
_GRID_PROMOTION_NOTICE_KEY = "scenario_grid_promotion_notice"
_GRID_CACHE_LIMIT = 3
_EXTRA_METRICS = [
    "AnnReturn",
    "AnnVol",
    "TE",
    "CVaR",
    "BreachProb",
    "ShortfallProb",
]


def _get_empty_dataframe() -> pd.DataFrame:
    """Return a cached empty DataFrame to avoid repeated object creation."""
    global _EMPTY_DATAFRAME
    if _EMPTY_DATAFRAME is None:
        _EMPTY_DATAFRAME = pd.DataFrame()
    return _EMPTY_DATAFRAME


def _get_grid_cache(cache_key: str) -> dict | None:
    cached = st.session_state.get(_GRID_CACHE_KEY)
    if isinstance(cached, dict):
        if cached.get("key") == cache_key:
            return cached
        entries = cached.get("entries")
        if isinstance(entries, dict) and cache_key in entries:
            order = cached.get("order")
            if isinstance(order, list):
                if cache_key in order:
                    order.remove(cache_key)
                order.append(cache_key)
            return entries[cache_key]
    return None


def _set_grid_cache(
    cache_key: str, grid_df: pd.DataFrame, y_col: str, total_fund: float
) -> None:
    entry = {
        "key": cache_key,
        "grid_df": grid_df,
        "y_col": y_col,
        "total_fund": total_fund,
    }
    cached = st.session_state.get(_GRID_CACHE_KEY)
    if not isinstance(cached, dict) or "entries" not in cached:
        cached = {"entries": {}, "order": []}
    entries = cached.get("entries")
    order = cached.get("order")
    if not isinstance(entries, dict) or not isinstance(order, list):
        cached = {"entries": {}, "order": []}
        entries = cached["entries"]
        order = cached["order"]
    entries[cache_key] = entry
    if cache_key in order:
        order.remove(cache_key)
    order.append(cache_key)
    while len(order) > _GRID_CACHE_LIMIT:
        oldest = order.pop(0)
        entries.pop(oldest, None)
    st.session_state[_GRID_CACHE_KEY] = cached


def _read_csv(file) -> pd.DataFrame:
    try:
        data = file.getvalue() if hasattr(file, "getvalue") else file.read()
        return pd.read_csv(io.BytesIO(data))
    except (
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:  # pragma: no cover - user input variability
        st.error(f"Failed to parse CSV: {exc}")
        return _get_empty_dataframe()


def _extract_plotly_click(selection) -> tuple[float, float] | None:
    if not isinstance(selection, dict):
        return None
    selection_state = selection.get("selection", {})
    points = selection_state.get("points", [])
    if not points:
        return None
    point = points[-1]
    try:
        return float(point["x"]), float(point["y"])
    except (TypeError, ValueError, KeyError):
        return None


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
                    pivot = (
                        df.pivot(
                            index="ExtPA_frac", columns="AE_leverage", values="Sharpe"
                        )
                        .sort_index()
                        .sort_index(axis=1)
                    )
                    xs = []
                    ys = []
                    for col in pivot.columns:
                        col_series = pivot[col]
                        if col_series.notna().any():
                            y_best = float(col_series.idxmax())
                            xs.append(float(col))
                            ys.append(y_best)
                    if xs and ys:
                        fig.add_scatter(
                            x=xs,
                            y=ys,
                            mode="lines+markers",
                            name="Frontier",
                            line=dict(color="white", width=2),
                            marker=dict(color="white"),
                        )
                except (ValueError, KeyError) as exc:
                    st.error(f"Failed to compute frontier overlay: {exc}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"CSV must include columns: {sorted(required)}")
        else:
            st.info("Provide a CSV to render the grid heatmap.")

    with tab2:
        st.markdown("Compute an alpha-shares grid via Monte Carlo sweep.")
        idx_file = st.file_uploader(
            "Index returns CSV (single numeric column)", type=["csv"], key="idx_csv"
        )
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

        st.sidebar.markdown("---")
        st.sidebar.subheader("Axis options")
        y_axis_mode = st.sidebar.selectbox(
            "Y axis",
            options=["External alpha fraction (theta)", "External PA $ (mm)"],
            index=0,
        )

        promotion_notice = st.session_state.pop(_GRID_PROMOTION_NOTICE_KEY, None)
        if promotion_notice:
            st.success(promotion_notice)
            st.page_link(
                "pages/2_Portfolio_Builder.py",
                label="→ Go to Portfolio Builder",
            )

        run_sweep = st.button("Run sweep")
        if idx_file is None:
            st.info("Upload an index returns CSV to run the sweep.")
            return

        try:
            idx_df = _read_csv(idx_file)
            # Use first numeric column as index series
            num_cols = [
                c for c in idx_df.columns if pd.api.types.is_numeric_dtype(idx_df[c])
            ]
            if not num_cols:
                st.error("No numeric column found in index CSV")
                return
            # Ensure a plain pandas Series for type-checker
            index_series = pd.Series(idx_df[num_cols[0]].dropna().to_numpy())

            base_cfg = ModelConfig.model_validate(
                {
                    "Number of simulations": 1000,
                    "Number of months": 12,
                }
            )
            cfg = base_cfg.model_copy(
                update={
                    "analysis_mode": "alpha_shares",
                    "external_pa_alpha_min_pct": float(ext_min),
                    "external_pa_alpha_max_pct": float(ext_max),
                    "external_pa_alpha_step_pct": float(ext_step),
                    "active_share_min_pct": float(act_min),
                    "active_share_max_pct": float(act_max),
                    "active_share_step_pct": float(act_step),
                }
            )

            cache_key = make_grid_cache_key(cfg, index_series, int(seed), y_axis_mode)
            cached = _get_grid_cache(cache_key)
            if cached is None and not run_sweep:
                st.info("Run the sweep to generate the grid.")
                return

            if cached is None:
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
                    lambda r: (r["AnnReturn"] / r["AnnVol"]) if r["AnnVol"] else 0.0,
                    axis=1,
                )
                # Compute external PA dollars (mm) if total fund capital present
                if "external_pa_capital" in base_rows.columns:
                    base_rows["external_pa_dollars_mm"] = base_rows[
                        "external_pa_capital"
                    ].astype(float)
                elif hasattr(base_cfg, "total_fund_capital"):
                    # Derive from theta and total fund capital when available
                    try:
                        if "theta_extpa" in base_rows.columns:
                            theta_series = base_rows["theta_extpa"].astype(float)
                        else:
                            theta_series = pd.Series(
                                0.0, index=base_rows.index, dtype=float
                            )
                        base_rows["external_pa_dollars_mm"] = theta_series * float(
                            getattr(base_cfg, "total_fund_capital")
                        )
                    except Exception:
                        base_rows["external_pa_dollars_mm"] = float("nan")

                # Decide y-axis based on selection
                if (
                    y_axis_mode.startswith("External PA $")
                    and "external_pa_dollars_mm" in base_rows.columns
                ):
                    y_col = "external_pa_dollars_mm"
                else:
                    y_col = "theta_extpa"

                # Use native parameter names for axes and attach metrics
                grid_df = base_rows[["active_share", y_col, "Sharpe"]].copy()
                extra_cols = [c for c in _EXTRA_METRICS if c in base_rows.columns]
                # Append extra columns into grid for custom hover
                if extra_cols:
                    grid_df = pd.concat([grid_df, base_rows[extra_cols]], axis=1)
                grid_df = grid_df.rename(columns={y_col: "y_axis"})
                total_fund = float(getattr(base_cfg, "total_fund_capital", 1.0))
                _set_grid_cache(cache_key, grid_df, y_col, total_fund)
                cached = {"grid_df": grid_df, "y_col": y_col, "total_fund": total_fund}

            grid_df = cached["grid_df"]
            y_col = cached["y_col"]
            total_fund = float(cached.get("total_fund", 1.0))
            extra_cols = [c for c in _EXTRA_METRICS if c in grid_df.columns]

            # Help static checker understand this is a DataFrame
            from typing import cast

            # Build heatmap with custom fields for richer hover
            fig2 = grid_heatmap.make(
                cast(pd.DataFrame, grid_df),
                x="active_share",
                y="y_axis",
                z="Sharpe",
                custom_fields=extra_cols,
            )  # type: ignore[arg-type]
            # Customize hover to show metrics; use %{customdata[i]}
            hover_lines = [
                "active_share=%{x:.2f}",
                (
                    "external_pa_dollars_mm=%{y:.2f}"
                    if y_col == "external_pa_dollars_mm"
                    else "theta_extpa=%{y:.2f}"
                ),
                "Sharpe=%{z:.2f}",
            ]
            # Map extra field labels
            field_map = {
                "AnnReturn": "AnnReturn",
                "AnnVol": "AnnVol",
                "TE": "TE",
                "CVaR": "CVaR",
                "BreachProb": "BreachProb",
                "ShortfallProb": "ShortfallProb",
            }
            for i, field in enumerate(extra_cols):
                label = field_map.get(field, field)
                hover_lines.append(f"{label}=%{{customdata[{i}]}}")
            try:
                fig2.update_traces(hovertemplate="<br>".join(hover_lines))
            except Exception:
                pass
            # Frontier overlay: best theta per active_share
            try:
                pv = (
                    grid_df.rename(columns={"y_axis": y_col})
                    .pivot(index=y_col, columns="active_share", values="Sharpe")
                    .sort_index()
                    .sort_index(axis=1)
                )
                xs2: list[float] = []
                ys2: list[float] = []
                for col in pv.columns:
                    col_series = pv[col]
                    if col_series.notna().any():
                        y_best = float(col_series.idxmax())
                        xs2.append(float(col))
                        ys2.append(y_best)
                if xs2 and ys2:
                    fig2.add_scatter(
                        x=xs2,
                        y=ys2,
                        mode="lines+markers",
                        name="Frontier",
                        line=dict(color="white", width=2),
                        marker=dict(color="white"),
                    )
            except Exception as exc:
                st.error(f"Frontier overlay failed: {exc}")
            selection = st.plotly_chart(
                fig2,
                use_container_width=True,
                key="scenario_grid_heatmap",
                on_select="rerun",
                selection_mode="points",
            )
            clicked = _extract_plotly_click(selection)
            if clicked is not None:
                sel_x, sel_y = clicked
                if total_fund == 0:
                    total_fund = 1.0
                theta_val = (
                    float(sel_y) / total_fund
                    if y_col == "external_pa_dollars_mm"
                    else float(sel_y)
                )
                selection_payload = build_alpha_shares_payload(
                    float(sel_x), float(theta_val)
                )
                if selection_payload is not None:
                    st.session_state["scenario_grid_selection"] = selection_payload
                    st.session_state["promoted_alpha_shares"] = selection_payload
                    bump_session_token(
                        st.session_state, "scenario_grid_promotion_token"
                    )
                    st.success(
                        "Heatmap selection captured. Open Portfolio Builder to "
                        "see the values populated."
                    )
                    st.page_link(
                        "pages/2_Portfolio_Builder.py",
                        label="→ Go to Portfolio Builder",
                    )

            # Simple promote: allow choosing a point from available values
            st.subheader("Promote selection")
            x_vals = sorted(grid_df["active_share"].unique().tolist())
            y_vals = sorted(grid_df["y_axis"].unique().tolist())
            if not x_vals or not y_vals:
                st.warning("No grid points available to promote.")
                return
            sel_x = st.selectbox("Active share", options=x_vals, index=0)
            sel_y = st.selectbox(
                "External PA value",
                options=y_vals,
                index=0,
                format_func=(
                    lambda v: (
                        f"${v:,.0f} mm"
                        if y_col == "external_pa_dollars_mm"
                        else f"{v:.2f}"
                    )
                ),
            )
            if st.button("Promote to session"):
                if sel_x is None or sel_y is None:
                    st.warning("Please select both axes values before promoting.")
                    return
                promoted_theta = (
                    float(sel_y)
                    if y_col != "external_pa_dollars_mm"
                    else float(sel_y) / total_fund
                )
                promoted_selection = build_alpha_shares_payload(
                    float(sel_x), promoted_theta
                )
                if promoted_selection is None:
                    st.warning("Selection could not be normalized. Try again.")
                    return
                st.session_state["promoted_alpha_shares"] = promoted_selection
                st.session_state["scenario_grid_selection"] = promoted_selection
                bump_session_token(st.session_state, "scenario_grid_promotion_token")
                st.session_state[_GRID_PROMOTION_NOTICE_KEY] = (
                    "Selection promoted. Other pages can read "
                    "session_state['promoted_alpha_shares']."
                )
                st.rerun()

            # Offer a ready-to-run scenario YAML with promoted parameters
            promoted = st.session_state.get("promoted_alpha_shares")
            if isinstance(promoted, dict):
                try:
                    active_share_val = float(promoted["active_share"])
                    theta_val = float(promoted["theta_extpa"])
                except (TypeError, ValueError, KeyError):
                    active_share_val = None
                    theta_val = None
                if active_share_val is not None and theta_val is not None:
                    active_share_pct = (
                        active_share_val * 100.0
                        if active_share_val <= 1.0
                        else active_share_val
                    )
                    cfg_yaml = {
                        "Number of simulations": 1000,
                        "Number of months": 12,
                        "analysis_mode": "alpha_shares",
                        "theta_extpa": theta_val,
                        "active_share": active_share_pct,
                        "risk_metrics": ["Return", "Risk", "ShortfallProb"],
                    }
                    yaml_str = yaml.safe_dump(cfg_yaml, sort_keys=False)
                    st.download_button(
                        "Download Scenario YAML",
                        yaml_str,
                        file_name="scenario_alpha_shares.yml",
                        mime="application/x-yaml",
                    )

            # Optional: export heatmap to PNG if kaleido is available
            try:
                img_bytes = fig2.to_image(format="png", scale=2)
                st.download_button(
                    label="Download heatmap PNG",
                    data=img_bytes,
                    file_name="scenario_grid.png",
                    mime="image/png",
                )
            except Exception:
                pass
        except ValueError as exc:
            st.error(f"Invalid configuration: {exc}")
        except RuntimeError as exc:
            st.error(f"Simulation failed: {exc}")


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
