"""Results page showing simulation outputs and plots."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.app import (
    _DEF_THEME,
    _DEF_XLSX,
    PLOTS,
    _get_plot_fn,
    apply_theme,
    load_data,
)
from dashboard.components.explain_results import render_explain_results_panel
from dashboard.glossary import tooltip
from pa_core.contracts import (
    SUMMARY_BREACH_PROB_COLUMN,
    SUMMARY_CVAR_COLUMN,
    SUMMARY_SHEET_NAME,
    SUMMARY_TE_COLUMN,
    SUMMARY_TRACKING_ERROR_LEGACY_COLUMN,
    manifest_path_for_output,
)


def _render_explain_results(
    *, summary: pd.DataFrame, manifest_data: dict | None, xlsx: str
) -> None:
    try:
        render_explain_results_panel(summary_df=summary, manifest=manifest_data, xlsx_path=xlsx)
    except ModuleNotFoundError:
        st.info("LLM features unavailable. Install .[llm] to enable Explain Results.")
    except Exception:
        st.info("LLM features unavailable. Install .[llm] to enable Explain Results.")


def main() -> None:
    st.title("Results")
    xlsx = st.sidebar.text_input("Results file", _DEF_XLSX)
    theme_path = st.sidebar.text_input("Theme file", _DEF_THEME)
    apply_theme(theme_path)
    if not Path(xlsx).exists():
        st.warning(f"File {xlsx} not found")
        st.stop()
    summary, paths = load_data(xlsx)

    # Load manifest if available for display and export embedding
    manifest_data = None
    manifest_path = manifest_path_for_output(xlsx)
    if manifest_path.exists():
        try:
            manifest_data = json.loads(manifest_path.read_text())
            seed = manifest_data.get("seed")
            if seed is not None:
                st.caption(f"Seed: {seed}")
        except Exception:
            manifest_data = None

    months = st.sidebar.slider("Months", 1, summary.shape[0], summary.shape[0])

    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    te_column = None
    if SUMMARY_TE_COLUMN in summary:
        te_column = SUMMARY_TE_COLUMN
    elif SUMMARY_TRACKING_ERROR_LEGACY_COLUMN in summary:
        te_column = SUMMARY_TRACKING_ERROR_LEGACY_COLUMN
    if te_column is not None:
        col1.metric(
            "Tracking Error", f"{summary[te_column].mean():.2%}", help=tooltip("monthly_TE")
        )
    if SUMMARY_CVAR_COLUMN in summary:
        col2.metric(
            "monthly_CVaR",
            f"{summary[SUMMARY_CVAR_COLUMN].mean():.2%}",
            help=tooltip("monthly_CVaR"),
        )
    if SUMMARY_BREACH_PROB_COLUMN in summary:
        col3.metric(
            "Breach Prob",
            f"{summary[SUMMARY_BREACH_PROB_COLUMN].mean():.2%}",
            help=tooltip("breach probability"),
        )

    _render_explain_results(summary=summary, manifest_data=manifest_data, xlsx=xlsx)

    if "Config" in summary.columns:
        config_options = summary["Config"].unique().tolist()
        agents = st.sidebar.multiselect("Agents", config_options, config_options)
    else:
        if summary.index.name:
            agent_options = summary.index.tolist()
        else:
            agent_options = summary.index.tolist() if len(summary.index) > 0 else ["All"]
        agents = st.sidebar.multiselect("Agents", agent_options, agent_options)
    st.sidebar.number_input("Risk-free rate", value=0.0)
    auto = st.sidebar.checkbox("Auto-refresh")
    interval = st.sidebar.number_input("Refresh every (s)", 5, 300, 60)

    tab_names = list(PLOTS) + ["Diagnostics"]
    tabs = st.tabs(tab_names)

    for name, tab in zip(tab_names[:-1], tabs[:-1]):
        fn = _get_plot_fn(PLOTS[name])
        with tab:
            if name == "Headline":
                st.plotly_chart(fn(summary), use_container_width=True)
            elif paths is not None:
                sub_paths = paths[agents].iloc[:, :months]
                st.plotly_chart(fn(sub_paths), use_container_width=True)

    with tabs[-1]:
        st.dataframe(summary)

    # Export buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        # PNG export with error handling
        try:
            png = _get_plot_fn(PLOTS["Headline"])(summary).to_image(format="png", engine="kaleido")
            st.download_button("Download PNG", png, file_name="risk_return.png", mime="image/png")
        except RuntimeError as e:
            if "kaleido" in str(e).lower() or "chrome" in str(e).lower():
                st.warning(
                    "📷 PNG export requires Kaleido or Chrome/Chromium. Install via `pip install kaleido` or `sudo apt-get install -y chromium-browser`"
                )
                st.info(
                    "💡 Tip: Use browser screenshot or install Kaleido (preferred) or Chrome for PNG exports"
                )
            else:
                st.error(f"PNG export error: {e}")

    with col2:
        # Excel download
        with open(xlsx, "rb") as fh:
            st.download_button("Download XLSX", fh, file_name=Path(xlsx).name)

    with col3:
        # Export packet button
        if st.button("📦 Export Committee Packet"):
            try:
                # Import pa_core modules here to avoid startup issues with heavy dependencies
                # (Streamlit dashboard should start quickly even if pa_core has import delays)
                sys.path.append(str(Path(__file__).parents[1]))
                from pa_core.reporting.export_packet import create_export_packet

                # Create figure
                fig = _get_plot_fn(PLOTS["Headline"])(summary)

                # Create raw returns dict for export
                raw_returns_dict = {SUMMARY_SHEET_NAME: summary}

                # Extract inputs from summary or create minimal inputs
                inputs_dict = {
                    "Data_Source": xlsx,
                    "Generated_At": str(pd.Timestamp.now()),
                    "Summary_Rows": len(summary),
                }

                # Create packet with timestamp to avoid conflicts
                base_name = f"committee_packet_{int(time.time())}"

                prev_manifest_data = None
                prev_summary_df = None
                prev_manifest_path = None
                if manifest_data:
                    prev_ref = manifest_data.get("previous_run")
                    if isinstance(prev_ref, str):
                        candidate = Path(prev_ref)
                        if candidate.exists():
                            prev_manifest_path = candidate
                if prev_manifest_path is not None:
                    try:
                        prev_manifest_data = json.loads(prev_manifest_path.read_text())
                        prev_out = (
                            prev_manifest_data.get("cli_args", {}).get("output")
                            if isinstance(prev_manifest_data, dict)
                            else None
                        )
                        if prev_out and Path(prev_out).exists():
                            prev_summary_df = pd.read_excel(prev_out, sheet_name=SUMMARY_SHEET_NAME)
                    except Exception:
                        prev_manifest_data = None
                        prev_summary_df = None

                stress_delta_df = None
                try:
                    stress_delta_df = pd.read_excel(xlsx, sheet_name="StressDelta")
                    if stress_delta_df.empty:
                        stress_delta_df = None
                except (FileNotFoundError, PermissionError, ValueError, OSError):
                    stress_delta_df = None

                with st.spinner("Creating export packet..."):
                    pptx_path, excel_path = create_export_packet(
                        figs=[fig],
                        summary_df=summary,
                        raw_returns_dict=raw_returns_dict,
                        inputs_dict=inputs_dict,
                        base_filename=base_name,
                        manifest=manifest_data,
                        prev_summary_df=prev_summary_df,
                        prev_manifest=prev_manifest_data,
                        stress_delta_df=stress_delta_df,
                    )

                st.success("✅ Export packet created!")

                # Provide download links
                with open(pptx_path, "rb") as pptx_file:
                    st.download_button(
                        "📋 Download PowerPoint",
                        pptx_file.read(),
                        file_name=Path(pptx_path).name,
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    )

                with open(excel_path, "rb") as excel_file:
                    st.download_button(
                        "📊 Download Enhanced Excel",
                        excel_file.read(),
                        file_name=Path(excel_path).name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

            except RuntimeError as e:
                if (
                    "kaleido" in str(e).lower()
                    or "chrome" in str(e).lower()
                    or "chromium" in str(e).lower()
                ):
                    st.error(
                        "📷 Export packet requires Kaleido or Chrome/Chromium for chart generation."
                    )
                    st.info(
                        "Install with: `pip install kaleido` or `sudo apt-get install chromium-browser`"
                    )
                else:
                    st.error(f"Export packet failed: {e}")
            except Exception as e:
                st.error(f"Unexpected error creating export packet: {e}")
                st.info("Please check your environment and try individual exports instead.")

    if auto:
        time.sleep(interval)
        st.experimental_rerun()


if __name__ == "__main__":
    main()
