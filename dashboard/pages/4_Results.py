"""Results page showing simulation outputs and plots."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.app import (
    PLOTS,
    _DEF_THEME,
    _DEF_XLSX,
    _get_plot_fn,
    apply_theme,
    load_data,
)


def main() -> None:
    st.title("Results")
    xlsx = st.sidebar.text_input("Results file", _DEF_XLSX)
    theme_path = st.sidebar.text_input("Theme file", _DEF_THEME)
    apply_theme(theme_path)
    if not Path(xlsx).exists():
        st.warning(f"File {xlsx} not found")
        st.stop()
    summary, paths = load_data(xlsx)

    months = st.sidebar.slider("Months", 1, summary.shape[0], summary.shape[0])

    if "Config" in summary.columns:
        config_options = summary["Config"].unique().tolist()
        agents = st.sidebar.multiselect("Agents", config_options, config_options)
    else:
        if summary.index.name:
            agent_options = summary.index.tolist()
        else:
            agent_options = (
                summary.index.tolist() if len(summary.index) > 0 else ["All"]
            )
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
            png = _get_plot_fn(PLOTS["Headline"])(summary).to_image(format="png")
            st.download_button(
                "Download PNG", png, file_name="risk_return.png", mime="image/png"
            )
        except RuntimeError as e:
            if "Chrome" in str(e) or "Kaleido" in str(e):
                st.warning("📷 PNG export requires Chrome installation. Run: `sudo apt-get install -y chromium-browser`")
                st.info("💡 Tip: Use browser screenshot or install Chrome for PNG exports")
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
                # Import here to avoid startup issues
                from pathlib import Path
                import sys
                sys.path.append(str(Path(__file__).parents[1]))
                from pa_core.reporting.export_packet import create_export_packet
                from pa_core import viz
                
                # Create figure
                fig = _get_plot_fn(PLOTS["Headline"])(summary)
                
                # Create raw returns dict for export
                raw_returns_dict = {"Summary": summary}
                
                # Extract inputs from summary or create minimal inputs
                inputs_dict = {
                    "Data_Source": xlsx,
                    "Generated_At": str(pd.Timestamp.now()),
                    "Summary_Rows": len(summary),
                }
                
                # Create packet with timestamp to avoid conflicts
                base_name = f"committee_packet_{int(time.time())}"
                
                with st.spinner("Creating export packet..."):
                    pptx_path, excel_path = create_export_packet(
                        figs=[fig],
                        summary_df=summary,
                        raw_returns_dict=raw_returns_dict,
                        inputs_dict=inputs_dict,
                        base_filename=base_name,
                    )
                
                st.success("✅ Export packet created!")
                
                # Provide download links
                with open(pptx_path, "rb") as pptx_file:
                    st.download_button(
                        "📋 Download PowerPoint", 
                        pptx_file.read(), 
                        file_name=Path(pptx_path).name,
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    )
                
                with open(excel_path, "rb") as excel_file:
                    st.download_button(
                        "📊 Download Enhanced Excel", 
                        excel_file.read(), 
                        file_name=Path(excel_path).name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
            except RuntimeError as e:
                if "Chrome" in str(e) or "Chromium" in str(e) or "Kaleido" in str(e):
                    st.error("📷 Export packet requires Chrome/Chromium for chart generation.")
                    st.info("Install with: `sudo apt-get install chromium-browser`")
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
