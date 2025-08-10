"""Results page showing simulation outputs and plots."""

from __future__ import annotations

import time
from pathlib import Path

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

    png = _get_plot_fn(PLOTS["Headline"])(summary).to_image(format="png")
    st.download_button(
        "Download PNG", png, file_name="risk_return.png", mime="image/png"
    )
    with open(xlsx, "rb") as fh:
        st.download_button("Download XLSX", fh, file_name=Path(xlsx).name)

    if auto:
        time.sleep(interval)
        st.experimental_rerun()


if __name__ == "__main__":
    main()
