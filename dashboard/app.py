from __future__ import annotations

import pandas as pd
import streamlit as st
from pathlib import Path

from pa_core.viz import risk_return, fan, path_dist


_DEF_XLSX = "Outputs.xlsx"


def load_data(xlsx: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_excel(xlsx, sheet_name="Summary")
    paths = pd.read_parquet(Path(xlsx).with_suffix(".parquet"))
    return summary, paths


def main() -> None:
    st.title("Portable Alpha Dashboard")
    xlsx = st.sidebar.text_input("Results file", _DEF_XLSX)
    if not Path(xlsx).exists():
        st.warning(f"File {xlsx} not found")
        st.stop()
    summary, paths = load_data(xlsx)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Headline",
        "Funding fan",
        "Path dist",
        "Diagnostics",
    ])
    with tab1:
        st.plotly_chart(risk_return.make(summary), use_container_width=True)
    with tab2:
        st.plotly_chart(fan.make(paths), use_container_width=True)
    with tab3:
        st.plotly_chart(path_dist.make(paths), use_container_width=True)
    with tab4:
        st.dataframe(summary)

    png = risk_return.make(summary).to_image(format="png")
    st.download_button(
        label="Download PNG",
        data=png,
        file_name="risk_return.png",
        mime="image/png",
    )


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
