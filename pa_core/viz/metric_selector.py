from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import streamlit as st

from . import risk_return


def make(df_summary: pd.DataFrame, metrics: Sequence[str]) -> None:
    """Display metric selector widget and plot."""
    metric = st.selectbox("Metric", metrics)
    df = df_summary.rename(columns={metric: "Selected"})
    fig = risk_return.make(df)
    st.plotly_chart(fig, use_container_width=True)
