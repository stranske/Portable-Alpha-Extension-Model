from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(df_summary: pd.DataFrame) -> go.Figure:
    """Return heatmap of threshold breaches by month."""
    df = df_summary.copy()
    month_series = df.get("Month")
    months = month_series if month_series is not None else df.index
    te_cap = theme.THRESHOLDS.get("te_cap", 0.03)
    short_cap = theme.THRESHOLDS.get("shortfall_amber", 0.1)
    te_vals = (
        df["TrackingErr"] if "TrackingErr" in df else pd.Series(0.0, index=df.index)
    )
    short_vals = (
        df["ShortfallProb"] if "ShortfallProb" in df else pd.Series(0.0, index=df.index)
    )
    te_breach = (te_vals > te_cap).astype(float)
    short_breach = (short_vals > short_cap).astype(float)
    z = np.vstack([te_breach.to_numpy(), short_breach.to_numpy()])
    fig = go.Figure(
        data=go.Heatmap(z=z, x=list(months), y=["TE", "Shortfall"]),
        layout_template=theme.TEMPLATE,
    )
    fig.update_layout(xaxis_title="Month", yaxis_title="Metric")
    return fig
