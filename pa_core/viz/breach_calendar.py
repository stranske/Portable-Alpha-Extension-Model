from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme
from .utils import safe_to_numpy


def make(df_summary: pd.DataFrame) -> go.Figure:
    """Return heatmap of threshold breaches by month."""
    df = df_summary.copy()
    month_series = df.get("Month")
    if isinstance(month_series, pd.Series):
        months = month_series.tolist()
    else:
        months = df.index.tolist()
    te_cap = theme.THRESHOLDS.get("te_cap", 0.03)
    short_cap = theme.THRESHOLDS.get("shortfall_amber", theme.LOW_BUFFER_THRESHOLD)
    te_vals_raw = (
        df["TrackingErr"] if "TrackingErr" in df else pd.Series(0.0, index=df.index)
    )
    short_vals_raw = (
        df["ShortfallProb"] if "ShortfallProb" in df else pd.Series(theme.DEFAULT_SHORTFALL_PROB, index=df.index)
    )
    
    # Ensure values are numeric, converting non-numeric to NaN
    te_vals = pd.Series(pd.to_numeric(te_vals_raw, errors='coerce'), index=df.index).fillna(0.0)
    short_vals = pd.Series(pd.to_numeric(short_vals_raw, errors='coerce'), index=df.index).fillna(float(theme.DEFAULT_SHORTFALL_PROB))
    
    te_breach = (te_vals > te_cap).astype(float)
    short_breach = (short_vals > short_cap).astype(float)
    z = np.vstack([safe_to_numpy(te_breach), safe_to_numpy(short_breach)])
    fig = go.Figure(
        data=go.Heatmap(z=z, x=months, y=["TE", "Shortfall"]),
        layout_template=theme.TEMPLATE,
    )
    fig.update_layout(xaxis_title="Month", yaxis_title="Metric")
    return fig
