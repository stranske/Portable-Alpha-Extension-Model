from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(df_summary: pd.DataFrame) -> go.Figure:
    """Return risk-return scatter plot.

    Parameters
    ----------
    df_summary : pandas.DataFrame
        Must contain AnnReturn, AnnVol, Agent. ShortfallProb is optional.
    """
    df = df_summary.copy()
    df["ShortfallProb"] = df.get("ShortfallProb", theme.DEFAULT_SHORTFALL_PROB)
    color = []
    thr = theme.THRESHOLDS
    vol_cap = thr.get("vol_cap", 0.03)
    for prob in df["ShortfallProb"].fillna(theme.DEFAULT_SHORTFALL_PROB):
        if prob <= thr.get("shortfall_green", 0.05):
            color.append("green")
        elif prob <= thr.get("shortfall_amber", theme.LOW_BUFFER_THRESHOLD):
            color.append("orange")
        else:
            color.append("red")

    fig = go.Figure(layout_template=theme.TEMPLATE)
    fig.add_trace(
        go.Scatter(
            x=df["AnnVol"],
            y=df["AnnReturn"],
            mode="markers",
            marker=dict(size=12, color=color),
            text=df["Agent"],
            hovertemplate="%{text}<br>Vol=%{x:.2%}<br>Return=%{y:.2%}<extra></extra>",
        )
    )

    # sweet-spot rectangle
    rect = dict(
        type="rect",
        xref="x",
        yref="y",
        x0=0,
        x1=vol_cap,
        y0=thr.get("excess_return_floor", 0.03),
        y1=thr.get("excess_return_target", 0.05),
        fillcolor="lightgrey",
        opacity=0.3,
        line_width=0,
    )
    fig.add_vline(x=vol_cap, line_dash="dash")
    fig.add_hline(y=thr.get("excess_return_target", 0.05), line_dash="dash")
    fig.update_layout(
        shapes=[rect],
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Return",
        template=theme.TEMPLATE,
    )
    return fig
