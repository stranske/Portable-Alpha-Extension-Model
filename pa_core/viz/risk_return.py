from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(df_summary: pd.DataFrame) -> go.Figure:
    """Return tracking-error or volatility vs return scatter plot.

    Parameters
    ----------
    df_summary : pandas.DataFrame
        Must contain Agent and either TE (tracking error) or AnnVol. For the
        y-axis, prefer ExcessReturn and fall back to AnnReturn. TrackingErr is
        accepted as a legacy alias for TE. ShortfallProb is optional.
    """
    df = df_summary.copy()
    df["ShortfallProb"] = df.get("ShortfallProb", theme.DEFAULT_SHORTFALL_PROB)
    color = []
    thr = theme.THRESHOLDS

    if "TE" in df:
        x_col = "TE"
        x_label = "Tracking Error"
        x_hover = "Tracking Error"
        cap = thr.get("te_cap", 0.03)
    elif "TrackingErr" in df:
        df["TE"] = df["TrackingErr"]
        x_col = "TE"
        x_label = "Tracking Error"
        x_hover = "Tracking Error"
        cap = thr.get("te_cap", 0.03)
    else:
        x_col = "AnnVol"
        x_label = "Annualized Volatility"
        x_hover = "Annualized Volatility"
        cap = thr.get("vol_cap", 0.03)

    if "ExcessReturn" in df:
        y_col = "ExcessReturn"
        y_label = "Annualized Excess Return"
        y_hover = "Annualized Excess Return"
    else:
        y_col = "AnnReturn"
        y_label = "Annualized Return"
        y_hover = "Annualized Return"

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
            x=df[x_col],
            y=df[y_col],
            mode="markers",
            marker=dict(size=12, color=color),
            text=df["Agent"],
            hovertemplate=(
                f"%{{text}}<br>{x_hover}=%{{x:.2%}}<br>{y_hover}=%{{y:.2%}}" "<extra></extra>"
            ),
        )
    )

    # sweet-spot rectangle
    rect = dict(
        type="rect",
        xref="x",
        yref="y",
        x0=0,
        x1=cap,
        y0=thr.get("excess_return_floor", 0.03),
        y1=thr.get("excess_return_target", 0.05),
        fillcolor="lightgrey",
        opacity=0.3,
        line_width=0,
    )
    fig.add_vline(x=cap, line_dash="dash")
    fig.add_hline(y=thr.get("excess_return_target", 0.05), line_dash="dash")
    fig.update_layout(
        shapes=[rect],
        xaxis_title=x_label,
        yaxis_title=y_label,
        template=theme.TEMPLATE,
    )
    return fig
