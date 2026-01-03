from __future__ import annotations

from typing import Iterable

import pandas as pd
import plotly.graph_objects as go

from . import theme


def _format_constraint_lines(parts: Iterable[str]) -> str:
    lines = [p for p in parts if p]
    if not lines:
        return ""
    return "<br>".join(lines)


def make(
    df: pd.DataFrame,
    *,
    max_te: float | None = None,
    max_cvar: float | None = None,
    max_breach: float | None = None,
    max_shortfall: float | None = None,
) -> go.Figure:
    """Return a return-vs-TE frontier scatter with CVaR coloring."""
    fig = go.Figure(layout_template=theme.TEMPLATE)
    if df.empty or "frontier_risk" not in df or "frontier_return" not in df:
        fig.update_layout(
            xaxis_title="Tracking Error",
            yaxis_title="Annualized Return",
        )
        return fig

    work = df.copy()
    if "constraints_satisfied" in work:
        satisfied = work["constraints_satisfied"].fillna(False).astype(bool)
    else:
        satisfied = pd.Series(True, index=work.index)
    if "is_frontier" in work:
        frontier = work["is_frontier"].fillna(False).astype(bool)
    else:
        frontier = pd.Series(False, index=work.index)

    cvar = (
        work["frontier_cvar"].fillna(0.0).astype(float)
        if "frontier_cvar" in work
        else pd.Series(0.0, index=work.index, dtype=float)
    )
    feasible = work[satisfied]
    infeasible = work[~satisfied]

    if not infeasible.empty:
        fig.add_trace(
            go.Scatter(
                x=infeasible["frontier_risk"],
                y=infeasible["frontier_return"],
                mode="markers",
                name="Infeasible",
                marker=dict(
                    color="rgba(160, 160, 160, 0.55)",
                    symbol="x",
                    size=7,
                ),
                hovertemplate="TE=%{x:.2%}<br>Return=%{y:.2%}<extra>Infeasible</extra>",
            )
        )

    if not feasible.empty:
        cvar_feasible = cvar.loc[feasible.index]
        colorbar = dict(title="Monthly CVaR")
        if max_cvar is not None:
            colorbar = dict(
                title="Monthly CVaR",
                tickmode="array",
                tickvals=[float(cvar_feasible.min()), float(max_cvar)],
                ticktext=["Min", "Max CVaR"],
            )
        fig.add_trace(
            go.Scatter(
                x=feasible["frontier_risk"],
                y=feasible["frontier_return"],
                mode="markers",
                name="Feasible",
                marker=dict(
                    color=cvar_feasible,
                    colorscale="Viridis",
                    showscale=True,
                    size=8,
                    colorbar=colorbar,
                    line=dict(width=0.4, color="white"),
                ),
                hovertemplate=(
                    "TE=%{x:.2%}<br>Return=%{y:.2%}<br>CVaR=%{marker.color:.2%}"
                    "<extra>Feasible</extra>"
                ),
            )
        )

    frontier_pts = work[frontier & satisfied].sort_values("frontier_risk")
    if not frontier_pts.empty:
        fig.add_trace(
            go.Scatter(
                x=frontier_pts["frontier_risk"],
                y=frontier_pts["frontier_return"],
                mode="lines+markers",
                name="Frontier",
                line=dict(color="#ffffff", width=2),
                marker=dict(color="#ffffff", size=6),
                hovertemplate="TE=%{x:.2%}<br>Return=%{y:.2%}<extra>Frontier</extra>",
            )
        )

    risk_max = float(work["frontier_risk"].max())
    if max_te is not None:
        fig.add_vline(
            x=float(max_te),
            line_dash="dash",
            line_color="red",
            annotation_text="Max TE",
            annotation_position="top",
        )
        if risk_max > max_te:
            fig.add_vrect(
                x0=float(max_te),
                x1=risk_max * 1.02,
                fillcolor="rgba(255, 0, 0, 0.08)",
                line_width=0,
                layer="below",
            )

    constraint_text = _format_constraint_lines(
        [
            f"Max TE: {max_te:.2%}" if max_te is not None else "",
            f"Max CVaR: {max_cvar:.2%}" if max_cvar is not None else "",
            f"Max Breach: {max_breach:.2%}" if max_breach is not None else "",
            f"Max Shortfall: {max_shortfall:.2%}" if max_shortfall is not None else "",
        ]
    )
    if constraint_text:
        fig.add_annotation(
            text=constraint_text,
            xref="paper",
            yref="paper",
            x=0.01,
            y=0.01,
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
            align="left",
            font=dict(size=11),
            bgcolor="rgba(0, 0, 0, 0.35)",
            bordercolor="rgba(255, 255, 255, 0.4)",
        )

    fig.update_layout(
        xaxis_title="Tracking Error (annualized)",
        yaxis_title="Annualized Return",
        legend_title_text="",
    )
    return fig
