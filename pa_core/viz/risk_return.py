from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import theme

_DEFAULT_COLORWAY = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
]
_AGENT_SYMBOLS = [
    "circle",
    "square",
    "triangle-up",
    "cross",
    "x",
    "triangle-down",
]
_FIXED_AGENT_STYLES = {
    "Total": {"color": "#111827", "symbol": "star", "size": 18},
    "Base": {"color": "#6b7280", "symbol": "diamond", "size": 16},
}
_SHORTFALL_COLORS = {
    "green": "#2ca02c",
    "orange": "#ff7f0e",
    "red": "#d62728",
}


def make(df_summary: pd.DataFrame) -> go.Figure:
    """Return tracking-error or volatility vs return scatter plot.

    Parameters
    ----------
    df_summary : pandas.DataFrame
        Must contain Agent and either monthly_TE (tracking error) or
        monthly_AnnVol. Prefer monthly_TE when it has data; fall back to
        TrackingErr (legacy) or monthly_AnnVol. For the y-axis, prefer
        terminal_ExcessReturn and fall back to terminal_AnnReturn when it has
        data. terminal_ShortfallProb is optional.
    """
    df = df_summary.copy()
    df["terminal_ShortfallProb"] = df.get("terminal_ShortfallProb", theme.DEFAULT_SHORTFALL_PROB)
    thr = theme.THRESHOLDS

    def has_data(col: str) -> bool:
        return col in df and df[col].notna().any()

    if has_data("monthly_TE"):
        x_col = "monthly_TE"
        x_label = "Tracking Error"
        x_hover = "Tracking Error"
        cap = thr.get("te_cap", 0.03)
        _fill_benchmark_tracking_error(df, x_col)
    elif has_data("TrackingErr"):
        df["monthly_TE"] = df["TrackingErr"]
        x_col = "monthly_TE"
        x_label = "Tracking Error"
        x_hover = "Tracking Error"
        cap = thr.get("te_cap", 0.03)
        _fill_benchmark_tracking_error(df, x_col)
    else:
        x_col = "monthly_AnnVol"
        x_label = "Annualized Volatility"
        x_hover = "Annualized Volatility"
        cap = thr.get("vol_cap", 0.03)

    if has_data("terminal_ExcessReturn"):
        y_col = "terminal_ExcessReturn"
        y_label = "Annualized Excess Return"
        y_hover = "Annualized Excess Return"
    else:
        y_col = "terminal_AnnReturn"
        y_label = "Annualized Return"
        y_hover = "Annualized Return"

    fig = go.Figure(layout_template=theme.TEMPLATE)
    agents = _ordered_agents(df["Agent"])
    color_map = _agent_colors(agents)
    symbol_map = _agent_symbols(agents)
    for agent in agents:
        agent_rows = df[df["Agent"].astype(str) == agent]
        probs = pd.to_numeric(
            agent_rows["terminal_ShortfallProb"],
            errors="coerce",
        ).fillna(theme.DEFAULT_SHORTFALL_PROB)
        marker_line_colors = [_shortfall_color(prob, thr) for prob in probs]
        fixed_style = _FIXED_AGENT_STYLES.get(agent, {})
        marker_size = fixed_style.get("size", 12)
        text = agent_rows["Agent"] if agent in _FIXED_AGENT_STYLES else None
        mode = "markers+text" if text is not None else "markers"
        fig.add_trace(
            go.Scatter(
                x=agent_rows[x_col],
                y=agent_rows[y_col],
                mode=mode,
                name=agent,
                legendgroup=agent,
                marker=dict(
                    size=marker_size,
                    color=color_map[agent],
                    symbol=symbol_map[agent],
                    line=dict(
                        color=marker_line_colors,
                        width=3 if agent in _FIXED_AGENT_STYLES else 1.5,
                    ),
                ),
                text=text if text is not None else agent_rows["Agent"],
                textposition="top center",
                customdata=[[float(prob)] for prob in probs],
                hovertemplate=(
                    f"%{{text}}<br>{x_hover}=%{{x:.2%}}<br>{y_hover}=%{{y:.2%}}"
                    "<br>Shortfall=%{customdata[0]:.2%}<extra></extra>"
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
        legend_title_text="Sleeve / Scenario",
        template=theme.TEMPLATE,
    )
    return fig


def _ordered_agents(agent_series: pd.Series) -> list[str]:
    agents = list(dict.fromkeys(agent_series.astype(str)))
    highlighted = [agent for agent in _fixed_agent_order() if agent in agents]
    return [agent for agent in agents if agent not in _FIXED_AGENT_STYLES] + highlighted


def _fixed_agent_order() -> list[str]:
    preferred_order = ["Base", "Total"]
    preferred = [agent for agent in preferred_order if agent in _FIXED_AGENT_STYLES]
    extras = [agent for agent in _FIXED_AGENT_STYLES if agent not in preferred]
    return preferred + extras


def _fill_benchmark_tracking_error(df: pd.DataFrame, x_col: str) -> None:
    benchmark_mask = df["Agent"].astype(str) == "Base"
    df.loc[benchmark_mask, x_col] = df.loc[benchmark_mask, x_col].fillna(0.0)


def _agent_colors(agents: list[str]) -> dict[str, str]:
    colorway = list(theme.TEMPLATE.layout.colorway or _DEFAULT_COLORWAY)
    dynamic_agents = [agent for agent in agents if agent not in _FIXED_AGENT_STYLES]
    colors = {agent: colorway[idx % len(colorway)] for idx, agent in enumerate(dynamic_agents)}
    for agent, style in _FIXED_AGENT_STYLES.items():
        if agent in agents:
            colors[agent] = str(style["color"])
    return colors


def _agent_symbols(agents: list[str]) -> dict[str, str]:
    dynamic_agents = [agent for agent in agents if agent not in _FIXED_AGENT_STYLES]
    symbols = {
        agent: _AGENT_SYMBOLS[idx % len(_AGENT_SYMBOLS)] for idx, agent in enumerate(dynamic_agents)
    }
    for agent, style in _FIXED_AGENT_STYLES.items():
        if agent in agents:
            symbols[agent] = str(style["symbol"])
    return symbols


def _shortfall_color(prob: float, thresholds: dict[str, float]) -> str:
    if prob <= thresholds.get("shortfall_green", 0.05):
        return _SHORTFALL_COLORS["green"]
    if prob <= thresholds.get("shortfall_amber", theme.LOW_BUFFER_THRESHOLD):
        return _SHORTFALL_COLORS["orange"]
    return _SHORTFALL_COLORS["red"]
