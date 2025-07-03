from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go


def save(fig: go.Figure, path: str | Path) -> None:
    """Save a figure to an interactive HTML file."""
    fig.write_html(str(path), include_plotlyjs="cdn")

