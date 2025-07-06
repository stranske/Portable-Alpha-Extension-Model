from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go


def save(fig: go.Figure, path: str | Path, *, alt_text: str | None = None) -> None:
    """Save a figure to an interactive HTML file.

    If ``alt_text`` is provided, it is inserted as an ``aria-label`` so
    screen readers can describe the chart.
    """
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    if alt_text:
        html = html.replace(
            "<div>", f'<div role="img" aria-label="{alt_text}">', 1
        )
    Path(path).write_text(html, encoding="utf-8")

