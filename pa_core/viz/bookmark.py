from __future__ import annotations

import json

import plotly.graph_objects as go


def save(fig: go.Figure) -> str:
    """Return JSON bookmark for ``fig``."""
    return str(fig.to_json())


def load(blob: str) -> go.Figure:
    """Recreate figure from bookmark JSON."""
    data = json.loads(blob)
    return go.Figure(data=data.get("data"), layout=data.get("layout"))
