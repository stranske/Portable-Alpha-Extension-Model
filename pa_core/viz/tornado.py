from __future__ import annotations

from typing import Mapping

import pandas as pd
import plotly.graph_objects as go

try:  # theme is optional when loading module standalone
    from . import theme

    TEMPLATE = theme.TEMPLATE
except (ImportError, ModuleNotFoundError):  # pragma: no cover - fallback when package not fully available
    TEMPLATE = None


def make(contrib: Mapping[str, float] | pd.Series) -> go.Figure:
    """Return tornado chart sorted by absolute contribution.

    Parameters
    ----------
    contrib: Mapping[str, float] or Series
        Mapping of parameter name to delta contribution.
    """
    if isinstance(contrib, pd.Series):
        series = contrib.astype(float)
    else:
        series = pd.Series({k: float(v) for k, v in contrib.items()})

    series = series.reindex(series.abs().sort_values(ascending=False).index)
    layout = dict(template=TEMPLATE) if TEMPLATE else {}
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Bar(x=series.values, y=series.index, orientation="h"))
    fig.update_layout(xaxis_title="Delta", yaxis_title="Parameter")
    return fig
