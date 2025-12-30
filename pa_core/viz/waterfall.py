from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(contrib: Mapping[str, float] | pd.Series) -> go.Figure:
    """Return waterfall chart of risk or return contributions."""
    if isinstance(contrib, pd.Series):
        contrib_s: pd.Series = contrib
    else:
        contrib_s = pd.Series({str(k): float(v) for k, v in contrib.items()})
    labels = contrib_s.index.tolist()
    values = contrib_s.astype(float).tolist()
    fig = go.Figure(layout_template=theme.TEMPLATE)
    fig.add_trace(go.Waterfall(x=labels, y=values, connector=dict(line=dict(color="grey"))))
    fig.update_layout(xaxis_title="Agent", yaxis_title="Contribution")
    return fig
