from __future__ import annotations

from collections.abc import Iterable

import plotly.graph_objects as go


def apply(figs: Iterable[go.Figure]) -> None:
    """Apply unified hovermode to all figures."""
    for fig in figs:
        fig.update_layout(hovermode="x unified")
