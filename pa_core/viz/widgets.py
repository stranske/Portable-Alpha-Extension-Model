from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from . import risk_return

widgets: Any | None
display: Callable[..., Any] | None

try:  # pragma: no cover - optional dependency
    import ipywidgets as widgets
    from IPython.display import display as display_fn

    display: Callable[..., Any] | None = display_fn
except (
    ImportError,
    ModuleNotFoundError,
):  # pragma: no cover - ipywidgets not installed
    widgets = None
    display = None


def explore(df_summary: pd.DataFrame) -> None:
    """Display interactive risk-return explorer."""

    if widgets is None or display is None:  # pragma: no cover - import check
        msg = "ipywidgets is required for pa_core.viz.widgets"
        raise ImportError(msg)

    def _update(scale: float) -> None:
        scaled = df_summary.copy()
        scaled["AnnReturn"] *= scale
        fig = risk_return.make(scaled)
        display(fig)

    slider = widgets.FloatSlider(value=1.0, min=0.5, max=1.5, step=0.1)
    widgets.interact(_update, scale=slider)
