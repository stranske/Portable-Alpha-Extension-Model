from __future__ import annotations

import pandas as pd

try:  # pragma: no cover - optional dependency
    import ipywidgets as widgets  # type: ignore[import-not-found]
    from IPython.display import display  # type: ignore[import-not-found]
except (
    ImportError,
    ModuleNotFoundError,
):  # pragma: no cover - ipywidgets not installed
    widgets = None  # type: ignore[assignment]
    display = None  # type: ignore[assignment]

from . import risk_return


def explore(df_summary: pd.DataFrame) -> None:
    """Display interactive risk-return explorer."""

    if widgets is None or display is None:  # pragma: no cover - import check
        msg = "ipywidgets is required for pa_core.viz.widgets"
        raise ImportError(msg)

    def _update(scale: float) -> None:
        scaled = df_summary.copy()
        scaled["AnnReturn"] *= scale
        fig = risk_return.make(scaled)
        display(fig)  # type: ignore[no-untyped-call]

    slider = widgets.FloatSlider(value=1.0, min=0.5, max=1.5, step=0.1)
    widgets.interact(_update, scale=slider)
