from __future__ import annotations

from pathlib import Path
from typing import Iterable

import plotly.graph_objects as go

from . import html_export


def save(figs: Iterable[go.Figure], prefix: str | Path) -> None:
    """Save PNG, HTML and JSON for each figure using prefix stem."""
    base = Path(prefix)
    base.parent.mkdir(parents=True, exist_ok=True)
    for i, fig in enumerate(figs, start=1):
        stem = base.with_name(f"{base.stem}_{i}")
        try:
            fig.write_image(stem.with_suffix(".png"))
        except Exception:
            pass
        html_export.save(fig, stem.with_suffix(".html"))
        with open(stem.with_suffix(".json"), "w", encoding="utf-8") as fh:
            fh.write(fig.to_json())
