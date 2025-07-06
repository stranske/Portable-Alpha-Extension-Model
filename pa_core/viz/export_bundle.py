from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import plotly.graph_objects as go

from . import html_export


def save(
    figs: Iterable[go.Figure],
    prefix: str | Path,
    *,
    alt_texts: Sequence[str] | None = None,
) -> None:
    """Save PNG, HTML and JSON for each figure using ``prefix`` stem."""
    base = Path(prefix)
    base.parent.mkdir(parents=True, exist_ok=True)
    alt_iter = iter(alt_texts) if alt_texts is not None else None
    for i, fig in enumerate(figs, start=1):
        stem = base.with_name(f"{base.stem}_{i}")
        try:
            fig.write_image(stem.with_suffix(".png"))
        except Exception:
            pass
        alt = next(alt_iter, None) if alt_iter else None
        html_export.save(fig, stem.with_suffix(".html"), alt_text=alt)
        with open(stem.with_suffix(".json"), "w", encoding="utf-8") as fh:
            json_data = fig.to_json()
            fh.write(json_data if isinstance(json_data, str) else "")
