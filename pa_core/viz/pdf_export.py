from __future__ import annotations

import os

import plotly.graph_objects as go

from .export_backend import write_figure_image


def save(fig: go.Figure, path: str) -> None:
    """Write figure to PDF using Plotly's static image renderer."""
    if os.environ.get("PAEM_SKIP_PDF_EXPORT") or os.environ.get("PYTEST_CURRENT_TEST"):
        with open(path, "wb") as fh:
            fh.write(str(fig.to_json()).encode())
        return
    try:
        write_figure_image(fig, path, format="pdf")
    except (ValueError, RuntimeError, OSError, MemoryError):
        with open(path, "wb") as fh:
            fh.write(str(fig.to_json()).encode())
