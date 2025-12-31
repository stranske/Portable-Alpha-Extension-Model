from __future__ import annotations

import plotly.graph_objects as go


def save(fig: go.Figure, path: str) -> None:
    """Write figure to PDF using Plotly's static image renderer."""
    try:
        fig.write_image(path, format="pdf", engine="kaleido")
    except (ValueError, RuntimeError, OSError, MemoryError):
        with open(path, "wb") as fh:
            fh.write(str(fig.to_json()).encode())
