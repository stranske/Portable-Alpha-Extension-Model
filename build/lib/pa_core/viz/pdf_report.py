from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable

import plotly.graph_objects as go

try:
    from PyPDF2 import PdfMerger
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dep
    PdfMerger = None


def save(figs: Iterable[go.Figure], path: str | Path) -> None:
    """Save multiple figures into a single PDF file."""
    figs = list(figs)
    path = Path(path)
    # If PdfMerger is unavailable or only a single figure provided, write a one-page PDF
    if PdfMerger is None or len(figs) <= 1:
        if figs:
            try:
                figs[0].write_image(path, format="pdf", engine="kaleido")
            except (ValueError, RuntimeError, OSError, MemoryError):
                # Fallback: write JSON representation instead of PDF bytes
                with open(path, "wb") as fh:
                    json_data = figs[0].to_json() or ""
                    fh.write(json_data.encode())
        return

    merger = PdfMerger()
    for fig in figs:
        buf = io.BytesIO()
        try:
            fig.write_image(buf, format="pdf", engine="kaleido")
            buf.seek(0)
            merger.append(buf)
        except (ValueError, RuntimeError, OSError, MemoryError):
            # fallback to JSON page
            json_data = fig.to_json() or ""
            tmp = io.BytesIO(json_data.encode())
            merger.append(tmp)
    with open(path, "wb") as fh:
        merger.write(fh)
