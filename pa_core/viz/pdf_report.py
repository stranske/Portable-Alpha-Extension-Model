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
    if PdfMerger is None or not figs:
        if figs:
            try:
                figs[0].write_image(path, format="pdf")
            except Exception:
                with open(path, "wb") as fh:
                    json_data = figs[0].to_json() or ""
                    fh.write(json_data.encode())
        return

    merger = PdfMerger()
    for fig in figs:
        buf = io.BytesIO()
        try:
            fig.write_image(buf, format="pdf")
            buf.seek(0)
            merger.append(buf)
        except Exception:
            # fallback to JSON page
            json_data = fig.to_json() or ""
            tmp = io.BytesIO(json_data.encode())
            merger.append(tmp)
    with open(path, "wb") as fh:
        merger.write(fh)
