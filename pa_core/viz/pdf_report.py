from __future__ import annotations

from pathlib import Path
from typing import Iterable
import io

import plotly.graph_objects as go

try:
    from PyPDF2 import PdfMerger  # type: ignore
except Exception:  # pragma: no cover - optional dep
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
                    fh.write(figs[0].to_json().encode())
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
            tmp = io.BytesIO(fig.to_json().encode())
            merger.append(tmp)
    with open(path, "wb") as fh:
        merger.write(fh)
