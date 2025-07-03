from __future__ import annotations

from pathlib import Path
from typing import Iterable
import io

import plotly.graph_objects as go
from pptx import Presentation


def save(figs: Iterable[go.Figure], path: str | Path) -> None:
    """Save figures to a PowerPoint file, one per slide."""
    pres = Presentation()
    for fig in figs:
        slide = pres.slides.add_slide(pres.slide_layouts[5])
        try:
            img_bytes = fig.to_image(format="png")
            slide.shapes.add_picture(io.BytesIO(img_bytes), 0, 0)
        except Exception:
            # Fallback: ignore export errors
            pass
    pres.save(path)
