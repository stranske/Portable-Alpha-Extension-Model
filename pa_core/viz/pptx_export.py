from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, Sequence

import plotly.graph_objects as go
from pptx import Presentation
from pptx.util import Inches


def save(
    figs: Iterable[go.Figure],
    path: str | Path,
    *,
    alt_texts: Sequence[str] | None = None,
) -> None:
    """Save figures to a PowerPoint file, one per slide.

    Parameters
    ----------
    figs:
        Iterable of Plotly figures.
    path:
        Destination ``.pptx`` file.
    alt_texts:
        Optional sequence of alt text strings for accessibility. If omitted,
        each figure's layout title is used when present.
    """
    pres = Presentation()
    alt_iter = iter(alt_texts) if alt_texts is not None else None
    for fig in figs:
        slide = pres.slides.add_slide(pres.slide_layouts[5])
        try:
            img_bytes = fig.to_image(format="png", engine="kaleido")
            pic = slide.shapes.add_picture(io.BytesIO(img_bytes), Inches(0), Inches(0))
            alt = next(alt_iter, None) if alt_iter else None
            if not alt:
                alt = str(fig.layout.title.text) if fig.layout.title.text else ""
            if alt:
                el = pic._element.xpath("./p:nvPicPr/p:cNvPr")[0]
                el.set("descr", alt)
        except Exception:
            # Fallback: ignore export errors
            pass
    pres.save(str(path))
