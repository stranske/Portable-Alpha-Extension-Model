from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any, Iterable, Sequence, cast

import plotly.graph_objects as go
from pptx import Presentation
from pptx.util import Inches

__all__ = ["render_chart_png", "add_chart_slide", "save"]

# Tiny 1x1 PNG used as a placeholder in CI/pytest so chart slides never spawn a
# (potentially hanging) kaleido/Chromium subprocess during automated runs.
_ONE_PX_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMA"
    "ASsJTYQAAAAASUVORK5CYII="
)


def render_chart_png(fig: Any) -> bytes:
    """Render a Plotly figure to PNG bytes for embedding in a PPTX slide.

    In CI/pytest a tiny placeholder image is returned to avoid hanging kaleido
    subprocesses. Outside CI a real static render is attempted via Kaleido; if
    that renderer is unavailable the failure is surfaced as an actionable
    ``RuntimeError`` rather than being silently swallowed (a silently-skipped
    chart produces a divergent, incomplete board pack).
    """
    if os.environ.get("CI") or os.environ.get("PYTEST_CURRENT_TEST"):
        return _ONE_PX_PNG
    try:
        return cast(bytes, fig.to_image(format="png", engine="kaleido"))
    except Exception as e:  # pragma: no cover - exercised only without kaleido
        raise RuntimeError(
            "PPTX export requires a static image renderer (Kaleido/Chromium). "
            "Install Plotly Kaleido or Chrome/Chromium. For Debian/Ubuntu: "
            "pip install 'plotly[kaleido]' or sudo apt-get install -y chromium-browser. "
            f"Original error: {e}"
        ) from e


def add_chart_slide(
    prs: Any,
    fig: Any,
    *,
    alt: str | None = None,
    layout_index: int = 5,
) -> None:
    """Add a single full-bleed chart slide to ``prs`` for ``fig``.

    The image is rendered via :func:`render_chart_png` (shared error handling)
    and accessibility alt text is set from ``alt`` or the figure's layout title.
    """
    slide = prs.slides.add_slide(prs.slide_layouts[layout_index])
    img_bytes = render_chart_png(fig)
    pic = slide.shapes.add_picture(io.BytesIO(img_bytes), Inches(0), Inches(0))
    if not alt:
        layout = getattr(fig, "layout", None)
        title = getattr(layout, "title", None) if layout is not None else None
        text = getattr(title, "text", None) if title is not None else None
        alt = str(text) if text else ""
    if alt:
        el = pic._element.xpath("./p:nvPicPr/p:cNvPr")[0]
        el.set("descr", alt)


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

    Raises
    ------
    RuntimeError
        If a static image renderer (Kaleido/Chromium) is required but
        unavailable. This matches the committee export-packet behaviour so a
        missing renderer can never silently drop chart slides.
    """
    pres = Presentation()
    alt_iter = iter(alt_texts) if alt_texts is not None else None
    for fig in figs:
        add_chart_slide(pres, fig, alt=next(alt_iter, None) if alt_iter else None)
    pres.save(str(path))
