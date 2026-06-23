from __future__ import annotations

import io
import os
import warnings
from pathlib import Path
from typing import Any, Iterable

import plotly.graph_objects as go

from .export_backend import figure_to_pdf_bytes, run_with_browser_png_cache, write_figure_image

PdfWriterType: Any
try:
    from pypdf import PdfWriter as PdfWriterType
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dep
    PdfWriterType = None


def _write_json_fixture(figs: list[go.Figure], path: Path) -> None:
    payload = "[" + ",".join(fig.to_json() or "{}" for fig in figs) + "]"
    with open(path, "wb") as fh:
        fh.write(payload.encode())


def save(figs: Iterable[go.Figure], path: str | Path) -> None:
    """Save multiple figures into a single PDF file."""
    figs = list(figs)
    path = Path(path)
    if os.environ.get("PAEM_SKIP_PDF_EXPORT") or os.environ.get("PYTEST_CURRENT_TEST"):
        _write_json_fixture(figs, path)
        return
    if not figs:
        _write_json_fixture(figs, path)
        return

    if len(figs) == 1:
        try:
            write_figure_image(figs[0], path, format="pdf")
        except (ValueError, RuntimeError, OSError, MemoryError) as exc:
            warnings.warn(
                f"PDF export failed; writing figure JSON fallback instead: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            _write_json_fixture(figs, path)
        return

    if PdfWriterType is None:
        warnings.warn(
            "pypdf is required to merge multiple PDF figures; writing figure JSON fallback instead.",
            RuntimeWarning,
            stacklevel=2,
        )
        _write_json_fixture(figs, path)
        return

    writer = PdfWriterType()
    buffers: list[io.BytesIO] = []
    for fig in figs:
        buf = io.BytesIO()
        try:
            buf.write(figure_to_pdf_bytes(fig))
        except (ValueError, RuntimeError, OSError, MemoryError) as exc:
            warnings.warn(
                f"PDF export failed; writing all figure JSON fallbacks instead: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            _write_json_fixture(figs, path)
            return
        buf.seek(0)
        buffers.append(buf)

    for buf in buffers:
        writer.append(buf)
    with open(path, "wb") as fh:
        writer.write(fh)


async def save_async(figs: Iterable[go.Figure], path: str | Path) -> None:
    figs_list = list(figs)
    await run_with_browser_png_cache(figs_list, lambda: save(figs_list, path))
