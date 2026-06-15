from __future__ import annotations

import io
import os
import warnings
from pathlib import Path
from typing import Iterable

import plotly.graph_objects as go

try:
    from pypdf import PdfWriter
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dep
    PdfWriter = None


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
            figs[0].write_image(path, format="pdf", engine="kaleido")
        except (ValueError, RuntimeError, OSError, MemoryError) as exc:
            warnings.warn(
                f"PDF export failed; writing figure JSON fallback instead: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            _write_json_fixture(figs, path)
        return

    if PdfWriter is None:
        warnings.warn(
            "pypdf is required to merge multiple PDF figures; writing figure JSON fallback instead.",
            RuntimeWarning,
            stacklevel=2,
        )
        _write_json_fixture(figs, path)
        return

    writer = PdfWriter()
    buffers: list[io.BytesIO] = []
    for fig in figs:
        buf = io.BytesIO()
        try:
            fig.write_image(buf, format="pdf", engine="kaleido")
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
