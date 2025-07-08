from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class RunFlags:
    """Flags controlling report export and dashboard launch."""

    save_xlsx: str | None = "Outputs.xlsx"
    png: bool = False
    pdf: bool = False
    pptx: bool = False
    html: bool = False
    gif: bool = False
    dashboard: bool = False
    alt_text: str | None = None
