"""Adopt the shared design system (vendored at ../design-system).

Streamlit injects CSS per page-run and it does not persist across pages, so call
``apply_theme()`` once near the top of every page/entry-point. It is a safe no-op
when the kit is absent (bare/test environments).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOGGER = logging.getLogger(__name__)
_DS_DIR = Path(__file__).resolve().parents[1] / "design-system"
_ds = None

if _DS_DIR.is_dir():
    if str(_DS_DIR) not in sys.path:
        sys.path.insert(0, str(_DS_DIR))
    try:  # the kit is vendored by the Workflows design-system sync (sync-manifest.yml)
        import ds_streamlit as _ds
    except ModuleNotFoundError as exc:  # pragma: no cover - kit unavailable in bare/test mode
        if exc.name != "ds_streamlit":
            raise


def apply_theme() -> None:
    """Inject the shared Ink & Air theme. No-op if the design-system kit is absent."""
    if _ds is not None:
        try:
            _ds.inject_theme()
        except Exception:
            # Theming is cosmetic; never let it break a page render.
            _LOGGER.exception("Failed to inject shared Streamlit theme")


def ds():
    """Return the design-system Streamlit helper module (or None if unavailable)."""
    return _ds
