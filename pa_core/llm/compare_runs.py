"""Run-to-run LLM comparison helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def load_prior_manifest(
    manifest_data: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, Path | None]:
    """Load the prior-run manifest referenced by ``manifest_data['previous_run']``.

    Returns ``(prior_manifest, prior_manifest_path)``. If no usable previous-run
    reference is present, returns ``(None, None)``. If a path is present but
    missing or not a file, returns ``(None, path)``.

    Notes
    -----
    This function intentionally raises read/parse errors (for example
    ``PermissionError`` and ``json.JSONDecodeError``) so callers can decide how
    to surface unreadable artifact details.
    """

    if manifest_data is None:
        return None, None

    prev_ref = manifest_data.get("previous_run")
    if not isinstance(prev_ref, str) or not prev_ref.strip():
        return None, None

    prev_manifest_path = Path(prev_ref).expanduser()
    if not prev_manifest_path.exists() or not prev_manifest_path.is_file():
        return None, prev_manifest_path

    loaded = json.loads(prev_manifest_path.read_text())
    if not isinstance(loaded, dict):
        return None, prev_manifest_path
    return loaded, prev_manifest_path

