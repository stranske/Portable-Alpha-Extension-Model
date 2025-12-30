from __future__ import annotations

from typing import Any


def get(name: str = "default") -> dict[str, Any]:
    """Return a dummy dashboard layout template."""
    return {
        "tabs": ["Headline", "Funding fan", "Diagnostics"],
        "name": name,
    }
