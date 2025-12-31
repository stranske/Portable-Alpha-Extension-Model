from __future__ import annotations

from typing import Any, Dict


def get(name: str = "default") -> Dict[str, Any]:
    """Return a dummy dashboard layout template."""
    return {
        "tabs": ["Headline", "Funding fan", "Diagnostics"],
        "name": name,
    }
