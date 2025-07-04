from __future__ import annotations


def get(name: str = "default") -> dict:
    """Return a dummy dashboard layout template."""
    return {
        "tabs": ["Headline", "Funding fan", "Diagnostics"],
        "name": name,
    }
