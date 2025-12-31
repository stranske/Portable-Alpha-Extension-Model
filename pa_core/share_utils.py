"""Shared utilities for normalizing percentage-style inputs."""

from __future__ import annotations


def normalize_share(value: float | None) -> float | None:
    """Normalize percentage-style inputs to a 0..1 fraction.

    Used by both :class:`pa_core.config.ModelConfig` and
    :class:`pa_core.schema.Scenario` to keep share validation consistent.
    """
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if 1.0 < numeric <= 100.0:
        return numeric / 100.0
    return numeric
