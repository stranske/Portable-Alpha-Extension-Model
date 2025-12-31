"""Shared utilities for normalizing percentage-style inputs."""

from __future__ import annotations

SHARE_MIN = 0.0
SHARE_MAX = 1.0
SHARE_SUM_TOLERANCE = 1e-6


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
    # Treat values >= 2 as percentage-style inputs; 1.x is treated as a raw share.
    if 2.0 <= numeric <= 100.0:
        return numeric / 100.0
    return numeric
