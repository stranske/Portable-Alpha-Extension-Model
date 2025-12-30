"""Shared dashboard utility helpers."""

from __future__ import annotations


def normalize_share(value: float | None) -> float | None:
    """Normalize percentage-style inputs to a 0..1 fraction."""
    if value is None:
        return None
    if value > 1.0:
        return value / 100.0
    return value


__all__ = ["normalize_share"]
