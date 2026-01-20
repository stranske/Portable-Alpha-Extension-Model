from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WeightBoundsConstraint:
    """Per-asset weight bounds used for portfolio validation."""

    name: str
    min_weight: float
    max_weight: float
    description: str

    def __post_init__(self) -> None:
        if self.min_weight > self.max_weight:
            raise ValueError("min_weight must be <= max_weight")


COMMON_WEIGHT_BOUNDS: dict[str, WeightBoundsConstraint] = {
    "long_only": WeightBoundsConstraint(
        name="Long-only",
        min_weight=0.0,
        max_weight=1.0,
        description="Weights must be between 0 and 1 (no short positions).",
    ),
    "long_short_130_30": WeightBoundsConstraint(
        name="130/30",
        min_weight=-0.3,
        max_weight=1.3,
        description="Allows 30% short exposure with up to 130% long exposure.",
    ),
}


__all__ = ["WeightBoundsConstraint", "COMMON_WEIGHT_BOUNDS"]
