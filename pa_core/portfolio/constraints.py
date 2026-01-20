from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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


@dataclass(frozen=True)
class ConstraintViolation:
    """Represents a portfolio constraint violation with context details."""

    constraint_type: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def suggest_constraint_fixes(violation: ConstraintViolation) -> list[str]:
    """Return suggested fixes for a constraint violation."""

    suggestions: list[str] = []
    details = violation.details
    constraint_type = violation.constraint_type

    if constraint_type == "weight_bounds":
        asset = details.get("asset")
        weight = details.get("weight")
        min_weight = details.get("min_weight")
        max_weight = details.get("max_weight")
        if asset and weight is not None and max_weight is not None and weight > max_weight:
            suggestions.append(
                f"Reduce {asset} weight from {weight:.2%} to {max_weight:.2%} or below."
            )
        if asset and weight is not None and min_weight is not None and weight < min_weight:
            suggestions.append(
                f"Increase {asset} weight from {weight:.2%} to {min_weight:.2%} or above."
            )
        if min_weight is not None and max_weight is not None:
            suggestions.append(
                f"Clamp weights within {min_weight:.2%} to {max_weight:.2%} and rebalance to sum to 100%."
            )

    if constraint_type == "leverage":
        gross_exposure = details.get("gross_exposure")
        max_leverage = details.get("max_leverage")
        if gross_exposure and max_leverage and gross_exposure > 0:
            scale = max_leverage / gross_exposure
            suggestions.append(
                "Scale all weights by "
                f"{scale:.2f} to reduce gross exposure from {gross_exposure:.2f} to {max_leverage:.2f}."
            )
        suggestions.append("Reduce short or long positions to bring leverage within limits.")

    if constraint_type == "concentration":
        asset = details.get("asset")
        max_weight = details.get("max_weight")
        weight = details.get("weight")
        if asset and max_weight is not None:
            if weight is not None:
                suggestions.append(
                    f"Reduce {asset} weight from {weight:.2%} to {max_weight:.2%} or below."
                )
            else:
                suggestions.append(f"Reduce {asset} weight to {max_weight:.2%} or below.")
        suggestions.append("Spread allocation across more assets to reduce concentration.")

    if not suggestions:
        suggestions.append("Review the constraint settings and adjust weights to satisfy limits.")

    return suggestions


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


__all__ = [
    "ConstraintViolation",
    "WeightBoundsConstraint",
    "COMMON_WEIGHT_BOUNDS",
    "suggest_constraint_fixes",
]
