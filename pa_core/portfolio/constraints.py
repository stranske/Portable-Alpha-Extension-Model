from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


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

_WEIGHT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ConstraintValidator:
    """Validate portfolio weights against configured constraints."""

    weight_bounds: WeightBoundsConstraint | None = None
    max_leverage: float | None = None
    max_concentration: float | None = None

    def validate(
        self,
        weights: Mapping[str, float],
        *,
        portfolio_id: str | None = None,
    ) -> list[ConstraintViolation]:
        violations: list[ConstraintViolation] = []
        if self.weight_bounds:
            min_weight = self.weight_bounds.min_weight
            max_weight = self.weight_bounds.max_weight
            for asset, weight in weights.items():
                if weight < min_weight - _WEIGHT_TOLERANCE:
                    violations.append(
                        ConstraintViolation(
                            constraint_type="weight_bounds",
                            message=(
                                f"{asset} weight {weight:.2%} is below "
                                f"min {min_weight:.2%}."
                            ),
                            details={
                                "asset": asset,
                                "weight": weight,
                                "min_weight": min_weight,
                                "max_weight": max_weight,
                                "portfolio_id": portfolio_id,
                            },
                        )
                    )
                if weight > max_weight + _WEIGHT_TOLERANCE:
                    violations.append(
                        ConstraintViolation(
                            constraint_type="weight_bounds",
                            message=(
                                f"{asset} weight {weight:.2%} exceeds "
                                f"max {max_weight:.2%}."
                            ),
                            details={
                                "asset": asset,
                                "weight": weight,
                                "min_weight": min_weight,
                                "max_weight": max_weight,
                                "portfolio_id": portfolio_id,
                            },
                        )
                    )

        if self.max_leverage is not None:
            gross_exposure = sum(abs(weight) for weight in weights.values())
            if gross_exposure > self.max_leverage + _WEIGHT_TOLERANCE:
                violations.append(
                    ConstraintViolation(
                        constraint_type="leverage",
                        message=(
                            f"Gross exposure {gross_exposure:.2f} exceeds "
                            f"max leverage {self.max_leverage:.2f}."
                        ),
                        details={
                            "gross_exposure": gross_exposure,
                            "max_leverage": self.max_leverage,
                            "portfolio_id": portfolio_id,
                        },
                    )
                )

        if self.max_concentration is not None:
            for asset, weight in weights.items():
                if abs(weight) > self.max_concentration + _WEIGHT_TOLERANCE:
                    violations.append(
                        ConstraintViolation(
                            constraint_type="concentration",
                            message=(
                                f"{asset} weight {weight:.2%} exceeds "
                                f"concentration limit {self.max_concentration:.2%}."
                            ),
                            details={
                                "asset": asset,
                                "weight": weight,
                                "max_weight": self.max_concentration,
                                "portfolio_id": portfolio_id,
                            },
                        )
                    )

        return violations


__all__ = [
    "ConstraintViolation",
    "ConstraintValidator",
    "WeightBoundsConstraint",
    "COMMON_WEIGHT_BOUNDS",
    "suggest_constraint_fixes",
]
