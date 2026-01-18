"""Common portfolio constraint definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WeightBoundsConstraint:
    """Minimum and maximum allowed weight per asset."""

    min_weight: float
    max_weight: float

    def __post_init__(self) -> None:
        if self.min_weight > self.max_weight:
            raise ValueError("min_weight must be <= max_weight")


@dataclass(frozen=True)
class LeverageConstraint:
    """Maximum allowed gross leverage."""

    max_gross_leverage: float

    def __post_init__(self) -> None:
        if self.max_gross_leverage <= 0:
            raise ValueError("max_gross_leverage must be positive")


@dataclass(frozen=True)
class ConcentrationConstraint:
    """Limits on single-name and top-N concentration."""

    max_single_weight: float
    max_top_n_weight: float
    top_n: int = 5

    def __post_init__(self) -> None:
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        if self.max_single_weight <= 0:
            raise ValueError("max_single_weight must be positive")
        if self.max_top_n_weight <= 0:
            raise ValueError("max_top_n_weight must be positive")
        if self.max_single_weight > self.max_top_n_weight:
            raise ValueError("max_single_weight must be <= max_top_n_weight")


@dataclass(frozen=True)
class PortfolioConstraints:
    """Container for common portfolio constraints."""

    weight_bounds: WeightBoundsConstraint
    leverage: LeverageConstraint
    concentration: ConcentrationConstraint


COMMON_WEIGHT_BOUNDS = WeightBoundsConstraint(min_weight=0.0, max_weight=1.0)
COMMON_LEVERAGE = LeverageConstraint(max_gross_leverage=1.0)
COMMON_CONCENTRATION = ConcentrationConstraint(
    max_single_weight=0.2,
    max_top_n_weight=0.6,
    top_n=5,
)
COMMON_CONSTRAINTS = PortfolioConstraints(
    weight_bounds=COMMON_WEIGHT_BOUNDS,
    leverage=COMMON_LEVERAGE,
    concentration=COMMON_CONCENTRATION,
)

__all__ = [
    "WeightBoundsConstraint",
    "LeverageConstraint",
    "ConcentrationConstraint",
    "PortfolioConstraints",
    "COMMON_WEIGHT_BOUNDS",
    "COMMON_LEVERAGE",
    "COMMON_CONCENTRATION",
    "COMMON_CONSTRAINTS",
]
