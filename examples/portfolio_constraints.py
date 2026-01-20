from __future__ import annotations

from pa_core.portfolio import COMMON_WEIGHT_BOUNDS, ConstraintViolation, suggest_constraint_fixes


def main() -> None:
    weights = {"AssetA": 1.2, "AssetB": -0.2}
    bounds = COMMON_WEIGHT_BOUNDS["long_only"]

    violation = ConstraintViolation(
        constraint_type="weight_bounds",
        message="weight above max",
        details={
            "asset": "AssetA",
            "weight": weights["AssetA"],
            "min_weight": bounds.min_weight,
            "max_weight": bounds.max_weight,
        },
    )

    for suggestion in suggest_constraint_fixes(violation):
        print(f"- {suggestion}")


if __name__ == "__main__":
    main()
