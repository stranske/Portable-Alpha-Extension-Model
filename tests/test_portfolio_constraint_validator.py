from __future__ import annotations

from pa_core.portfolio import COMMON_WEIGHT_BOUNDS, ConstraintValidator


def test_constraint_validator_reports_violations() -> None:
    validator = ConstraintValidator(
        weight_bounds=COMMON_WEIGHT_BOUNDS["long_only"],
        max_leverage=1.0,
        max_concentration=0.6,
    )
    weights = {"A": 1.2, "B": -0.2}

    violations = validator.validate(weights, portfolio_id="p1")

    messages = {violation.message for violation in violations}
    assert any("A weight 120.00% exceeds max 100.00%" in msg for msg in messages)
    assert any("B weight -20.00% is below min 0.00%" in msg for msg in messages)
    assert any("Gross exposure 1.40 exceeds max leverage 1.00" in msg for msg in messages)
    assert any("A weight 120.00% exceeds concentration limit 60.00%" in msg for msg in messages)
