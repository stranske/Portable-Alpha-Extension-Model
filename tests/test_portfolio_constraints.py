from pa_core.portfolio.constraints import ConstraintViolation, suggest_constraint_fixes


def test_suggest_constraint_fixes_weight_bounds_above_max():
    violation = ConstraintViolation(
        constraint_type="weight_bounds",
        message="weight above max",
        details={"asset": "AssetA", "weight": 1.2, "min_weight": 0.0, "max_weight": 1.0},
    )

    suggestions = suggest_constraint_fixes(violation)

    assert "Reduce AssetA weight from 120.00% to 100.00% or below." in suggestions


def test_suggest_constraint_fixes_weight_bounds_below_min():
    violation = ConstraintViolation(
        constraint_type="weight_bounds",
        message="weight below min",
        details={"asset": "AssetB", "weight": -0.2, "min_weight": -0.1, "max_weight": 0.5},
    )

    suggestions = suggest_constraint_fixes(violation)

    assert "Increase AssetB weight from -20.00% to -10.00% or above." in suggestions


def test_suggest_constraint_fixes_leverage():
    violation = ConstraintViolation(
        constraint_type="leverage",
        message="gross exposure too high",
        details={"gross_exposure": 1.8, "max_leverage": 1.5},
    )

    suggestions = suggest_constraint_fixes(violation)

    assert "Scale all weights by 0.83 to reduce gross exposure from 1.80 to 1.50." in suggestions
    assert "Reduce short or long positions to bring leverage within limits." in suggestions


def test_suggest_constraint_fixes_concentration():
    violation = ConstraintViolation(
        constraint_type="concentration",
        message="single asset too large",
        details={"asset": "AssetC", "weight": 0.45, "max_weight": 0.3},
    )

    suggestions = suggest_constraint_fixes(violation)

    assert "Reduce AssetC weight from 45.00% to 30.00% or below." in suggestions
    assert "Spread allocation across more assets to reduce concentration." in suggestions


def test_suggest_constraint_fixes_fallback():
    violation = ConstraintViolation(
        constraint_type="unknown",
        message="unknown constraint",
    )

    suggestions = suggest_constraint_fixes(violation)

    assert suggestions == ["Review the constraint settings and adjust weights to satisfy limits."]
