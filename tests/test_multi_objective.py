import pytest

from pa_core.multi_objective import build_sleeve_multi_objective_problem


def test_build_sleeve_multi_objective_problem_defaults() -> None:
    problem = build_sleeve_multi_objective_problem(
        max_te=0.1,
        max_breach=0.2,
        max_cvar=0.3,
        max_shortfall=0.4,
        constraint_scope="total",
    )

    assert len(problem.objectives) == 1
    assert problem.objectives[0].metric == "AnnReturn"
    assert problem.objectives[0].direction == "max"

    constraints = {(c.metric, c.limit, c.scope, c.absolute) for c in problem.constraints}
    assert ("TE", 0.1, "total", False) in constraints
    assert ("BreachProb", 0.2, "total", False) in constraints
    assert ("CVaR", 0.3, "total", True) in constraints
    assert ("ShortfallProb", 0.4, "total", False) in constraints


def test_build_sleeve_multi_objective_problem_rejects_negative_limits() -> None:
    with pytest.raises(ValueError):
        build_sleeve_multi_objective_problem(
            max_te=-0.1,
            max_breach=0.2,
            max_cvar=0.3,
            max_shortfall=0.4,
            constraint_scope="sleeves",
        )
