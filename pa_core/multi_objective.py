from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

MetricName = Literal[
    "AnnReturn",
    "ExcessReturn",
    "TE",
    "BreachProb",
    "CVaR",
    "ShortfallProb",
]
ObjectiveDirection = Literal["max", "min"]
ConstraintScope = Literal["sleeves", "total", "both"]


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    metric: MetricName
    direction: ObjectiveDirection
    weight: float | None = None


@dataclass(frozen=True)
class ConstraintSpec:
    metric: MetricName
    limit: float
    scope: ConstraintScope
    absolute: bool = False

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ValueError("constraint limit must be non-negative")


@dataclass(frozen=True)
class MultiObjectiveProblem:
    objectives: tuple[ObjectiveSpec, ...]
    constraints: tuple[ConstraintSpec, ...]

    def metrics(self) -> tuple[MetricName, ...]:
        metrics: list[MetricName] = [obj.metric for obj in self.objectives]
        metrics.extend(constraint.metric for constraint in self.constraints)
        seen: set[MetricName] = set()
        ordered: list[MetricName] = []
        for metric in metrics:
            if metric in seen:
                continue
            seen.add(metric)
            ordered.append(metric)
        return tuple(ordered)


def build_sleeve_multi_objective_problem(
    *,
    return_metric: MetricName = "AnnReturn",
    max_te: float,
    max_breach: float,
    max_cvar: float,
    max_shortfall: float,
    constraint_scope: ConstraintScope,
    additional_objectives: Iterable[ObjectiveSpec] | None = None,
) -> MultiObjectiveProblem:
    if return_metric not in {"AnnReturn", "ExcessReturn"}:
        raise ValueError("return_metric must be AnnReturn or ExcessReturn")
    objectives = [ObjectiveSpec(name="expected_return", metric=return_metric, direction="max")]
    if additional_objectives is not None:
        objectives.extend(additional_objectives)
    constraints = (
        ConstraintSpec(metric="TE", limit=max_te, scope=constraint_scope),
        ConstraintSpec(metric="BreachProb", limit=max_breach, scope=constraint_scope),
        ConstraintSpec(metric="CVaR", limit=max_cvar, scope=constraint_scope, absolute=True),
        ConstraintSpec(metric="ShortfallProb", limit=max_shortfall, scope=constraint_scope),
    )
    return MultiObjectiveProblem(
        objectives=tuple(objectives),
        constraints=constraints,
    )
