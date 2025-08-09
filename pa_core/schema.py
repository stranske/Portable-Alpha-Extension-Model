from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from pydantic import BaseModel, field_validator, model_validator


class Index(BaseModel):
    id: str
    label: str | None = None
    mu: float
    sigma: float


class Asset(BaseModel):
    id: str
    label: str | None = None
    mu: float
    sigma: float


class Correlation(BaseModel):
    pair: Tuple[str, str]
    rho: float

    @field_validator("rho")
    @classmethod
    def _check_rho(cls, v: float) -> float:
        if not -0.999 <= v <= 0.999:
        if not CORRELATION_LOWER_BOUND <= v <= CORRELATION_UPPER_BOUND:
            raise ValueError(f"rho must be between {CORRELATION_LOWER_BOUND} and {CORRELATION_UPPER_BOUND}")
        return v


class Portfolio(BaseModel):
    id: str
    weights: Dict[str, float]

    @model_validator(mode="after")
    def _check_weights(self) -> "Portfolio":
        total = sum(self.weights.values())
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError("portfolio weights must sum to 1")
        return self


class Sleeve(BaseModel):
    alpha_source: str
    capital_share: float
    theta: float | None = None
    active_share: float | None = None


class Scenario(BaseModel):
    index: Index
    assets: List[Asset] = []
    correlations: List[Correlation] = []
    portfolios: List[Portfolio] = []
    sleeves: Dict[str, Sleeve] | None = None

    @model_validator(mode="after")
    def _check_correlations(self) -> "Scenario":
        ids = [self.index.id] + [a.id for a in self.assets]
        expected = {tuple(sorted(p)) for p in combinations(ids, 2)}
        provided = {tuple(sorted(c.pair)) for c in self.correlations}
        missing = expected - provided
        if missing:
            raise ValueError(f"missing correlations for pairs: {sorted(missing)}")
        return self


def load_scenario(path: str | Path) -> Scenario:
    data = yaml.safe_load(Path(path).read_text())
    return Scenario.model_validate(data)
