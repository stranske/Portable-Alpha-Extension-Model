from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

from collections import Counter
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


CORRELATION_LOWER_BOUND = -0.999
CORRELATION_UPPER_BOUND = 0.999
WEIGHT_SUM_TOLERANCE = 1e-6


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
        if not CORRELATION_LOWER_BOUND <= v <= CORRELATION_UPPER_BOUND:
            raise ValueError(
                f"rho must be between {CORRELATION_LOWER_BOUND} and {CORRELATION_UPPER_BOUND}"
            )
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
    assets: List[Asset] = Field(default_factory=list)
    correlations: List[Correlation] = Field(default_factory=list)
    portfolios: List[Portfolio] = Field(default_factory=list)
    sleeves: Dict[str, Sleeve] | None = None

    @model_validator(mode="after")
    def _check_correlations(self) -> "Scenario":
        ids = [self.index.id] + [a.id for a in self.assets]
        expected = {tuple(sorted(p)) for p in combinations(ids, 2)}
        pairs = [tuple(sorted(c.pair)) for c in self.correlations]
        dupes = [p for p, count in Counter(pairs).items() if count > 1]
        if dupes:
            raise ValueError(f"duplicate correlations for pairs: {sorted(dupes)}")
        provided = set(pairs)
        missing = expected - provided
        if missing:
            raise ValueError(f"missing correlations for pairs: {sorted(missing)}")
        return self

    @model_validator(mode="after")
    def _check_sleeves(self) -> "Scenario":
        if self.sleeves:
            total = sum(s.capital_share for s in self.sleeves.values())
            if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
                raise ValueError("sleeves capital_share must sum to 1")
        return self


def load_scenario(path: str | Path) -> Scenario:
    data = yaml.safe_load(Path(path).read_text())
    return Scenario.model_validate(data)
