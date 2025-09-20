from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple, cast

import yaml  # type: ignore[import-untyped]
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
    def _check_assets_and_portfolios(self) -> "Scenario":
        asset_ids = [a.id for a in self.assets]
        dup_assets = [i for i, c in Counter(asset_ids).items() if c > 1]
        if dup_assets:
            raise ValueError(f"duplicate asset ids: {sorted(dup_assets)}")

        port_ids = [p.id for p in self.portfolios]
        dup_ports = [i for i, c in Counter(port_ids).items() if c > 1]
        if dup_ports:
            raise ValueError(f"duplicate portfolio ids: {sorted(dup_ports)}")

        asset_id_set = set(asset_ids)
        for p in self.portfolios:
            unknown = set(p.weights) - asset_id_set
            if unknown:
                raise ValueError(
                    f"portfolio {p.id} references unknown assets: {sorted(unknown)}"
                )
        return self

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
        extra = provided - expected
        if extra:
            raise ValueError(f"unexpected correlations for pairs: {sorted(extra)}")
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
    return cast(Scenario, Scenario.model_validate(data))


def save_scenario(scenario: Scenario, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(scenario.model_dump()))
