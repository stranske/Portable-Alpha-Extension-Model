from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, cast

import pandas as pd
import yaml

from ..schema import Asset, Correlation, Index


MONTHS_PER_YEAR = 12
VOLATILITY_ANNUALIZATION_FACTOR = MONTHS_PER_YEAR**0.5


@dataclass
class CalibrationResult:
    index: Index
    assets: List[Asset]
    correlations: List[Correlation]


class CalibrationAgent:
    def __init__(self, min_obs: int = 36) -> None:
        self.min_obs = min_obs

    def calibrate(self, df: pd.DataFrame, index_id: str) -> CalibrationResult:
        counts = cast(pd.Series, df.groupby("id")["return"].count())
        filtered = cast(pd.Series, counts[counts >= self.min_obs])
        valid_ids = cast(pd.Index, filtered.index).tolist()
        df = cast(pd.DataFrame, df[df["id"].isin(valid_ids)].copy())
        grouped = df.groupby("id")["return"]
        mu = cast(pd.Series, grouped.mean()) * MONTHS_PER_YEAR
        sigma = cast(pd.Series, grouped.std(ddof=1)) * VOLATILITY_ANNUALIZATION_FACTOR
        if index_id not in mu.index:
            raise ValueError("index_id not present in data")
        index_obj = Index(
            id=index_id,
            label=index_id,
            mu=float(mu[index_id]),
            sigma=float(sigma[index_id]),
        )
        assets = [
            Asset(id=i, label=i, mu=float(mu[i]), sigma=float(sigma[i]))
            for i in mu.index
        ]
        pivot = df.pivot(index="date", columns="id", values="return")
        corr = pivot.corr()
        pairs: List[Correlation] = []
        ids = list(corr.columns)
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                pairs.append(Correlation(pair=(a, b), rho=float(corr.loc[a, b])))
        return CalibrationResult(index=index_obj, assets=assets, correlations=pairs)

    def to_yaml(self, result: CalibrationResult, path: str | Path) -> None:
        data = {
            "index": result.index.model_dump(),
            "assets": [a.model_dump() for a in result.assets],
            "correlations": [
                {"pair": list(c.pair), "rho": c.rho} for c in result.correlations
            ],
        }
        Path(path).write_text(yaml.safe_dump(data))
