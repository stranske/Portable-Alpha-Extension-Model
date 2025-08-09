from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

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
        counts = df.groupby("id")["return"].count()
        valid_ids = counts[counts >= self.min_obs].index
        df = df[df["id"].isin(valid_ids)].copy()
        grouped = df.groupby("id")["return"]
        mu = grouped.mean() * MONTHS_PER_YEAR
        sigma = grouped.std(ddof=1) * VOLATILITY_ANNUALIZATION_FACTOR
        assets = [
            Asset(id=i, label=i, mu=float(mu[i]), sigma=float(sigma[i]))
            for i in mu.index
        ]
        if index_id not in mu.index:
            raise ValueError("index_id not present in data")
        index_asset = next(a for a in assets if a.id == index_id)
        other_assets = [a for a in assets if a.id != index_id]
        pivot = df.pivot(index="date", columns="id", values="return")
        corr = pivot.corr()
        pairs: List[Correlation] = []
        ids = list(corr.columns)
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                pairs.append(Correlation(pair=(a, b), rho=float(corr.loc[a, b])))
        return CalibrationResult(
            index=index_asset, assets=other_assets, correlations=pairs
        )

    def to_yaml(self, result: CalibrationResult, path: str | Path) -> None:
        data = {
            "index": result.index.model_dump(),
            "assets": [a.model_dump() for a in result.assets],
            "correlations": [
                {"pair": list(c.pair), "rho": c.rho} for c in result.correlations
            ],
        }
        Path(path).write_text(yaml.safe_dump(data))
