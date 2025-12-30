from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, cast

import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]

from ..schema import (
    CORRELATION_LOWER_BOUND,
    CORRELATION_UPPER_BOUND,
    Asset,
    Correlation,
    Index,
)

MONTHS_PER_YEAR = 12
VOLATILITY_ANNUALIZATION_FACTOR = MONTHS_PER_YEAR**0.5


@dataclass
class CalibrationDiagnostics:
    covariance_shrinkage: Literal["none", "ledoit_wolf"]
    shrinkage_intensity: float | None
    vol_regime: Literal["single", "two_state"]
    vol_regime_window: int | None
    vol_regime_state: Dict[str, str]


@dataclass
class CalibrationResult:
    index: Index
    assets: List[Asset]
    correlations: List[Correlation]
    diagnostics: CalibrationDiagnostics | None = None


def _ledoit_wolf_shrinkage(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Return Ledoit-Wolf shrunk covariance and shrinkage intensity.

    The target is the identity scaled by the average variance. Input returns
    should be shaped (n_samples, n_features) and already numeric.
    """

    n_samples, n_features = returns.shape
    if n_samples <= 1 or n_features == 0:
        return np.cov(returns, rowvar=False, bias=True), 0.0

    centered = returns - returns.mean(axis=0, keepdims=True)
    sample_cov = (centered.T @ centered) / n_samples

    mu = np.trace(sample_cov) / n_features
    target = mu * np.eye(n_features)
    delta = sample_cov - target
    delta_norm2 = float(np.sum(delta**2))
    if delta_norm2 == 0.0:
        return sample_cov, 0.0

    squared = centered**2
    beta_matrix = (squared.T @ squared) / n_samples - sample_cov**2
    beta = float(np.sum(beta_matrix)) / n_samples
    beta = min(beta, delta_norm2)

    shrinkage = 0.0 if delta_norm2 == 0.0 else beta / delta_norm2
    shrunk_cov = (1 - shrinkage) * sample_cov + shrinkage * target
    return shrunk_cov, float(shrinkage)


class CalibrationAgent:
    def __init__(
        self,
        min_obs: int = 36,
        *,
        covariance_shrinkage: Literal["none", "ledoit_wolf"] = "none",
        vol_regime: Literal["single", "two_state"] = "single",
        vol_regime_window: int = 12,
    ) -> None:
        self.min_obs = min_obs
        self.covariance_shrinkage = covariance_shrinkage
        self.vol_regime = vol_regime
        self.vol_regime_window = vol_regime_window

    def calibrate(self, df: pd.DataFrame, index_id: str) -> CalibrationResult:
        counts = cast(pd.Series, df.groupby("id")["return"].count())
        filtered = cast(pd.Series, counts[counts >= self.min_obs])
        valid_ids = cast(pd.Index, filtered.index).tolist()
        df = cast(pd.DataFrame, df[df["id"].isin(valid_ids)].copy())
        pivot = df.pivot(index="date", columns="id", values="return")
        if self.covariance_shrinkage == "ledoit_wolf":
            pivot = pivot.dropna()
            if pivot.empty:
                raise ValueError(
                    "insufficient data after aligning returns for shrinkage"
                )
            returns = pivot.to_numpy(dtype=float)
            cov, shrinkage = _ledoit_wolf_shrinkage(returns)
            base_sigma = pd.Series(
                np.sqrt(np.diag(cov)) * VOLATILITY_ANNUALIZATION_FACTOR,
                index=pivot.columns,
            )
            mu = cast(pd.Series, pivot.mean()) * MONTHS_PER_YEAR
            sds = np.sqrt(np.diag(cov))
            denom = np.outer(sds, sds)
            with np.errstate(divide="ignore", invalid="ignore"):
                corr_mat = np.where(denom > 0, cov / denom, 0.0)
            corr = pd.DataFrame(corr_mat, index=pivot.columns, columns=pivot.columns)
        else:
            grouped = df.groupby("id")["return"]
            mu = cast(pd.Series, grouped.mean()) * MONTHS_PER_YEAR
            base_sigma = (
                cast(pd.Series, grouped.std(ddof=1)) * VOLATILITY_ANNUALIZATION_FACTOR
            )
            corr = pivot.corr()
            shrinkage = None
        if index_id not in mu.index:
            raise ValueError("index_id not present in data")
        regime_state: Dict[str, str] = {}
        sigma = base_sigma.copy()
        regime_window: int | None = None
        if self.vol_regime == "two_state":
            if self.vol_regime_window <= 1:
                raise ValueError("vol_regime_window must be > 1 for two_state regime")
            recent = pivot.tail(self.vol_regime_window)
            if not recent.empty:
                recent_sigma = recent.std(ddof=1) * VOLATILITY_ANNUALIZATION_FACTOR
                for asset_id in base_sigma.index:
                    recent_val = float(recent_sigma.get(asset_id, np.nan))
                    base_val = float(base_sigma.get(asset_id, np.nan))
                    if np.isnan(recent_val) or np.isnan(base_val):
                        continue
                    if recent_val >= base_val:
                        sigma.loc[asset_id] = recent_val
                        regime_state[asset_id] = "high"
                    else:
                        regime_state[asset_id] = "low"
                regime_window = int(min(self.vol_regime_window, len(recent)))

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
        pairs: List[Correlation] = []
        ids = list(corr.columns)
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                rho = float(corr.loc[a, b])
                if not np.isfinite(rho):
                    rho = 0.0
                else:
                    rho = float(
                        np.clip(rho, CORRELATION_LOWER_BOUND, CORRELATION_UPPER_BOUND)
                    )
                pairs.append(Correlation(pair=(a, b), rho=rho))
        diagnostics = CalibrationDiagnostics(
            covariance_shrinkage=self.covariance_shrinkage,
            shrinkage_intensity=shrinkage,
            vol_regime=self.vol_regime,
            vol_regime_window=regime_window,
            vol_regime_state=regime_state,
        )
        return CalibrationResult(
            index=index_obj, assets=assets, correlations=pairs, diagnostics=diagnostics
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
