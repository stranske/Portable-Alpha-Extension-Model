from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .schema import Asset, Correlation, Index, Scenario

MONTHS_PER_YEAR = 12
VOLATILITY_ANNUALIZATION_FACTOR = MONTHS_PER_YEAR**0.5


@dataclass(frozen=True)
class EstimateCI:
    lower: float
    upper: float


@dataclass(frozen=True)
class SeriesEstimate:
    mean: float
    volatility: float
    mean_ci: EstimateCI | None
    volatility_ci: EstimateCI | None
    n_obs: int


@dataclass(frozen=True)
class CorrelationEstimate:
    pair: tuple[str, str]
    rho: float
    ci: EstimateCI | None
    n_obs: int


@dataclass(frozen=True)
class CalibrationResult:
    index_id: str
    series: Mapping[str, SeriesEstimate]
    correlations: Sequence[CorrelationEstimate]
    corr_target: Literal["identity", "constant"]
    mean_shrinkage: float
    corr_shrinkage: float
    confidence_level: float
    annualize: bool

    def to_scenario(self) -> Scenario:
        index = self.series.get(self.index_id)
        if index is None:
            raise ValueError("index_id not found in series estimates")
        assets = [
            Asset(id=asset_id, label=asset_id, mu=stats.mean, sigma=stats.volatility)
            for asset_id, stats in self.series.items()
            if asset_id != self.index_id
        ]
        correlations = [Correlation(pair=corr.pair, rho=corr.rho) for corr in self.correlations]
        return Scenario(
            index=Index(
                id=self.index_id,
                label=self.index_id,
                mu=index.mean,
                sigma=index.volatility,
            ),
            assets=assets,
            correlations=correlations,
        )

    def to_model_config(self, *, alpha_map: Mapping[str, str]) -> dict[str, float]:
        required = {"H", "E", "M"}
        missing = required - set(alpha_map)
        if missing:
            raise ValueError(f"alpha_map must include {sorted(missing)}")
        data: dict[str, float] = {}
        for key, series_id in alpha_map.items():
            stats = self.series.get(series_id)
            if stats is None:
                raise ValueError(f"series {series_id!r} not found in calibration result")
            data[f"mu_{key}"] = stats.mean
            data[f"sigma_{key}"] = stats.volatility
        pair_lookup = {tuple(sorted(c.pair)): c.rho for c in self.correlations}
        for key, series_id in alpha_map.items():
            pair = tuple(sorted((self.index_id, series_id)))
            if pair not in pair_lookup:
                raise ValueError(f"missing correlation for pair {pair}")
            data[f"rho_idx_{key}"] = pair_lookup[pair]
        pairs = [
            ("H", "E"),
            ("H", "M"),
            ("E", "M"),
        ]
        for a, b in pairs:
            pair = tuple(sorted((alpha_map[a], alpha_map[b])))
            if pair not in pair_lookup:
                raise ValueError(f"missing correlation for pair {pair}")
            data[f"rho_{a}_{b}"] = pair_lookup[pair]
        return data


def _boost_shrinkage_for_short_samples(
    shrinkage: float, *, n_samples: int, n_features: int, min_samples: int = 36
) -> float:
    min_samples = max(12, min_samples, 2 * n_features)
    if n_samples >= min_samples:
        return float(shrinkage)
    ratio = n_samples / float(min_samples)
    boosted = 1.0 - (1.0 - shrinkage) * ratio
    return float(min(1.0, max(shrinkage, boosted)))


def _c4(n_samples: int) -> float:
    if n_samples <= 1:
        return float("nan")
    return math.sqrt(2.0 / (n_samples - 1)) * math.exp(
        math.lgamma(n_samples / 2.0) - math.lgamma((n_samples - 1) / 2.0)
    )


def _chi2_ppf(p: float, df: int) -> float:
    if df <= 0:
        return float("nan")
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return float("inf")
    z = NormalDist().inv_cdf(p)
    a = 2.0 / (9.0 * df)
    return df * (1.0 - a + z * math.sqrt(a)) ** 3


def _mean_ci(
    mean: float,
    sigma: float,
    n_obs: int,
    *,
    confidence_level: float,
    annualize: bool,
) -> EstimateCI | None:
    if n_obs < 2 or not math.isfinite(sigma):
        return None
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    se = sigma / math.sqrt(n_obs)
    lower = mean - z * se
    upper = mean + z * se
    if annualize:
        lower *= MONTHS_PER_YEAR
        upper *= MONTHS_PER_YEAR
    return EstimateCI(lower=lower, upper=upper)


def _vol_ci(
    sample_variance: float,
    n_obs: int,
    *,
    confidence_level: float,
    annualize: bool,
) -> EstimateCI | None:
    if n_obs < 2 or sample_variance < 0.0 or not math.isfinite(sample_variance):
        return None
    df = n_obs - 1
    alpha = 1.0 - confidence_level
    chi2_lower = _chi2_ppf(alpha / 2.0, df)
    chi2_upper = _chi2_ppf(1.0 - alpha / 2.0, df)
    if chi2_lower <= 0.0 or not math.isfinite(chi2_upper):
        return None
    lower = math.sqrt(df * sample_variance / chi2_upper)
    upper = math.sqrt(df * sample_variance / chi2_lower)
    if annualize:
        lower *= VOLATILITY_ANNUALIZATION_FACTOR
        upper *= VOLATILITY_ANNUALIZATION_FACTOR
    return EstimateCI(lower=lower, upper=upper)


def _corr_ci(
    rho: float,
    n_obs: int,
    *,
    confidence_level: float,
) -> EstimateCI | None:
    if n_obs <= 3 or not math.isfinite(rho):
        return None
    rho = max(-0.999, min(0.999, rho))
    z = 0.5 * math.log((1.0 + rho) / (1.0 - rho))
    se = 1.0 / math.sqrt(n_obs - 3)
    z_crit = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    lower = math.tanh(z - z_crit * se)
    upper = math.tanh(z + z_crit * se)
    return EstimateCI(lower=lower, upper=upper)


def _annualize_mean(mean: float, *, annualize: bool) -> float:
    return mean * MONTHS_PER_YEAR if annualize else mean


def _annualize_vol(vol: float, *, annualize: bool) -> float:
    return vol * VOLATILITY_ANNUALIZATION_FACTOR if annualize else vol


def calibrate_returns(
    returns: pd.DataFrame,
    *,
    index_id: str,
    mean_shrinkage: float = 0.2,
    corr_shrinkage: float = 0.2,
    corr_target: Literal["identity", "constant"] = "identity",
    confidence_level: float = 0.95,
    annualize: bool = True,
) -> CalibrationResult:
    if returns.empty:
        raise ValueError("returns data is empty")
    if index_id not in returns.columns:
        raise ValueError("index_id not found in returns data")

    clean = returns.copy()
    clean = clean.apply(pd.to_numeric, errors="coerce")
    clean = clean.sort_index()

    n_obs_series = clean.count()
    sample_means = clean.mean()
    grand_mean_series = sample_means.dropna()
    if grand_mean_series.empty:
        grand_mean = float("nan")
    else:
        weights = n_obs_series.reindex(grand_mean_series.index).astype(float)
        valid_weights = weights.where(weights > 0.0)
        mask = valid_weights.notna()
        if not mask.any():
            grand_mean = float(grand_mean_series.mean())
        else:
            grand_mean = float(
                np.average(
                    grand_mean_series[mask].to_numpy(),
                    weights=valid_weights[mask].to_numpy(),
                )
            )

    mean_shrinkage = float(mean_shrinkage)
    corr_shrinkage = float(corr_shrinkage)
    if not 0.0 <= mean_shrinkage <= 1.0:
        raise ValueError("mean_shrinkage must be between 0 and 1")
    if not 0.0 <= corr_shrinkage <= 1.0:
        raise ValueError("corr_shrinkage must be between 0 and 1")
    if corr_target not in {"identity", "constant"}:
        raise ValueError("corr_target must be 'identity' or 'constant'")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    n_features = len(sample_means)
    mean_series: dict[str, SeriesEstimate] = {}
    for series_id in sample_means.index:
        n_obs = int(n_obs_series.get(series_id, 0))
        mean_boost = _boost_shrinkage_for_short_samples(
            mean_shrinkage, n_samples=n_obs, n_features=n_features
        )
        series_mean = float(sample_means[series_id])
        if not math.isfinite(series_mean):
            series_mean = grand_mean
        mean_value = (1.0 - mean_boost) * series_mean + mean_boost * grand_mean

        sample_std = float(clean[series_id].std(ddof=1))
        if math.isnan(sample_std) or n_obs < 2:
            sigma = float("nan")
            sample_var = float("nan")
        else:
            sample_var = sample_std**2
            correction = _c4(n_obs)
            sigma = sample_std / correction if math.isfinite(correction) else sample_std

        mean_ci = _mean_ci(
            float(mean_value),
            float(sample_std),
            n_obs,
            confidence_level=confidence_level,
            annualize=annualize,
        )
        vol_ci = _vol_ci(
            float(sample_var),
            n_obs,
            confidence_level=confidence_level,
            annualize=annualize,
        )
        mean_series[series_id] = SeriesEstimate(
            mean=_annualize_mean(float(mean_value), annualize=annualize),
            volatility=_annualize_vol(float(sigma), annualize=annualize),
            mean_ci=mean_ci,
            volatility_ci=vol_ci,
            n_obs=n_obs,
        )

    sample_corr = clean.corr(min_periods=2)
    if corr_target == "identity":
        target = pd.DataFrame(
            np.eye(n_features), index=sample_corr.index, columns=sample_corr.columns
        )
    else:
        off_diag = sample_corr.where(~np.eye(n_features, dtype=bool))
        off_diag_values = off_diag.stack().dropna()
        constant = float(off_diag_values.mean()) if not off_diag_values.empty else 0.0
        target = pd.DataFrame(
            np.full((n_features, n_features), constant),
            index=sample_corr.index,
            columns=sample_corr.columns,
        )
        np.fill_diagonal(target.values, 1.0)
    sample_corr = sample_corr.where(~sample_corr.isna(), target)

    effective_n = int(n_obs_series.min()) if not n_obs_series.empty else 0
    corr_boost = _boost_shrinkage_for_short_samples(
        corr_shrinkage, n_samples=effective_n, n_features=n_features
    )
    corr_shrunk = (1.0 - corr_boost) * sample_corr + corr_boost * target
    np.fill_diagonal(corr_shrunk.values, 1.0)

    correlations: list[CorrelationEstimate] = []
    ids = list(sample_corr.columns)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            pair_returns = clean[[a, b]].dropna()
            n_pair = int(len(pair_returns))
            rho = float(corr_shrunk.loc[a, b])
            rho = max(-0.999, min(0.999, rho))
            ci = _corr_ci(rho, n_pair, confidence_level=confidence_level)
            correlations.append(CorrelationEstimate(pair=(a, b), rho=rho, ci=ci, n_obs=n_pair))

    return CalibrationResult(
        index_id=index_id,
        series=mean_series,
        correlations=correlations,
        corr_target=corr_target,
        mean_shrinkage=mean_shrinkage,
        corr_shrinkage=corr_shrinkage,
        confidence_level=confidence_level,
        annualize=annualize,
    )


def build_calibration_report(
    result: CalibrationResult,
    *,
    alpha_map: Mapping[str, str] | None = None,
) -> dict[str, object]:
    series_payload = {}
    for series_id, stats in result.series.items():
        series_payload[series_id] = {
            "n_obs": stats.n_obs,
            "mean_ci": ([stats.mean_ci.lower, stats.mean_ci.upper] if stats.mean_ci else None),
            "volatility_ci": (
                [stats.volatility_ci.lower, stats.volatility_ci.upper]
                if stats.volatility_ci
                else None
            ),
        }

    corr_payload = []
    for corr in result.correlations:
        corr_payload.append(
            {
                "pair": list(corr.pair),
                "n_obs": corr.n_obs,
                "rho_ci": [corr.ci.lower, corr.ci.upper] if corr.ci else None,
            }
        )

    report = {
        "settings": {
            "mean_shrinkage": result.mean_shrinkage,
            "corr_shrinkage": result.corr_shrinkage,
            "corr_target": result.corr_target,
            "confidence_level": result.confidence_level,
            "annualize": result.annualize,
        },
        "series": series_payload,
        "correlations": corr_payload,
    }
    if alpha_map is not None:
        report["alpha_map"] = dict(alpha_map)
    return report


def infer_series_ids(
    columns: Iterable[str], *, index_id: str | None, alpha_ids: Sequence[str] | None
) -> tuple[str, tuple[str, str, str]]:
    col_list = list(columns)
    if index_id is None:
        if not col_list:
            raise ValueError("returns data has no columns")
        index_id = col_list[0]
    if alpha_ids is None:
        candidates = [c for c in col_list if c != index_id]
        if len(candidates) != 3:
            raise ValueError("expected exactly three alpha series; use --alpha-ids")
        alpha_ids = candidates
    if len(alpha_ids) != 3:
        raise ValueError("alpha_ids must contain exactly three series")
    return index_id, (alpha_ids[0], alpha_ids[1], alpha_ids[2])
