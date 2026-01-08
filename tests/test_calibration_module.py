from __future__ import annotations

from itertools import combinations
from typing import Mapping

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from pa_core.calibration import (
    CalibrationResult,
    CorrelationEstimate,
    SeriesEstimate,
    _boost_shrinkage_for_short_samples,
    _c4,
    calibrate_returns,
)
from pa_core.schema import Scenario


def test_mean_shrinkage_moves_toward_grand_mean() -> None:
    rng = np.random.default_rng(7)
    data = rng.normal(loc=0.01, scale=0.02, size=(12, 4))
    df = pd.DataFrame(data, columns=["IDX", "H", "E", "M"])

    result = calibrate_returns(
        df,
        index_id="IDX",
        mean_shrinkage=0.5,
        corr_shrinkage=0.0,
        annualize=False,
    )

    sample_means = df.mean()
    grand_mean = float(sample_means.mean())
    for series_id, sample_mean in sample_means.items():
        shrunk_mean = result.series[series_id].mean
        assert abs(shrunk_mean - grand_mean) <= abs(sample_mean - grand_mean) + 1e-12


def test_volatility_small_sample_correction() -> None:
    df = pd.DataFrame(
        {
            "IDX": [0.01, 0.02, 0.03, 0.05],
            "H": [0.03, 0.02, 0.01, 0.00],
            "E": [0.02, 0.02, 0.02, 0.02],
            "M": [0.01, 0.01, 0.02, 0.03],
        }
    )
    n_obs = len(df)
    sample_std = float(df["IDX"].std(ddof=1))
    expected = sample_std / _c4(n_obs)

    result = calibrate_returns(
        df,
        index_id="IDX",
        mean_shrinkage=0.0,
        corr_shrinkage=0.0,
        annualize=False,
    )
    assert result.series["IDX"].volatility == pytest.approx(expected)


def test_correlation_shrinkage_matches_expected() -> None:
    df = pd.DataFrame(
        {
            "IDX": [1.0, 2.0, 3.0, 4.0],
            "H": [1.0, 2.0, 3.0, 4.0],
            "E": [1.0, 2.0, 3.0, 4.0],
            "M": [2.0, 2.0, 2.0, 2.0],
        }
    )
    shrinkage = 0.4
    expected_shrinkage = _boost_shrinkage_for_short_samples(
        shrinkage, n_samples=len(df), n_features=df.shape[1]
    )

    result = calibrate_returns(
        df,
        index_id="IDX",
        mean_shrinkage=0.0,
        corr_shrinkage=shrinkage,
        corr_target="identity",
        annualize=False,
    )
    pair = next(c for c in result.correlations if set(c.pair) == {"H", "E"})
    expected_rho = (1.0 - expected_shrinkage) * 1.0 + expected_shrinkage * 0.0
    assert pair.rho == pytest.approx(expected_rho)


def test_correlation_shrinkage_boosts_with_fewer_pair_obs() -> None:
    df = pd.DataFrame(
        {
            "IDX": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "H": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "E": [1.0, 2.0, np.nan, np.nan, np.nan, np.nan],
            "M": [1.0, 2.0, np.nan, np.nan, np.nan, np.nan],
        }
    )
    result = calibrate_returns(
        df,
        index_id="IDX",
        mean_shrinkage=0.0,
        corr_shrinkage=0.2,
        corr_target="identity",
        annualize=False,
    )

    pair_full = next(c for c in result.correlations if set(c.pair) == {"IDX", "H"})
    pair_sparse = next(c for c in result.correlations if set(c.pair) == {"E", "M"})
    assert abs(pair_sparse.rho) < abs(pair_full.rho)


def test_confidence_intervals_and_missing_data() -> None:
    df = pd.DataFrame(
        {
            "IDX": [0.01, 0.02, 0.03, 0.04, 0.05],
            "H": [0.02, 0.01, np.nan, 0.03, 0.04],
            "E": [0.00, 0.01, 0.02, 0.02, 0.02],
            "M": [0.03, 0.03, 0.03, np.nan, 0.03],
        }
    )
    result = calibrate_returns(
        df,
        index_id="IDX",
        mean_shrinkage=0.1,
        corr_shrinkage=0.2,
        annualize=False,
    )

    idx_stats = result.series["IDX"]
    assert idx_stats.mean_ci is not None
    assert idx_stats.volatility_ci is not None

    pair = next(c for c in result.correlations if set(c.pair) == {"IDX", "H"})
    assert pair.n_obs == 4
    assert pair.ci is not None


def test_missing_series_mean_uses_grand_mean() -> None:
    df = pd.DataFrame(
        {
            "IDX": [0.01, 0.02, 0.03, 0.04],
            "H": [np.nan, np.nan, np.nan, np.nan],
            "E": [0.00, 0.01, 0.01, 0.02],
            "M": [0.03, 0.03, 0.02, 0.01],
        }
    )
    result = calibrate_returns(
        df,
        index_id="IDX",
        mean_shrinkage=0.3,
        corr_shrinkage=0.0,
        annualize=False,
    )
    sample_means = df.mean()
    grand_mean = float(sample_means.dropna().mean())
    assert result.series["H"].mean == pytest.approx(grand_mean)


def test_model_config_and_scenario_conversion() -> None:
    df = pd.DataFrame(
        {
            "IDX": [0.01, 0.02, 0.03, 0.04],
            "H": [0.02, 0.01, 0.03, 0.01],
            "E": [0.01, 0.01, 0.02, 0.02],
            "M": [0.03, 0.02, 0.01, 0.00],
        }
    )
    result = calibrate_returns(df, index_id="IDX", annualize=False)
    alpha_map: Mapping[str, str] = {"H": "H", "E": "E", "M": "M"}
    params = result.to_model_config(alpha_map=alpha_map)

    for key in ("mu_H", "mu_E", "mu_M", "sigma_H", "sigma_E", "sigma_M"):
        assert key in params
    scenario = result.to_scenario()
    assert scenario.index.id == "IDX"
    assert {asset.id for asset in scenario.assets} == {"H", "E", "M"}


def test_to_scenario_excludes_index_from_assets() -> None:
    series = {
        "IDX": SeriesEstimate(
            mean=0.01,
            volatility=0.02,
            mean_ci=None,
            volatility_ci=None,
            n_obs=12,
        ),
        "A": SeriesEstimate(
            mean=0.02,
            volatility=0.03,
            mean_ci=None,
            volatility_ci=None,
            n_obs=12,
        ),
    }
    correlations = [
        CorrelationEstimate(pair=("IDX", "A"), rho=0.5, ci=None, n_obs=12),
    ]
    result = CalibrationResult(
        index_id="IDX",
        series=series,
        correlations=correlations,
        corr_target="identity",
        mean_shrinkage=0.0,
        corr_shrinkage=0.0,
        confidence_level=0.95,
        annualize=False,
    )

    scenario = result.to_scenario()
    assert scenario.index.id == "IDX"
    assert [asset.id for asset in scenario.assets] == ["A"]


def test_to_scenario_index_only_assets_empty() -> None:
    series = {
        "IDX": SeriesEstimate(
            mean=0.01,
            volatility=0.02,
            mean_ci=None,
            volatility_ci=None,
            n_obs=12,
        ),
    }
    result = CalibrationResult(
        index_id="IDX",
        series=series,
        correlations=[],
        corr_target="identity",
        mean_shrinkage=0.0,
        corr_shrinkage=0.0,
        confidence_level=0.95,
        annualize=False,
    )

    scenario = result.to_scenario()
    assert scenario.index.id == "IDX"
    assert scenario.assets == []


def test_scenario_validation_rejects_index_in_assets() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Scenario.model_validate(
            {
                "index": {"id": "IDX", "mu": 0.05, "sigma": 0.1},
                "assets": [{"id": "IDX", "mu": 0.04, "sigma": 0.08}],
                "correlations": [],
            }
        )
    errors = excinfo.value.errors()
    assert errors[0]["msg"] == "Value error, Assets must not include index.id"
    assert errors[0]["type"] == "value_error"


def test_to_scenario_raises_when_index_missing() -> None:
    series = {
        "A": SeriesEstimate(
            mean=0.02,
            volatility=0.03,
            mean_ci=None,
            volatility_ci=None,
            n_obs=12,
        ),
    }
    result = CalibrationResult(
        index_id="IDX",
        series=series,
        correlations=[],
        corr_target="identity",
        mean_shrinkage=0.0,
        corr_shrinkage=0.0,
        confidence_level=0.95,
        annualize=False,
    )

    with pytest.raises(ValueError, match="index_id not found in series estimates"):
        result.to_scenario()


@pytest.mark.parametrize(
    ("ordered_ids", "expected_assets"),
    [
        (["IDX", "A"], ["A"]),
        (["A", "IDX"], ["A"]),
        (["A", "IDX", "B"], ["A", "B"]),
        (["B", "A", "IDX", "C"], ["B", "A", "C"]),
    ],
)
def test_to_scenario_excludes_index_across_orderings(
    ordered_ids: list[str], expected_assets: list[str]
) -> None:
    series: dict[str, SeriesEstimate] = {}
    for offset, series_id in enumerate(ordered_ids):
        series[series_id] = SeriesEstimate(
            mean=0.01 + 0.01 * offset,
            volatility=0.02 + 0.01 * offset,
            mean_ci=None,
            volatility_ci=None,
            n_obs=12,
        )
    correlations = [
        CorrelationEstimate(pair=pair, rho=0.1, ci=None, n_obs=12)
        for pair in combinations(ordered_ids, 2)
    ]
    result = CalibrationResult(
        index_id="IDX",
        series=series,
        correlations=correlations,
        corr_target="identity",
        mean_shrinkage=0.0,
        corr_shrinkage=0.0,
        confidence_level=0.95,
        annualize=False,
    )

    scenario = result.to_scenario()
    assert [asset.id for asset in scenario.assets] == expected_assets
