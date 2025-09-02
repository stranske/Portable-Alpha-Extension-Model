from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pandas.testing import assert_frame_equal

from pa_core.data import CalibrationAgent, DataImportAgent

# ruff: noqa: E402


def test_calibration_wide_csv() -> None:
    path = Path("templates/asset_timeseries_wide_returns.csv")
    importer = DataImportAgent(date_col="Date", min_obs=1)
    df = importer.load(path)
    calib = CalibrationAgent(min_obs=1)
    result = calib.calibrate(df, index_id="SP500_TR")

    pivot = df.pivot(index="date", columns="id", values="return")
    expected_mu = pivot.mean() * 12.0
    expected_sigma = pivot.std(ddof=1) * (12.0**0.5)

    assert result.index.mu == pytest.approx(expected_mu["SP500_TR"])
    assert result.index.sigma == pytest.approx(expected_sigma["SP500_TR"])

    asset_a = next(a for a in result.assets if a.id == "FUND_A")
    assert asset_a.mu == pytest.approx(expected_mu["FUND_A"])
    assert asset_a.sigma == pytest.approx(expected_sigma["FUND_A"])

    corr = pivot.corr()
    corr_ab = corr.loc["FUND_A", "FUND_B"]
    pair_ab = next(
        c for c in result.correlations if set(c.pair) == {"FUND_A", "FUND_B"}
    )
    assert pair_ab.rho == pytest.approx(corr_ab)


def test_import_long_equals_wide() -> None:
    wide = Path("templates/asset_timeseries_wide_returns.csv")
    long = Path("templates/asset_timeseries_long_returns.csv")

    df_wide = DataImportAgent(date_col="Date", min_obs=1).load(wide)
    df_long = DataImportAgent(
        date_col="Date", id_col="Id", value_col="Return", wide=False, min_obs=1
    ).load(long)

    assert_frame_equal(df_wide, df_long)


def test_calibration_to_yaml(tmp_path: Path) -> None:
    path = Path("templates/asset_timeseries_wide_returns.csv")
    importer = DataImportAgent(date_col="Date", min_obs=1)
    df = importer.load(path)
    calib = CalibrationAgent(min_obs=1)
    result = calib.calibrate(df, index_id="SP500_TR")

    out = tmp_path / "library.yaml"
    calib.to_yaml(result, out)
    data = yaml.safe_load(out.read_text())

    assert data["index"]["id"] == "SP500_TR"
    assert any(a["id"] == "FUND_A" for a in data["assets"])
    assert any({"SP500_TR", "FUND_B"} == set(c["pair"]) for c in data["correlations"])


def test_import_daily_prices_to_monthly_returns(tmp_path: Path) -> None:
    dates = pd.date_range("2020-01-01", "2020-02-29", freq="D")
    prices = (1.01) ** np.arange(len(dates))
    df = pd.DataFrame({"Date": dates, "A": prices})

    csv_path = tmp_path / "prices.csv"
    xlsx_path = tmp_path / "prices.xlsx"
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    importer_csv = DataImportAgent(
        date_col="Date", frequency="daily", value_type="prices", min_obs=1
    )
    importer_xlsx = DataImportAgent(
        date_col="Date", frequency="daily", value_type="prices", min_obs=1
    )

    df_csv = importer_csv.load(csv_path)
    df_xlsx = importer_xlsx.load(xlsx_path)

    assert_frame_equal(df_csv, df_xlsx)

    expected = pd.DataFrame(
        {
            "id": ["A", "A"],
            "date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "return": [(1.01**31) - 1, (1.01**29) - 1],
        }
    )
    assert_frame_equal(df_csv.reset_index(drop=True), expected)

    assert importer_csv.metadata["frequency"] == "daily"
    assert importer_csv.metadata["value_type"] == "prices"


def test_import_daily_returns_to_monthly_returns(tmp_path: Path) -> None:
    dates = pd.date_range("2020-01-01", "2020-02-29", freq="D")
    DAILY_RETURN = 0.001  # Fixed daily return of 0.1%
    returns = pd.Series(DAILY_RETURN, index=dates)
    df = pd.DataFrame({"Date": dates, "A": returns.values})

    path = tmp_path / "returns.csv"
    df.to_csv(path, index=False)

    importer = DataImportAgent(
        date_col="Date", frequency="daily", value_type="returns", min_obs=1
    )
    out = importer.load(path)

    # Calculate expected monthly returns by compounding daily returns within each month
    jan_days = (dates.month == 1).sum()
    feb_days = (dates.month == 2).sum()
    expected = pd.DataFrame(
        {
            "id": ["A", "A"],
            "date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "return": [
                ((1 + DAILY_RETURN) ** jan_days) - 1,
                ((1 + DAILY_RETURN) ** feb_days) - 1,
            ],
        }
    )
    assert_frame_equal(out.reset_index(drop=True), expected)


def test_import_min_obs_enforced(tmp_path: Path) -> None:
    dates = pd.date_range("2020-01-31", periods=10, freq="ME")
    df = pd.DataFrame({"Date": dates, "A": np.arange(10)})
    path = tmp_path / "short.csv"
    df.to_csv(path, index=False)
    agent = DataImportAgent(date_col="Date", min_obs=12)
    with pytest.raises(ValueError, match="insufficient data"):
        agent.load(path)


@pytest.mark.parametrize(
    "periods,min_obs,should_fail",
    [
        (10, 12, True),  # Insufficient data - should fail
        (15, 12, False),  # Sufficient data - should pass
        (12, 12, False),  # Exactly enough data - should pass
        (5, 3, False),  # More than enough - should pass
    ],
)
def test_import_min_obs_validation_scenarios(
    tmp_path: Path, periods, min_obs, should_fail
) -> None:
    """Test minimum observation validation with different data sizes and requirements."""
    dates = pd.date_range("2020-01-31", periods=periods, freq="ME")
    df = pd.DataFrame({"Date": dates, "A": np.arange(periods)})
    path = tmp_path / f"data_{periods}_{min_obs}.csv"
    df.to_csv(path, index=False)

    agent = DataImportAgent(date_col="Date", min_obs=min_obs)

    if should_fail:
        with pytest.raises(ValueError, match="insufficient data"):
            agent.load(path)
    else:
        result = agent.load(path)  # Should not raise exception
        assert len(result) == periods


def test_import_duplicate_dates_fail(tmp_path: Path) -> None:
    df = pd.DataFrame({"Date": ["2020-01-31", "2020-01-31"], "A": [0.1, 0.2]})
    path = tmp_path / "dup.csv"
    df.to_csv(path, index=False)
    agent = DataImportAgent(date_col="Date", min_obs=1)
    with pytest.raises(ValueError, match="strictly increasing"):
        agent.load(path)


@pytest.mark.parametrize(
    "date_sequence,should_fail",
    [
        (["2020-01-31", "2020-01-31"], True),  # Exact duplicate - should fail
        (["2020-01-31", "2020-02-29"], False),  # Valid sequence - should pass
        (
            ["2020-01-01", "2020-02-01", "2020-03-01"],
            False,
        ),  # Valid sequence - should pass
    ],
)
def test_import_date_validation_scenarios(
    tmp_path: Path, date_sequence, should_fail
) -> None:
    """Test date validation with various date sequence scenarios."""
    df = pd.DataFrame({"Date": date_sequence, "A": [0.1] * len(date_sequence)})
    path = tmp_path / f"dates_{len(date_sequence)}.csv"
    df.to_csv(path, index=False)

    agent = DataImportAgent(date_col="Date", min_obs=1)

    if should_fail:
        with pytest.raises(ValueError, match="strictly increasing"):
            agent.load(path)
    else:
        result = agent.load(path)  # Should not raise exception
        assert len(result) == len(date_sequence)


def test_daily_to_monthly_robust_frequency_handling(tmp_path: Path) -> None:
    """Test that daily-to-monthly conversion correctly handles months with different day counts.

    This test validates that the system doesn't rely on hardcoded day-count assumptions
    like 365 days per year, but properly compounds returns within actual calendar months.
    """
    # Include months with different day counts: Feb (28 days), Apr (30 days), May (31 days)
    dates = pd.date_range("2021-02-01", "2021-05-31", freq="D")
    DAILY_RETURN = 0.0005  # 0.05% daily return
    returns = pd.Series(DAILY_RETURN, index=dates)
    df = pd.DataFrame({"Date": dates, "A": returns.values})

    path = tmp_path / "variable_months.csv"
    df.to_csv(path, index=False)

    importer = DataImportAgent(
        date_col="Date", frequency="daily", value_type="returns", min_obs=1
    )
    result = importer.load(path)

    # Verify we get the correct number of monthly observations
    assert len(result) == 4  # Feb, Mar, Apr, May 2021

    # Calculate expected returns for each month based on actual day counts
    feb_days = (dates.month == 2).sum()  # 28 days in Feb 2021
    mar_days = (dates.month == 3).sum()  # 31 days in Mar 2021
    apr_days = (dates.month == 4).sum()  # 30 days in Apr 2021
    may_days = (dates.month == 5).sum()  # 31 days in May 2021

    expected_returns = [
        ((1 + DAILY_RETURN) ** feb_days) - 1,  # Feb: 28 days
        ((1 + DAILY_RETURN) ** mar_days) - 1,  # Mar: 31 days
        ((1 + DAILY_RETURN) ** apr_days) - 1,  # Apr: 30 days
        ((1 + DAILY_RETURN) ** may_days) - 1,  # May: 31 days
    ]

    expected_dates = pd.to_datetime(
        ["2021-02-28", "2021-03-31", "2021-04-30", "2021-05-31"]
    )

    for i, (expected_return, expected_date) in enumerate(
        zip(expected_returns, expected_dates)
    ):
        assert result.iloc[i]["return"] == pytest.approx(expected_return)
        assert result.iloc[i]["date"] == expected_date
        assert result.iloc[i]["id"] == "A"

    # Verify that Feb (28 days) has a different monthly return than Mar/May (31 days)
    feb_return = result.iloc[0]["return"]
    mar_return = result.iloc[1]["return"]
    may_return = result.iloc[3]["return"]

    assert (
        feb_return != mar_return
    )  # Different day counts should produce different returns
    assert mar_return == pytest.approx(
        may_return
    )  # Same day counts should produce same returns
