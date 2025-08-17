from __future__ import annotations

# ruff: noqa: E402

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pandas.testing import assert_frame_equal

import types
import sys

PKG = types.ModuleType("pa_core")
PKG.__path__ = [str(Path("pa_core"))]
sys.modules.setdefault("pa_core", PKG)

from pa_core.data import CalibrationAgent, DataImportAgent


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
            "return": [(1.01 ** 31) - 1, (1.01 ** 29) - 1],
        }
    )
    assert_frame_equal(df_csv.reset_index(drop=True), expected)

    assert importer_csv.metadata["frequency"] == "daily"
    assert importer_csv.metadata["value_type"] == "prices"


def test_import_daily_returns_to_monthly_returns(tmp_path: Path) -> None:
    dates = pd.date_range("2020-01-01", "2020-02-29", freq="D")
    returns = pd.Series(0.01, index=dates)
    df = pd.DataFrame({"Date": dates, "A": returns.values})

    path = tmp_path / "returns.csv"
    df.to_csv(path, index=False)

    importer = DataImportAgent(
        date_col="Date", frequency="daily", value_type="returns", min_obs=1
    )
    out = importer.load(path)

    expected = pd.DataFrame(
        {
            "id": ["A", "A"],
            "date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "return": [(1.01 ** 31) - 1, (1.01 ** 29) - 1],
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


def test_import_duplicate_dates_fail(tmp_path: Path) -> None:
    df = pd.DataFrame({"Date": ["2020-01-31", "2020-01-31"], "A": [0.1, 0.2]})
    path = tmp_path / "dup.csv"
    df.to_csv(path, index=False)
    agent = DataImportAgent(date_col="Date", min_obs=1)
    with pytest.raises(ValueError, match="strictly increasing"):
        agent.load(path)
