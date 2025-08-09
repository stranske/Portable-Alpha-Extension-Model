from __future__ import annotations

# ruff: noqa: E402

from pathlib import Path

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
    importer = DataImportAgent(date_col="Date")
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

    df_wide = DataImportAgent(date_col="Date").load(wide)
    df_long = DataImportAgent(
        date_col="Date", id_col="Id", value_col="Return", wide=False
    ).load(long)

    assert_frame_equal(df_wide, df_long)


def test_calibration_to_yaml(tmp_path: Path) -> None:
    path = Path("templates/asset_timeseries_wide_returns.csv")
    importer = DataImportAgent(date_col="Date")
    df = importer.load(path)
    calib = CalibrationAgent(min_obs=1)
    result = calib.calibrate(df, index_id="SP500_TR")

    out = tmp_path / "library.yaml"
    calib.to_yaml(result, out)
    data = yaml.safe_load(out.read_text())

    assert data["index"]["id"] == "SP500_TR"
    assert any(a["id"] == "FUND_A" for a in data["assets"])
    assert any({"SP500_TR", "FUND_B"} == set(c["pair"]) for c in data["correlations"])
