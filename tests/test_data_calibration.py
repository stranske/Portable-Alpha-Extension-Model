from __future__ import annotations

# ruff: noqa: E402

from pathlib import Path

import pytest

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
