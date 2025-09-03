from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

PKG = types.ModuleType("pa_core")
PKG.__path__ = [str(Path("pa_core"))]
sys.modules.setdefault("pa_core", PKG)

from pa_core.data import DataImportAgent


def test_template_roundtrip_csv(tmp_path: Path) -> None:
    csv_path = Path("templates/asset_timeseries_wide_returns.csv")
    importer = DataImportAgent(date_col="Date", min_obs=1)
    df = importer.load(csv_path)

    tmpl = tmp_path / "mapping.yaml"
    importer.save_template(tmpl)

    cloned = DataImportAgent.from_template(tmpl)
    df2 = cloned.load(csv_path)

    assert_frame_equal(df, df2)


def test_template_roundtrip_excel(tmp_path: Path) -> None:
    csv_path = Path("templates/asset_timeseries_wide_returns.csv")
    df_csv = pd.read_csv(csv_path)
    xlsx_path = tmp_path / "data.xlsx"
    df_csv.to_excel(xlsx_path, index=False)

    importer = DataImportAgent(date_col="Date", min_obs=1)
    df = importer.load(xlsx_path)

    tmpl = tmp_path / "mapping.yaml"
    importer.save_template(tmpl)

    cloned = DataImportAgent.from_template(tmpl)
    df2 = cloned.load(xlsx_path)

    assert_frame_equal(df, df2)
