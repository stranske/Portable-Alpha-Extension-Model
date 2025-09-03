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


def test_templates_across_formats_produce_identical_results(tmp_path: Path) -> None:
    wide_path = Path("templates/asset_timeseries_wide_returns.csv")
    long_path = Path("templates/asset_timeseries_long_returns.csv")

    wide_importer = DataImportAgent(date_col="Date", min_obs=1)
    _ = wide_importer.load(wide_path)
    wide_tmpl = tmp_path / "wide_mapping.yaml"
    wide_importer.save_template(wide_tmpl)

    long_importer = DataImportAgent(
        date_col="Date",
        id_col="Id",
        value_col="Return",
        wide=False,
        min_obs=1,
    )
    _ = long_importer.load(long_path)
    long_tmpl = tmp_path / "long_mapping.yaml"
    long_importer.save_template(long_tmpl)

    wide_clone = DataImportAgent.from_template(wide_tmpl)
    long_clone = DataImportAgent.from_template(long_tmpl)

    df_wide = wide_clone.load(wide_path)
    df_long = long_clone.load(long_path)

    assert_frame_equal(
        df_wide.sort_values(["id", "date"]).reset_index(drop=True),
        df_long.sort_values(["id", "date"]).reset_index(drop=True),
    )
