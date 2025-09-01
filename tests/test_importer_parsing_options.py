from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pa_core.data import DataImportAgent


def test_csv_decimal_and_thousands_prices(tmp_path: Path) -> None:
    # Wide-format prices using European number formatting: thousands='.' and decimal=','
    csv_path = tmp_path / "prices_eu.csv"
    data = {
        "Date": ["2024-01-31", "2024-02-29", "2024-03-31"],
        "A": ["1.000,00", "1.010,00", "1.020,10"],  # 0% -> ~1% -> ~1% month-over-month
        "B": ["2.000,00", "1.980,00", "2.019,60"],  # -1% -> +2% approx
    }
    pd.DataFrame(data).to_csv(csv_path, index=False)

    importer = DataImportAgent(
        date_col="Date",
        value_type="prices",
        frequency="monthly",
        wide=True,
        min_obs=1,
        decimal=",",
        thousands=".",
    )
    df = importer.load(csv_path)

    # Expect monthly returns for each id and strictly increasing dates
    assert set(df.columns) == {"id", "date", "return"}
    assert set(df["id"]) == {"A", "B"}
    # We should have 2 return rows per asset (n-1 for monthly prices)
    counts = df.groupby("id").size()
    assert counts["A"] == 2 and counts["B"] == 2
    assert df.groupby("id")["date"].is_monotonic_increasing.all()


def test_excel_sheet_selection_with_template(tmp_path: Path) -> None:
    # Create an Excel file with two sheets; data should be on 'Data'
    xlsx_path = tmp_path / "data.xlsx"
    df_data = pd.DataFrame(
        {
            "Date": ["2024-01-31", "2024-02-29"],
            "Asset1": [0.01, 0.02],
        }
    )
    df_other = pd.DataFrame({"junk": [1, 2]})
    with pd.ExcelWriter(xlsx_path) as writer:
        df_other.to_excel(writer, sheet_name="Other", index=False)
        df_data.to_excel(writer, sheet_name="Data", index=False)

    importer = DataImportAgent(date_col="Date", wide=True, min_obs=1, sheet_name="Data")
    df = importer.load(xlsx_path)
    assert set(df.columns) == {"id", "date", "return"}
    assert set(df["id"]) == {"Asset1"}

    # Roundtrip via template should preserve sheet_name
    tmpl = tmp_path / "mapping.yaml"
    importer.save_template(tmpl)
    cloned = DataImportAgent.from_template(tmpl)
    df2 = cloned.load(xlsx_path)
    pd.testing.assert_frame_equal(df.reset_index(drop=True), df2.reset_index(drop=True))


def test_custom_na_values_dropped(tmp_path: Path) -> None:
    # Long-format returns with a custom NA marker
    csv_path = tmp_path / "returns_with_na.csv"
    data = pd.DataFrame(
        {
            "Date": ["2024-01-31", "2024-02-29", "2024-03-31"],
            "Id": ["X", "X", "X"],
            "Return": ["NA", "0.01", "NA"],
        }
    )
    data.to_csv(csv_path, index=False)

    importer = DataImportAgent(
        date_col="Date", id_col="Id", value_col="Return", wide=False, min_obs=1, na_values=["NA"]
    )
    df = importer.load(csv_path)
    # Only one valid row remains
    assert len(df) == 1
    assert df["return"].iloc[0] == pytest.approx(0.01)


def test_no_valid_data_after_parsing_raises(tmp_path: Path) -> None:
    # All rows are NA after applying na_values
    csv_path = tmp_path / "all_na.csv"
    data = pd.DataFrame(
        {
            "Date": ["2024-01-31", "2024-02-29"],
            "Id": ["X", "X"],
            "Return": ["MISSING", "NA"],
        }
    )
    data.to_csv(csv_path, index=False)

    importer = DataImportAgent(
        date_col="Date",
        id_col="Id",
        value_col="Return",
        wide=False,
        min_obs=1,
        na_values=["MISSING", "NA"],
    )
    with pytest.raises(ValueError, match="no valid data after parsing"):
        _ = importer.load(csv_path)
