import io

import openpyxl
import pandas as pd
import pytest

from pa_core.reporting.stress_delta import (
    build_delta_table,
    build_stress_workbook,
    format_delta_table_text,
)


def test_build_delta_table_includes_total_row():
    base = pd.DataFrame(
        {
            "Agent": ["Base", "ExternalPA"],
            "AnnReturn": [0.05, 0.07],
            "BreachCount": [1, 2],
        }
    )
    stressed = pd.DataFrame(
        {
            "Agent": ["Base", "ExternalPA"],
            "AnnReturn": [0.02, 0.06],
            "BreachCount": [3, 5],
        }
    )

    delta_df = build_delta_table(base, stressed)

    assert "Total" in delta_df["Agent"].tolist()
    assert "Base" not in delta_df["Agent"].tolist()
    assert delta_df["Agent"].iloc[-1] == "Total"

    total_row = delta_df[delta_df["Agent"] == "Total"].iloc[0]
    assert total_row["AnnReturn"] == pytest.approx(-0.03)
    assert total_row["BreachCount"] == pytest.approx(2)


def test_build_stress_workbook_contains_expected_sheets():
    base = pd.DataFrame({"Agent": ["Base"], "AnnReturn": [0.05], "BreachCount": [1]})
    stressed = pd.DataFrame(
        {"Agent": ["Base"], "AnnReturn": [0.02], "BreachCount": [3]}
    )
    delta_df = build_delta_table(base, stressed)
    config_diff = pd.DataFrame({"Parameter": ["mu_H"], "Base": [0.01], "Stressed": [0]})

    data = build_stress_workbook(base, stressed, delta_df, config_diff)
    wb = openpyxl.load_workbook(io.BytesIO(data))

    assert "BaseSummary" in wb.sheetnames
    assert "StressedSummary" in wb.sheetnames
    assert "Delta" in wb.sheetnames
    assert "ConfigDiff" in wb.sheetnames


def test_format_delta_table_text_adds_signs():
    delta_df = pd.DataFrame(
        {"Agent": ["Total"], "AnnReturn": [-0.03], "AnnVol": [0.01], "BreachCount": [2]}
    )
    formatted = format_delta_table_text(delta_df)

    assert formatted.loc[0, "AnnReturn"] == "-3.00%"
    assert formatted.loc[0, "AnnVol"] == "+1.00%"
    assert formatted.loc[0, "BreachCount"] == "+2.0000"
