from __future__ import annotations

from typing import Any

import pandas as pd

from pa_core.reporting import console as reporting_console


class _CaptureConsole:
    def __init__(self) -> None:
        self.printed: list[Any] = []

    def print(self, obj: Any) -> None:
        self.printed.append(obj)


def test_print_summary_with_mapping(monkeypatch) -> None:
    capture = _CaptureConsole()
    monkeypatch.setattr(reporting_console, "Console", lambda: capture)

    reporting_console.print_summary({"AnnReturn": 0.1, "Label": "Base"})

    assert len(capture.printed) == 1
    table = capture.printed[0]
    headers = [col.header for col in table.columns]
    assert headers == ["AnnReturn", "Label"]
    assert table.columns[0]._cells == ["0.1000"]
    assert table.columns[1]._cells == ["Base"]


def test_print_summary_with_dataframe(monkeypatch) -> None:
    capture = _CaptureConsole()
    monkeypatch.setattr(reporting_console, "Console", lambda: capture)

    df = pd.DataFrame({"Metric": ["AnnReturn"], "Value": [0.2]})
    reporting_console.print_summary(df)

    assert len(capture.printed) == 1
    table = capture.printed[0]
    headers = [col.header for col in table.columns]
    assert headers == ["Metric", "Value"]
    assert table.columns[0]._cells == ["AnnReturn"]
    assert table.columns[1]._cells == ["0.2000"]


def test_print_run_diff_with_data(monkeypatch) -> None:
    capture = _CaptureConsole()
    monkeypatch.setattr(reporting_console, "Console", lambda: capture)

    cfg_df = pd.DataFrame(
        {
            "Parameter": ["N_SIMULATIONS"],
            "Current": [1000],
            "Previous": [500],
            "Delta": [500],
        }
    )
    metric_df = pd.DataFrame(
        {
            "Metric": ["AnnReturn"],
            "Agent": ["Base"],
            "Current": [0.1],
            "Previous": [0.08],
            "Delta": [0.02],
        }
    )

    reporting_console.print_run_diff(cfg_df, metric_df, max_rows=5)

    assert len(capture.printed) == 2
    titles = [table.title for table in capture.printed]
    assert "Config Changes vs Previous" in titles
    assert "Metric Changes vs Previous" in titles


def test_print_run_diff_no_changes(monkeypatch) -> None:
    capture = _CaptureConsole()
    monkeypatch.setattr(reporting_console, "Console", lambda: capture)

    reporting_console.print_run_diff(pd.DataFrame(), pd.DataFrame(), max_rows=5)

    assert capture.printed == ["No changes detected vs previous run."]
