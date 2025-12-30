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
