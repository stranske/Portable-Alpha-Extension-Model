from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd


class _FakeStreamlit(ModuleType):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.session_state: dict = {}
        self.errors: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)


def _load_module(monkeypatch) -> tuple[dict, _FakeStreamlit]:
    fake_st = _FakeStreamlit("streamlit")
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    module = runpy.run_path(str(Path("dashboard/pages/5_Scenario_Grid.py")))
    return module, fake_st


def test_get_empty_dataframe_cached(monkeypatch) -> None:
    module, _ = _load_module(monkeypatch)
    first = module["_get_empty_dataframe"]()
    second = module["_get_empty_dataframe"]()

    assert first is second
    assert isinstance(first, pd.DataFrame)


def test_grid_cache_round_trip_and_eviction(monkeypatch) -> None:
    module, fake_st = _load_module(monkeypatch)
    fake_st.session_state.clear()
    df = pd.DataFrame({"x": [1.0], "y": [2.0]})

    module["_set_grid_cache"]("alpha", df, "y_axis", 100.0)
    cached = module["_get_grid_cache"]("alpha")
    assert cached["grid_df"].equals(df)
    assert cached["y_col"] == "y_axis"
    assert cached["total_fund"] == 100.0

    module["_set_grid_cache"]("beta", df, "y_axis", 100.0)
    module["_set_grid_cache"]("gamma", df, "y_axis", 100.0)
    module["_set_grid_cache"]("delta", df, "y_axis", 100.0)

    cache = fake_st.session_state[module["_GRID_CACHE_KEY"]]
    assert "alpha" not in cache["entries"]
    assert cache["order"] == ["beta", "gamma", "delta"]


def test_extract_plotly_click(monkeypatch) -> None:
    module, _ = _load_module(monkeypatch)
    selection = {"selection": {"points": [{"x": "1.5", "y": 2}]}}
    assert module["_extract_plotly_click"](selection) == (1.5, 2.0)

    assert module["_extract_plotly_click"]({}) is None
    assert module["_extract_plotly_click"]({"selection": {"points": []}}) is None


def test_read_csv_from_bytes(monkeypatch) -> None:
    module, _ = _load_module(monkeypatch)

    class _Upload:
        def getvalue(self) -> bytes:
            return b"col\n1\n2\n"

    df = module["_read_csv"](_Upload())
    assert df["col"].tolist() == [1, 2]
