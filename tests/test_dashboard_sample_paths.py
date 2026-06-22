"""No-upload dashboard sample paths for Asset Library and Portfolio Builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from dashboard.utils import (
    SAMPLE_ASSET_TIMESERIES_FILENAME,
    SAMPLE_PORTFOLIO_TEMPLATE_FILENAME,
    bundled_asset_timeseries_path,
    bundled_portfolio_template_path,
    load_bundled_asset_returns,
)
from pa_core.schema import load_scenario
from tests.test_portfolio_builder_page import FakeStreamlit, _Asset, _Scenario, _load_module


def test_asset_library_bundled_sample_returns_are_usable() -> None:
    path = bundled_asset_timeseries_path()
    parsed = load_bundled_asset_returns()

    assert path.exists()
    assert path.name == SAMPLE_ASSET_TIMESERIES_FILENAME
    assert isinstance(parsed, pd.DataFrame)
    assert {"date", "id", "return"}.issubset(parsed.columns)
    assert parsed["id"].nunique() > 1
    assert not parsed["return"].isna().any()


def test_portfolio_builder_bundled_sample_template_loads_assets() -> None:
    path = bundled_portfolio_template_path()
    scenario = load_scenario(path)

    assert path.exists()
    assert path.name == SAMPLE_PORTFOLIO_TEMPLATE_FILENAME
    assert scenario.assets
    assert scenario.portfolios


def test_dashboard_pages_expose_no_upload_sample_affordances() -> None:
    asset_source = Path("dashboard/pages/1_Asset_Library.py").read_text()
    portfolio_source = Path("dashboard/pages/2_Portfolio_Builder.py").read_text()

    assert "Use bundled sample asset data (no upload needed)" in asset_source
    assert "Download asset return template" in asset_source
    assert "Load bundled sample portfolio" in portfolio_source
    assert "Download starter portfolio YAML" in portfolio_source


def test_portfolio_builder_loads_bundled_sample_without_upload(monkeypatch) -> None:
    fake_st = FakeStreamlit("streamlit", number_inputs=[0.5, 0.5, 1.0, 0.0])
    fake_st._checkbox_values = [True]
    module = _load_module(monkeypatch, fake_st)

    loaded_paths: list[Any] = []

    def fake_load_scenario(path: Any) -> _Scenario:
        loaded_paths.append(path)
        return _Scenario([_Asset("A"), _Asset("B")], correlations="corr")

    module["main"].__globals__["load_scenario"] = fake_load_scenario
    module["main"]()

    assert loaded_paths == [bundled_portfolio_template_path()]
    assert ("checkbox", "Load bundled sample portfolio") in fake_st.calls
    assert not any(call[0] == "info" for call in fake_st.calls)
