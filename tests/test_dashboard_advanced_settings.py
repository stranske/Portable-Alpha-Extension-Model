"""Tests for the sidebar 'Advanced / Settings' expander (issue #1902).

The theme-file and results-file path inputs used to sit as bare
``st.sidebar.text_input`` widgets at the top of every page. They are now
tucked inside a collapsed ``"Advanced / Settings"`` sidebar expander, and the
home page sets a real page title instead of the Streamlit default ("app").
"""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import pytest
import streamlit as st

import dashboard.app as app
from pa_core.config import load_config
from pa_core.validators import calculate_margin_requirement
from pa_core.wizard_schema import AnalysisMode, get_default_config


class _FakeExpander:
    def __init__(self, st: "FakeStreamlit", label: str) -> None:
        self._st = st
        self._label = label

    def __enter__(self) -> "_FakeExpander":
        self._st.container_stack.append(f"expander:{self._label}")
        return self

    def __exit__(self, *exc: Any) -> bool:
        self._st.container_stack.pop()
        return False


class _FakeSidebar:
    def __init__(self, st: "FakeStreamlit") -> None:
        self._st = st

    def expander(self, label: str, expanded: bool = False) -> _FakeExpander:
        self._st.events.append(("sidebar.expander", label, expanded))
        return _FakeExpander(self._st, label)

    def text_input(self, label: str, value: str = "") -> str:
        # A bare sidebar text input -- the regression we are guarding against.
        self._st.events.append(("sidebar.text_input", label, tuple(self._st.container_stack)))
        return value

    def markdown(self, *args: Any, **kwargs: Any) -> None:
        pass

    def subheader(self, *args: Any, **kwargs: Any) -> None:
        pass


class FakeStreamlit:
    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []
        self.container_stack: list[str] = []
        self.session_state: dict[str, Any] = {}
        self.sidebar = _FakeSidebar(self)

    def text_input(self, label: str, value: str = "") -> str:
        self.events.append(("text_input", label, tuple(self.container_stack)))
        return value

    def set_page_config(self, **kwargs: Any) -> None:
        self.events.append(("set_page_config", kwargs))

    def expander(self, label: str, expanded: bool = False) -> _FakeExpander:
        # Top-level ``st.expander`` is a context manager (e.g. the getting-started
        # block in ``main`` via ``_render_getting_started``). Without this the
        # catch-all ``__getattr__`` below returns a no-op that yields ``None``,
        # which is not a context manager and breaks ``with st.expander(...)``.
        self.events.append(("expander", label, expanded))
        return _FakeExpander(self, label)

    def __getattr__(self, _name: str):  # no-op for title/write/page_link/etc.
        def _noop(*args: Any, **kwargs: Any) -> None:
            return None

        return _noop


@pytest.fixture()
def fake_st(monkeypatch: pytest.MonkeyPatch) -> FakeStreamlit:
    fake = FakeStreamlit()
    monkeypatch.setattr(app, "st", fake)
    return fake


def test_theme_input_lives_inside_advanced_expander(fake_st: FakeStreamlit) -> None:
    results, theme = app.render_settings_sidebar()

    assert results is None
    assert theme == app._DEF_THEME

    # The expander is rendered in the sidebar, collapsed by default.
    assert ("sidebar.expander", app.ADVANCED_SETTINGS_LABEL, False) in fake_st.events

    # The theme input is a text input nested inside that expander, not a bare
    # sidebar widget.
    theme_inputs = [e for e in fake_st.events if e[0] == "text_input" and e[1] == "Theme file"]
    assert len(theme_inputs) == 1
    assert theme_inputs[0][2] == (f"expander:{app.ADVANCED_SETTINGS_LABEL}",)
    assert not [e for e in fake_st.events if e[0] == "sidebar.text_input"]


def test_results_input_included_when_default_given(fake_st: FakeStreamlit) -> None:
    results, theme = app.render_settings_sidebar("Outputs.xlsx")

    assert results == "Outputs.xlsx"
    assert theme == app._DEF_THEME

    labels_in_expander = [
        e[1]
        for e in fake_st.events
        if e[0] == "text_input" and e[2] == (f"expander:{app.ADVANCED_SETTINGS_LABEL}",)
    ]
    assert labels_in_expander == ["Results file", "Theme file"]


def test_home_page_sets_real_title(fake_st: FakeStreamlit) -> None:
    app.main()

    page_configs = [e for e in fake_st.events if e[0] == "set_page_config"]
    assert len(page_configs) == 1
    assert page_configs[0][1].get("page_title") == "Portable Alpha Dashboard"


def test_wizard_default_allocation_is_feasible(tmp_path: Path) -> None:
    helpers = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
    default_allocation = helpers["_default_capital_allocation"]
    build_yaml = helpers["_build_yaml_from_config"]
    schedule_path = tmp_path / "margin_schedule.csv"
    schedule_path.write_text("term,multiplier\n1,10\n", encoding="utf-8")

    st.session_state.clear()
    try:
        config = get_default_config(AnalysisMode.RETURNS)
        st.session_state["financing_settings"] = {
            "financing_model": "schedule",
            "reference_sigma": 0.01,
            "volatility_multiple": 3.0,
            "schedule_path": str(schedule_path),
            "term_months": 1.0,
        }
        allocation = default_allocation(
            total_fund_capital=config.total_fund_capital,
            reference_sigma=0.01,
            volatility_multiple=3.0,
            financing_model="schedule",
            schedule_path=schedule_path,
            term_months=1.0,
        )
        config.external_pa_capital = allocation["external_pa_capital"]
        config.active_ext_capital = allocation["active_ext_capital"]
        config.internal_pa_capital = allocation["internal_pa_capital"]

        yaml_dict = build_yaml(config)
        model_config = load_config(yaml_dict)
        margin_requirement = calculate_margin_requirement(
            reference_sigma=model_config.reference_sigma,
            volatility_multiple=model_config.volatility_multiple,
            total_capital=model_config.total_fund_capital,
            financing_model=model_config.financing_model,
            schedule_path=model_config.financing_schedule_path,
            term_months=model_config.financing_term_months,
        )
        buffer_after_margin = (
            model_config.total_fund_capital - margin_requirement - model_config.internal_pa_capital
        )

        assert model_config.financing_model == "schedule"
        assert margin_requirement == pytest.approx(100.0)
        assert model_config.internal_pa_capital <= model_config.total_fund_capital * 0.96
        assert buffer_after_margin >= 0
    finally:
        st.session_state.clear()


def test_wizard_default_allocation_ignores_stale_schedule_path(tmp_path: Path) -> None:
    helpers = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
    default_allocation = helpers["_default_capital_allocation"]

    allocation = default_allocation(
        total_fund_capital=1000.0,
        reference_sigma=0.01,
        volatility_multiple=3.0,
        financing_model="schedule",
        schedule_path=tmp_path / "missing-schedule.csv",
        term_months=1.0,
    )

    assert allocation["margin_requirement"] == pytest.approx(30.0)
    assert allocation["internal_pa_capital"] == pytest.approx(960.0)
