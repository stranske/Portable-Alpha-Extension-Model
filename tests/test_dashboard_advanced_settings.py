"""Tests for the sidebar 'Advanced / Settings' expander (issue #1902).

The theme-file and results-file path inputs used to sit as bare
``st.sidebar.text_input`` widgets at the top of every page. They are now
tucked inside a collapsed ``"Advanced / Settings"`` sidebar expander, and the
home page sets a real page title instead of the Streamlit default ("app").
"""

from __future__ import annotations

from typing import Any

import pytest

import dashboard.app as app


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
