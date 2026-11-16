"""AppTest journey for Scenario Wizard run → index upload → Results handoff."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from dashboard.utils import CURRENT_RESULTS_PATH_KEY
from pa_core import cli as pa_cli
from pa_core.wizard_schema import AnalysisMode, get_default_config

_WIZARD_PAGE = Path(__file__).resolve().parents[1] / "dashboard" / "pages" / "3_Scenario_Wizard.py"

_MIN_INDEX_CSV = b"Date,Monthly_TR\n" + b"\n".join(
    f"2025-{month:02d}-28,0.01".encode() for month in range(1, 13)
)


class _UploadedIndex:
    """Minimal uploaded-file stand-in for the wizard index CSV uploader."""

    def __init__(self, payload: bytes, *, name: str = "index.csv") -> None:
        self.name = name
        self._payload = payload

    def getvalue(self) -> bytes:
        return self._payload


def _step_5_app() -> AppTest:
    app = AppTest.from_file(str(_WIZARD_PAGE))
    app.session_state["wizard_step"] = 5
    app.session_state["wizard_config"] = get_default_config(AnalysisMode.RETURNS)
    return app.run(timeout=60)


def _run_simulation_button(app: AppTest):
    return next(button for button in app.button if button.label.endswith("Run Simulation"))


def _subheader_text(app: AppTest) -> list[str]:
    return [header.value for header in app.subheader]


def test_index_uploader_survives_rerun_after_run_click(monkeypatch, tmp_path) -> None:
    """Run click must keep the index gate open across reruns and complete the Results handoff.

    Deliberate-break check (issue #2511): clearing ``wizard_run_requested`` after Run makes
    this test fail on the post-click rerun until the persistence logic is restored.
    """
    app = _step_5_app()
    assert not app.exception

    app = _run_simulation_button(app).click().run(timeout=60)
    assert not app.exception
    assert any("Index Data Required" in text for text in _subheader_text(app))

    app = app.run(timeout=60)
    assert not app.exception
    assert any("Index Data Required" in text for text in _subheader_text(app))
    assert any(uploader.label == "Index CSV" for uploader in app.file_uploader)

    page_links: list[str] = []
    output_path = tmp_path / "Outputs.xlsx"

    def fake_page_link(page: str, *, label: str | None = None, **kwargs: object) -> None:
        page_links.append(label or page)

    def fake_uploader(label: str, *args: object, **kwargs: object):
        if label == "Index CSV":
            return _UploadedIndex(_MIN_INDEX_CSV)
        return st.file_uploader(label, *args, **kwargs)

    def fake_main(args: list[str]) -> None:
        output_path.write_bytes(b"wizard-apptest-stub")

    monkeypatch.setattr(st, "page_link", fake_page_link)
    monkeypatch.setattr(st, "file_uploader", fake_uploader)
    monkeypatch.setattr(pa_cli, "main", fake_main)

    app = app.run(timeout=60)
    assert not app.exception
    assert app.session_state[CURRENT_RESULTS_PATH_KEY] == "Outputs.xlsx"
    assert page_links == ["📈 View Results →"]


def test_wizard_run_request_cleared_when_leaving_step_5() -> None:
    """Backing out of step 5 must drop a stale run request."""
    app = _step_5_app()
    app = _run_simulation_button(app).click().run(timeout=60)
    assert app.session_state["wizard_run_requested"] is True

    back = next(button for button in app.button if button.label == "⬅️ Back: Correlations")
    app = back.click().run(timeout=60)
    assert not app.exception
    assert app.session_state.wizard_step == 4
    assert "wizard_run_requested" not in app.session_state
