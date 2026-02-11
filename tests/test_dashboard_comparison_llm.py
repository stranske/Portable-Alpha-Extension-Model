from __future__ import annotations

from typing import Any

import dashboard.components.comparison_llm as comparison_module


class FakeStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, Any] = {}
        self.messages: list[tuple[str, str]] = []

    def subheader(self, message: str) -> None:
        self.messages.append(("subheader", message))

    def caption(self, message: str) -> None:
        self.messages.append(("caption", message))

    def info(self, message: str) -> None:
        self.messages.append(("info", message))

    def markdown(self, message: str) -> None:
        self.messages.append(("markdown", message))


def test_render_comparison_llm_panel_shows_skeleton_message(monkeypatch) -> None:
    fake_st = FakeStreamlit()
    monkeypatch.setattr(comparison_module, "st", fake_st)

    comparison_module.render_comparison_llm_panel(run_key="run-a-vs-run-b")

    assert "comparison_llm_cache" in fake_st.session_state
    assert any(
        kind == "subheader" and "LLM Comparison" in text for kind, text in fake_st.messages
    )
    assert any(kind == "info" and "scaffold is ready" in text for kind, text in fake_st.messages)
