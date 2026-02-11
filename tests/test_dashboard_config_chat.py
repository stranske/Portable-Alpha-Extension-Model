from __future__ import annotations

from typing import Any

import dashboard.components.config_chat as config_chat_module


class _FakeContext:
    def __enter__(self) -> "_FakeContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class FakeStreamlit:
    def __init__(self, *, clicked: set[str] | None = None) -> None:
        self.session_state: dict[str, Any] = {}
        self.clicked = clicked or set()
        self.labels: list[str] = []
        self.text_area_labels: list[str] = []

    def text_area(self, label: str, key: str, value: str, help: str) -> str:
        self.text_area_labels.append(label)
        self.session_state.setdefault(key, value)
        return str(self.session_state[key])

    def columns(self, n: int) -> list[_FakeContext]:
        return [_FakeContext() for _ in range(n)]

    def button(self, label: str, key: str) -> bool:
        self.labels.append(label)
        return label in self.clicked


def test_render_config_chat_panel_renders_required_controls(monkeypatch) -> None:
    fake_st = FakeStreamlit()
    monkeypatch.setattr(config_chat_module, "st", fake_st)

    config_chat_module.render_config_chat_panel()

    assert fake_st.text_area_labels == ["Instruction"]
    assert "Preview" in fake_st.labels
    assert "Apply" in fake_st.labels
    assert "Apply+Validate" in fake_st.labels
    assert "Revert" in fake_st.labels


def test_render_config_chat_panel_invokes_callbacks_for_clicked_buttons(monkeypatch) -> None:
    fake_st = FakeStreamlit(clicked={"Preview", "Revert"})
    fake_st.session_state["wizard_config_chat::instruction"] = "  increase simulations to 5000  "
    monkeypatch.setattr(config_chat_module, "st", fake_st)

    called: dict[str, Any] = {"preview": None, "apply": None, "apply_validate": None, "revert": 0}

    result = config_chat_module.render_config_chat_panel(
        on_preview=lambda text: called.__setitem__("preview", text),
        on_apply=lambda text: called.__setitem__("apply", text),
        on_apply_validate=lambda text: called.__setitem__("apply_validate", text),
        on_revert=lambda: called.__setitem__("revert", called["revert"] + 1),
    )

    assert result.instruction == "increase simulations to 5000"
    assert result.preview_clicked is True
    assert result.apply_clicked is False
    assert result.apply_validate_clicked is False
    assert result.revert_clicked is True
    assert called["preview"] == "increase simulations to 5000"
    assert called["apply"] is None
    assert called["apply_validate"] is None
    assert called["revert"] == 1
