from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import dashboard.components.config_chat as config_chat_module


class _FakeContext:
    def __enter__(self) -> "_FakeContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


@dataclass
class _ApplyCall:
    validate: bool
    preview: dict[str, Any]


class FakeStreamlit:
    def __init__(
        self, *, button_values: dict[str, bool], session_state: dict[str, Any] | None = None
    ):
        self.button_values = dict(button_values)
        self.session_state: dict[str, Any] = session_state or {}
        self.messages: list[tuple[str, str]] = []
        self.code_blocks: list[tuple[str, str | None]] = []

    def subheader(self, message: str) -> None:
        self.messages.append(("subheader", message))

    def caption(self, message: str) -> None:
        self.messages.append(("caption", message))

    def text_area(self, label: str, value: str, key: str, help: str) -> str:
        self.session_state.setdefault(key, value)
        return str(self.session_state[key])

    def columns(self, n: int) -> list[_FakeContext]:
        return [_FakeContext() for _ in range(n)]

    def button(self, label: str, key: str) -> bool:
        return bool(self.button_values.get(key, False))

    def warning(self, message: str) -> None:
        self.messages.append(("warning", message))

    def error(self, message: str) -> None:
        self.messages.append(("error", message))

    def success(self, message: str) -> None:
        self.messages.append(("success", message))

    def info(self, message: str) -> None:
        self.messages.append(("info", message))

    def markdown(self, message: str) -> None:
        self.messages.append(("markdown", message))

    def code(self, body: str, language: str | None = None) -> None:
        self.code_blocks.append((body, language))


def test_render_config_chat_preview_populates_session_and_renders_diff(monkeypatch) -> None:
    fake_st = FakeStreamlit(
        button_values={"wizard_config_chat_preview": True},
        session_state={"wizard_config_chat_instruction": "increase simulations to 5000"},
    )
    monkeypatch.setattr(config_chat_module, "st", fake_st)

    config_chat_module.render_config_chat_panel(
        on_preview=lambda instruction: {
            "summary": f"Applied instruction: {instruction}",
            "risk_flags": ["stripped_unknown_output_keys"],
            "unified_diff": "--- before\n+++ after\n@@\n-n_simulations: 1000\n+n_simulations: 5000\n",
            "before_text": "n_simulations: 1000\n",
            "after_text": "n_simulations: 5000\n",
        }
    )

    preview = fake_st.session_state["wizard_config_chat_preview"]
    assert preview["summary"].startswith("Applied instruction:")
    assert any(kind == "warning" and "Risk flags" in message for kind, message in fake_st.messages)
    assert any(language == "diff" for _, language in fake_st.code_blocks)
    assert any(language == "yaml" for _, language in fake_st.code_blocks)


def test_render_config_chat_apply_and_apply_validate_call_handler(monkeypatch) -> None:
    apply_calls: list[_ApplyCall] = []
    preview_payload = {"summary": "ready"}
    base_session = {"wizard_config_chat_preview": preview_payload}

    fake_apply = FakeStreamlit(
        button_values={"wizard_config_chat_apply": True},
        session_state=dict(base_session),
    )
    monkeypatch.setattr(config_chat_module, "st", fake_apply)
    config_chat_module.render_config_chat_panel(
        on_preview=lambda instruction: {},
        on_apply=lambda preview, validate: (
            apply_calls.append(_ApplyCall(validate=validate, preview=dict(preview))) or (True, "ok")
        ),
    )

    fake_apply_validate = FakeStreamlit(
        button_values={"wizard_config_chat_apply_validate": True},
        session_state=dict(base_session),
    )
    monkeypatch.setattr(config_chat_module, "st", fake_apply_validate)
    config_chat_module.render_config_chat_panel(
        on_preview=lambda instruction: {},
        on_apply=lambda preview, validate: (
            apply_calls.append(_ApplyCall(validate=validate, preview=dict(preview))) or (True, "ok")
        ),
    )

    assert apply_calls == [
        _ApplyCall(validate=False, preview=preview_payload),
        _ApplyCall(validate=True, preview=preview_payload),
    ]


def test_render_config_chat_revert_calls_handler(monkeypatch) -> None:
    called = {"value": False}
    fake_st = FakeStreamlit(button_values={"wizard_config_chat_revert": True})
    monkeypatch.setattr(config_chat_module, "st", fake_st)

    config_chat_module.render_config_chat_panel(
        on_preview=lambda instruction: {},
        on_revert=lambda: (called.__setitem__("value", True) or (True, "reverted")),
    )

    assert called["value"] is True
    assert any(kind == "success" and message == "reverted" for kind, message in fake_st.messages)
