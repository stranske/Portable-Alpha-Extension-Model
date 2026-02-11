"""Config Chat controls for wizard config editing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import streamlit as st

_INSTRUCTION_DEFAULT = (
    "Describe the change you want, for example: increase simulations to 5000 and "
    "reduce breach tolerance."
)


@dataclass(frozen=True)
class ConfigChatActions:
    """Button click results for the current render pass."""

    instruction: str
    preview_clicked: bool
    apply_clicked: bool
    apply_validate_clicked: bool
    revert_clicked: bool


def render_config_chat_panel(
    *,
    key_prefix: str = "wizard_config_chat",
    instruction_help: str | None = None,
    on_preview: Callable[[str], None] | None = None,
    on_apply: Callable[[str], None] | None = None,
    on_apply_validate: Callable[[str], None] | None = None,
    on_revert: Callable[[], None] | None = None,
) -> ConfigChatActions:
    """Render Config Chat controls and invoke callbacks for clicked actions."""

    instruction_key = f"{key_prefix}::instruction"
    instruction = st.text_area(
        "Instruction",
        key=instruction_key,
        value=str(st.session_state.get(instruction_key, _INSTRUCTION_DEFAULT)),
        help=instruction_help
        or "Use plain English to request safe wizard config updates before applying.",
    )
    normalized_instruction = instruction.strip()

    col1, col2 = st.columns(2)
    with col1:
        preview_clicked = st.button("Preview", key=f"{key_prefix}::preview")
        apply_clicked = st.button("Apply", key=f"{key_prefix}::apply")
    with col2:
        apply_validate_clicked = st.button("Apply+Validate", key=f"{key_prefix}::apply_validate")
        revert_clicked = st.button("Revert", key=f"{key_prefix}::revert")

    if preview_clicked and on_preview is not None:
        on_preview(normalized_instruction)
    if apply_clicked and on_apply is not None:
        on_apply(normalized_instruction)
    if apply_validate_clicked and on_apply_validate is not None:
        on_apply_validate(normalized_instruction)
    if revert_clicked and on_revert is not None:
        on_revert()

    return ConfigChatActions(
        instruction=normalized_instruction,
        preview_clicked=preview_clicked,
        apply_clicked=apply_clicked,
        apply_validate_clicked=apply_validate_clicked,
        revert_clicked=revert_clicked,
    )


__all__ = ["ConfigChatActions", "render_config_chat_panel"]
