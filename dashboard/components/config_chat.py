"""Config Chat sidebar controls for Scenario Wizard patch preview/apply flow."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import streamlit as st

_INSTRUCTION_KEY = "config_chat_instruction"
_PREVIEW_KEY = "config_chat_preview"
_STATUS_KEY = "config_chat_status"
_PREVIEW_PATCH_KEY = "preview_patch"
_PREVIEW_UNIFIED_DIFF_KEY = "preview_unified_diff"
_PREVIEW_SIDEBYSIDE_DIFF_KEY = "preview_sidebyside_diff"


PreviewHandler = Callable[[str], Mapping[str, Any]]
ApplyHandler = Callable[[Mapping[str, Any], bool], tuple[bool, str] | None]
RevertHandler = Callable[[], tuple[bool, str] | None]


def render_config_chat_panel(
    *,
    on_preview: PreviewHandler,
    on_apply: ApplyHandler | None = None,
    on_revert: RevertHandler | None = None,
    key_prefix: str = "wizard",
) -> None:
    """Render Config Chat controls with preview/apply/apply+validate/revert actions."""

    instruction_key = f"{key_prefix}_{_INSTRUCTION_KEY}"
    preview_key = f"{key_prefix}_{_PREVIEW_KEY}"
    status_key = f"{key_prefix}_{_STATUS_KEY}"

    st.subheader("Config Chat")
    st.caption("Describe a config change in plain English, then preview before applying.")
    st.text_area(
        "Instruction",
        value=st.session_state.get(instruction_key, ""),
        key=instruction_key,
        help="Example: increase simulations to 5000 and reduce breach tolerance.",
    )

    col_preview, col_apply, col_apply_validate, col_revert = st.columns(4)
    with col_preview:
        preview_clicked = st.button("Preview", key=f"{key_prefix}_config_chat_preview")
    with col_apply:
        apply_clicked = st.button("Apply", key=f"{key_prefix}_config_chat_apply")
    with col_apply_validate:
        apply_validate_clicked = st.button(
            "Apply+Validate", key=f"{key_prefix}_config_chat_apply_validate"
        )
    with col_revert:
        revert_clicked = st.button("Revert", key=f"{key_prefix}_config_chat_revert")

    if preview_clicked:
        _handle_preview(
            instruction=str(st.session_state.get(instruction_key, "")),
            on_preview=on_preview,
            preview_key=preview_key,
            status_key=status_key,
        )
    elif apply_clicked:
        _handle_apply(
            preview_key=preview_key,
            status_key=status_key,
            validate=False,
            on_apply=on_apply,
        )
    elif apply_validate_clicked:
        _handle_apply(
            preview_key=preview_key,
            status_key=status_key,
            validate=True,
            on_apply=on_apply,
        )
    elif revert_clicked:
        _handle_revert(status_key=status_key, on_revert=on_revert)

    _render_status(status_key)
    _render_preview(preview=st.session_state.get(preview_key))


def _handle_preview(
    *,
    instruction: str,
    on_preview: PreviewHandler,
    preview_key: str,
    status_key: str,
) -> None:
    if not instruction.strip():
        st.warning("Enter an instruction before previewing.")
        return
    try:
        preview = dict(on_preview(instruction.strip()))
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(str(exc))
        return
    st.session_state[preview_key] = preview
    patch_payload = preview.get("patch")
    st.session_state[_PREVIEW_PATCH_KEY] = dict(patch_payload) if isinstance(patch_payload, Mapping) else {}
    unified_diff = preview.get("unified_diff")
    st.session_state[_PREVIEW_UNIFIED_DIFF_KEY] = (
        str(unified_diff).strip() if isinstance(unified_diff, str) else ""
    )
    sidebyside_diff = preview.get("sidebyside_diff")
    st.session_state[_PREVIEW_SIDEBYSIDE_DIFF_KEY] = (
        str(sidebyside_diff).strip() if isinstance(sidebyside_diff, str) else ""
    )
    st.session_state[status_key] = "Preview generated."


def _handle_apply(
    *,
    preview_key: str,
    status_key: str,
    validate: bool,
    on_apply: ApplyHandler | None,
) -> None:
    if on_apply is None:
        st.warning("Apply is not configured for this panel yet.")
        return
    preview = st.session_state.get(preview_key)
    if not isinstance(preview, Mapping):
        st.warning("Generate a preview before applying changes.")
        return
    try:
        outcome = on_apply(preview, validate)
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(str(exc))
        return
    if isinstance(outcome, tuple):
        ok, message = outcome
        st.session_state[status_key] = message
        if ok:
            st.success(message)
        else:
            st.error(message)
    else:
        st.session_state[status_key] = "Config changes applied."


def _handle_revert(*, status_key: str, on_revert: RevertHandler | None) -> None:
    if on_revert is None:
        st.warning("Revert is not configured for this panel yet.")
        return
    try:
        outcome = on_revert()
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(str(exc))
        return
    if isinstance(outcome, tuple):
        ok, message = outcome
        st.session_state[status_key] = message
        if ok:
            st.session_state.pop(_PREVIEW_PATCH_KEY, None)
            st.session_state.pop(_PREVIEW_UNIFIED_DIFF_KEY, None)
            st.session_state.pop(_PREVIEW_SIDEBYSIDE_DIFF_KEY, None)
            st.success(message)
        else:
            st.error(message)
    else:
        st.session_state[status_key] = "Reverted to last applied config."


def _render_status(status_key: str) -> None:
    status = st.session_state.get(status_key)
    if isinstance(status, str) and status.strip():
        st.caption(status)


def _render_preview(preview: Any) -> None:
    if not isinstance(preview, Mapping):
        st.info("No preview yet. Enter an instruction and click Preview.")
        return

    summary = str(preview.get("summary", "")).strip()
    if summary:
        st.markdown(f"**Summary:** {summary}")

    risk_flags = preview.get("risk_flags")
    if isinstance(risk_flags, list) and risk_flags:
        st.warning(f"Risk flags: {', '.join(str(flag) for flag in risk_flags)}")

    unified_diff = preview.get("unified_diff") or st.session_state.get(_PREVIEW_UNIFIED_DIFF_KEY)
    if isinstance(unified_diff, str) and unified_diff.strip():
        st.markdown("**Unified diff**")
        st.code(unified_diff, language="diff")

    sidebyside_diff = preview.get("sidebyside_diff") or st.session_state.get(
        _PREVIEW_SIDEBYSIDE_DIFF_KEY
    )
    if isinstance(sidebyside_diff, str) and sidebyside_diff.strip():
        st.markdown("**Side-by-side diff**")
        st.code(sidebyside_diff, language="diff")
        return

    before_text = preview.get("before_text")
    after_text = preview.get("after_text")
    if isinstance(before_text, str) and isinstance(after_text, str):
        st.markdown("**Side-by-side preview**")
        left, right = st.columns(2)
        with left:
            st.code(before_text, language="yaml")
        with right:
            st.code(after_text, language="yaml")


__all__ = ["render_config_chat_panel"]
