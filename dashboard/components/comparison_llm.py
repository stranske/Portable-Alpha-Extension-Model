"""Run-to-run comparison panel for LLM analysis in the Results page.

This module provides the initial Streamlit component scaffold for comparing
the current run against a previous run. The structure mirrors the Trend
reference component and is intentionally lightweight until follow-up tasks add
full controls, prompts, and export actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import streamlit as st

DEFAULT_COMPARISON_QUESTIONS = (
    "Compare these runs and explain the main drivers of performance changes."
)
_CACHE_KEY = "comparison_llm_cache"


@dataclass(frozen=True)
class ComparisonLLMResult:
    """Cached output from a comparison request."""

    text: str
    trace_url: str | None
    created_at: str


def _cache_bucket() -> dict[str, ComparisonLLMResult]:
    """Return/create session cache bucket for comparison outputs."""

    cache = st.session_state.get(_CACHE_KEY)
    if not isinstance(cache, dict):
        cache = {}
        st.session_state[_CACHE_KEY] = cache
    return cache


def render_comparison_llm_panel(*, run_key: str) -> None:
    """Render a skeleton comparison panel.

    Parameters
    ----------
    run_key
        Stable key used for Streamlit widget state and cache partitioning.
    """

    st.subheader("LLM Comparison")
    st.caption("Compare the current run against a previous run with an LLM summary.")

    _cache_bucket()
    cached = st.session_state[_CACHE_KEY].get(run_key)
    if cached is None:
        st.info(
            "Comparison panel scaffold is ready. Run selectors, prompt controls, "
            "and exports will be enabled in the next tasks."
        )
        return

    st.markdown(cached.text)
    if cached.trace_url:
        st.caption(f"Trace URL: {cached.trace_url}")
    st.caption(f"Generated: {cached.created_at}")


def _new_result(text: str, trace_url: str | None = None) -> ComparisonLLMResult:
    """Create a timestamped comparison result."""

    return ComparisonLLMResult(
        text=text,
        trace_url=trace_url,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


__all__ = [
    "ComparisonLLMResult",
    "DEFAULT_COMPARISON_QUESTIONS",
    "render_comparison_llm_panel",
]
