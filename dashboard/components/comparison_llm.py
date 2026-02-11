"""Run-to-run comparison panel for LLM analysis in the Results page."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import streamlit as st

from dashboard.components.llm_settings import (
    default_api_key,
    resolve_api_key_input,
    resolve_llm_provider_config,
)
from pa_core.llm.compare_runs import CompareRunsPayload, compare_runs, load_prior_manifest

DEFAULT_COMPARISON_QUESTIONS = """Compare the current and previous runs and explain:
1. Main performance drivers and constraint effects.
2. Changes in tracking error, breach probability, and CVaR.
3. Whether parameter changes likely improved risk-adjusted outcomes.
4. One practical next run to test."""

_PROVIDER_KEY = "comparison_llm_provider"
_API_KEY_KEY = "comparison_llm_api_key"
_MODEL_KEY = "comparison_llm_model"
_QUESTIONS_KEY = "comparison_llm_questions"
_RUN_SELECTOR_KEY = "comparison_llm_run_selector"
_CURRENT_LABEL = "Current run"
_PREVIOUS_LABEL = "Previous run (auto-detected)"
_DEFAULT_TEXT = "Click Compare Runs to generate an explanation."
_MISSING_MESSAGE = (
    "LLM comparison is unavailable because the previous run manifest or summary could not be loaded."
)
_CACHE_KEY = "comparison_llm_cache"


@dataclass(frozen=True)
class ComparisonLLMResult:
    """Cached output from a comparison request."""

    text: str
    trace_url: str | None
    created_at: str
    config_diff: str = ""


def _cache_bucket() -> dict[str, ComparisonLLMResult]:
    """Return/create session cache bucket for comparison outputs."""

    cache = st.session_state.get(_CACHE_KEY)
    if not isinstance(cache, dict):
        cache = {}
        st.session_state[_CACHE_KEY] = cache
    return cache


def _payload_to_json_dict(
    payload: CompareRunsPayload, result: ComparisonLLMResult
) -> dict[str, Any]:
    return {
        "text": result.text,
        "trace_url": result.trace_url,
        "created_at": result.created_at,
        "config_diff": payload.config_diff,
        "metric_catalog_a": payload.metric_catalog_a,
        "metric_catalog_b": payload.metric_catalog_b,
        "questions": payload.questions,
        "prior_manifest_path": payload.prior_manifest_path,
        "prior_summary_path": payload.prior_summary_path,
    }


def _payload_to_txt(payload: CompareRunsPayload, result: ComparisonLLMResult) -> str:
    sections = [
        "LLM Comparison Output",
        "",
        result.text,
        "",
        "Config Diff",
        payload.config_diff,
    ]
    if result.trace_url:
        sections.extend(["", f"Trace URL: {result.trace_url}"])
    return "\n".join(sections)


def render_comparison_llm_panel(
    *,
    summary_df: pd.DataFrame,
    manifest_data: Mapping[str, Any] | None,
    run_key: str,
) -> None:
    """Render run-to-run comparison controls and output."""
    st.subheader("LLM Comparison")
    st.caption("Compare current vs previous run to explain why results moved.")

    run_options = [_CURRENT_LABEL, _PREVIOUS_LABEL]
    st.selectbox("Runs", run_options, index=1, key=f"{_RUN_SELECTOR_KEY}::{run_key}")
    st.text_area(
        "Questions",
        value=st.session_state.get(_QUESTIONS_KEY, DEFAULT_COMPARISON_QUESTIONS),
        key=f"{_QUESTIONS_KEY}::{run_key}",
        help="Override default comparison questions if needed.",
    )

    with st.expander("LLM Settings", expanded=False):
        provider_options = ["openai", "anthropic", "azure_openai"]
        provider_key = f"{_PROVIDER_KEY}::{run_key}"
        api_key_key = f"{_API_KEY_KEY}::{run_key}"
        model_key = f"{_MODEL_KEY}::{run_key}"

        default_provider = str(st.session_state.get(provider_key, "openai")).lower()
        if default_provider not in provider_options:
            default_provider = "openai"
        st.selectbox(
            "Provider",
            provider_options,
            index=provider_options.index(default_provider),
            key=provider_key,
        )
        provider = str(st.session_state.get(provider_key, default_provider)).lower()
        if not st.session_state.get(api_key_key):
            env_key = default_api_key(provider)
            if env_key:
                st.session_state[api_key_key] = env_key
        st.text_input(
            "API Key",
            value="",
            key=api_key_key,
            type="password",
            help="Enter a literal key or env var name.",
        )
        st.text_input(
            "Model (optional)",
            value=st.session_state.get(model_key, ""),
            key=model_key,
        )

    cached = _cache_bucket().get(run_key)
    payload: CompareRunsPayload | None = st.session_state.get(f"comparison_payload::{run_key}")

    if st.button("Compare Runs", key=f"compare_runs_btn::{run_key}"):
        with st.spinner("Generating run comparison..."):
            try:
                prior_manifest, _ = load_prior_manifest(manifest_data)
                if prior_manifest is None:
                    st.info(_MISSING_MESSAGE)
                    return
                questions_key = f"{_QUESTIONS_KEY}::{run_key}"
                provider_key = f"{_PROVIDER_KEY}::{run_key}"
                api_key_key = f"{_API_KEY_KEY}::{run_key}"
                model_key = f"{_MODEL_KEY}::{run_key}"

                provider = str(st.session_state.get(provider_key, "openai")).lower()
                model = str(st.session_state.get(model_key, "")).strip() or None
                raw_key = st.session_state.get(api_key_key)
                resolved_key = resolve_api_key_input(raw_key) or default_api_key(provider)
                resolve_llm_provider_config(provider=provider, model=model, api_key=resolved_key)

                text, trace_url, payload = compare_runs(
                    current_summary=summary_df,
                    current_manifest=manifest_data,
                    questions=str(
                        st.session_state.get(questions_key, DEFAULT_COMPARISON_QUESTIONS)
                    ),
                    provider_name=provider,
                    model_name=model,
                    api_key=resolved_key,
                )
                cached = _new_result(text=text, trace_url=trace_url)
                cached = ComparisonLLMResult(
                    text=cached.text,
                    trace_url=cached.trace_url,
                    created_at=cached.created_at,
                    config_diff=payload.config_diff,
                )
                _cache_bucket()[run_key] = cached
                st.session_state[f"comparison_payload::{run_key}"] = payload
            except ModuleNotFoundError:
                st.info("LLM features unavailable. Install .[llm] to enable LLM Comparison.")
                return
            except Exception as exc:
                st.error(str(exc))
                return

    if cached is None:
        st.info(_DEFAULT_TEXT)
        return
    st.markdown(cached.text)
    if cached.trace_url:
        st.caption(f"Trace URL: {cached.trace_url}")
    st.caption(f"Generated: {cached.created_at}")

    if payload is None:
        payload = CompareRunsPayload(
            config_diff=cached.config_diff,
            metric_catalog_a={},
            metric_catalog_b={},
            prompt="",
            questions=DEFAULT_COMPARISON_QUESTIONS,
            prior_manifest_path=None,
            prior_summary_path=None,
        )

    stem = Path(str(run_key).split("::")[0]).stem if run_key else "comparison"
    txt_body = _payload_to_txt(payload, cached)
    json_body = json.dumps(
        _payload_to_json_dict(payload, cached), sort_keys=True, indent=2, default=str
    )

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download TXT",
            data=txt_body,
            file_name=f"comparison_llm_{stem}.txt",
            mime="text/plain",
        )
    with col2:
        st.download_button(
            "Download JSON",
            data=json_body,
            file_name=f"comparison_llm_{stem}.json",
            mime="application/json",
        )


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
