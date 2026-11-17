"""Run-to-run comparison panel for LLM analysis in the Results page."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import streamlit as st

from dashboard.components.explain_results import _payload_fingerprint
from dashboard.components.llm_settings import (
    default_api_key,
    default_api_version,
    default_base_url,
    default_model,
    default_provider,
    resolve_api_key_input,
    resolve_llm_provider_config,
)
from pa_core.llm.compare_runs import (
    DEFAULT_COMPARISON_QUESTIONS,
    CompareRunsPayload,
    compare_runs,
    format_config_diff,
    load_prior_manifest,
    load_prior_summary,
)
from pa_core.llm.provider import _DEFAULT_MODEL_NAMES
from pa_core.llm.result_explain import _manifest_highlights

_PROVIDER_KEY = "comparison_llm_provider"
_API_KEY_KEY = "comparison_llm_api_key"
_BASE_URL_KEY = "comparison_llm_base_url"
_API_VERSION_KEY = "comparison_llm_api_version"
_MODEL_KEY = "comparison_llm_model"
_QUESTIONS_KEY = "comparison_llm_questions"
_RUN_SELECTOR_KEY = "comparison_llm_run_selector"
_CURRENT_LABEL = "Current run"
_PREVIOUS_LABEL = "Previous run (auto-detected)"
_DEFAULT_TEXT = "Click Compare Runs to generate an explanation."
_MISSING_MESSAGE = "LLM comparison is unavailable because the previous run manifest or summary could not be loaded."
_CACHE_KEY = "comparison_llm_cache"


@dataclass(frozen=True)
class ComparisonLLMResult:
    """Cached output from a comparison request."""

    text: str
    trace_url: str | None
    created_at: str
    config_diff: str = ""
    comparison_id: str = ""


def _cache_bucket() -> dict[str, ComparisonLLMResult]:
    """Return/create session cache bucket for comparison outputs."""

    cache = st.session_state.get(_CACHE_KEY)
    if not isinstance(cache, dict):
        cache = {}
        st.session_state[_CACHE_KEY] = cache
    return cache


def _manifest_identity_payload(manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        return {}
    return {
        "identity": {
            key: manifest.get(key)
            for key in ("seed", "run_name", "config_hash", "data_files", "previous_run")
        },
        "highlights": _manifest_highlights(manifest),
        "cli_args": (
            manifest.get("cli_args") if isinstance(manifest.get("cli_args"), Mapping) else None
        ),
        "config": manifest.get("config") if isinstance(manifest.get("config"), Mapping) else None,
        "wizard": {
            key: manifest.get(key)
            for key in ("wizard", "wizard_config", "wizard_inputs")
            if isinstance(manifest.get(key), Mapping)
        },
    }


def _run_fingerprint(
    summary_df: pd.DataFrame | None, manifest: Mapping[str, Any] | None
) -> str | None:
    if not isinstance(summary_df, pd.DataFrame):
        return None
    payload = {
        "payload_fingerprint": _payload_fingerprint(summary_df, manifest),
        "manifest": _manifest_identity_payload(manifest),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    ).hexdigest()


def _resolve_prior_inputs(
    manifest_data: Mapping[str, Any] | None,
    *,
    prior_summary_df: pd.DataFrame | None = None,
    prior_manifest_data: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, pd.DataFrame | None, Path | None]:
    prior_manifest: dict[str, Any] | None = (
        dict(prior_manifest_data) if isinstance(prior_manifest_data, Mapping) else None
    )
    prior_manifest_path: Path | None = None
    if prior_manifest is None:
        prior_manifest, prior_manifest_path = load_prior_manifest(manifest_data)
    elif isinstance(manifest_data, Mapping):
        prev_ref = manifest_data.get("previous_run")
        if isinstance(prev_ref, str) and prev_ref.strip():
            prior_manifest_path = Path(prev_ref).expanduser()
    if prior_manifest is None:
        return None, None, prior_manifest_path

    prior_summary = prior_summary_df
    if prior_summary is None:
        prior_summary, _ = load_prior_summary(prior_manifest, manifest_path=prior_manifest_path)
    return prior_manifest, prior_summary, prior_manifest_path


def _resolve_request_settings(
    run_key: str,
    *,
    questions: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    endpoint: str | None = None,
    api_version: str | None = None,
) -> dict[str, str]:
    questions_key = f"{_QUESTIONS_KEY}::{run_key}"
    provider_key = f"{_PROVIDER_KEY}::{run_key}"
    base_url_key = f"{_BASE_URL_KEY}::{run_key}"
    api_version_key = f"{_API_VERSION_KEY}::{run_key}"
    model_key = f"{_MODEL_KEY}::{run_key}"

    raw_questions = (
        questions
        if questions is not None
        else str(
            st.session_state.get(
                questions_key,
                st.session_state.get(_QUESTIONS_KEY, DEFAULT_COMPARISON_QUESTIONS),
            )
        )
    )
    resolved_questions = raw_questions.strip() or DEFAULT_COMPARISON_QUESTIONS

    raw_provider = (
        provider
        if provider is not None
        else str(st.session_state.get(provider_key, default_provider()))
    )
    resolved_provider = raw_provider.strip().lower()
    if resolved_provider not in {"openai", "anthropic", "azure_openai"}:
        resolved_provider = "openai"

    raw_model = model if model is not None else str(st.session_state.get(model_key, ""))
    cleaned_model = (
        raw_model.strip()
        or default_model()
        or _DEFAULT_MODEL_NAMES.get(resolved_provider, "gpt-4o-mini")
    )
    effective_model = str(cleaned_model).strip()

    if endpoint is not None:
        resolved_endpoint = endpoint.strip()
    elif resolved_provider == "azure_openai":
        resolved_endpoint = (
            str(st.session_state.get(base_url_key, "")).strip()
            or (default_base_url() or "").strip()
        )
    else:
        resolved_endpoint = (default_base_url() or "").strip()

    if api_version is not None:
        resolved_api_version = api_version.strip()
    elif resolved_provider == "azure_openai":
        resolved_api_version = (
            str(st.session_state.get(api_version_key, "")).strip()
            or (default_api_version() or "").strip()
        )
    else:
        resolved_api_version = ""

    return {
        "questions": resolved_questions,
        "provider": resolved_provider,
        "effective_model": effective_model,
        "endpoint": resolved_endpoint,
        "api_version": resolved_api_version,
    }


def _comparison_identity(
    *,
    summary_df: pd.DataFrame,
    manifest_data: Mapping[str, Any] | None,
    run_key: str,
    prior_summary_df: pd.DataFrame | None = None,
    prior_manifest_data: Mapping[str, Any] | None = None,
    input_fingerprint: str | None = None,
    questions: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    endpoint: str | None = None,
    api_version: str | None = None,
) -> str | None:
    try:
        prior_manifest, prior_summary, _ = _resolve_prior_inputs(
            manifest_data,
            prior_summary_df=prior_summary_df,
            prior_manifest_data=prior_manifest_data,
        )
    except Exception:
        return None
    if prior_manifest is None or prior_summary is None:
        return None

    current_fp = _run_fingerprint(summary_df, manifest_data)
    prior_fp = _run_fingerprint(prior_summary, prior_manifest)
    if current_fp is None or prior_fp is None:
        return None

    request_settings = _resolve_request_settings(
        run_key,
        questions=questions,
        provider=provider,
        model=model,
        endpoint=endpoint,
        api_version=api_version,
    )
    payload = {
        "run_key": run_key,
        "input_fingerprint": input_fingerprint
        or f"{_payload_fingerprint(summary_df, manifest_data)}::{_payload_fingerprint(prior_summary, prior_manifest)}",
        "current_fingerprint": current_fp,
        "prior_fingerprint": prior_fp,
        "config_diff": format_config_diff(manifest_data, prior_manifest),
        "request": request_settings,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    ).hexdigest()
    return f"{run_key}::{digest[:16]}"


def _payload_to_json_dict(
    payload: CompareRunsPayload, result: ComparisonLLMResult
) -> dict[str, Any]:
    return {
        "text": result.text,
        "trace_url": result.trace_url,
        "created_at": result.created_at,
        "comparison_id": result.comparison_id,
        "config_diff": payload.config_diff,
        "metric_catalog_a": payload.metric_catalog_a,
        "metric_catalog_b": payload.metric_catalog_b,
        "questions": payload.questions,
        "prior_manifest_path": payload.prior_manifest_path,
        "prior_summary_path": payload.prior_summary_path,
        "status": payload.status,
        "error_category": payload.error_category,
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
    if result.comparison_id:
        sections.extend(["", f"Comparison ID: {result.comparison_id}"])
    if result.trace_url:
        sections.extend(["", f"Trace URL: {result.trace_url}"])
    if payload.status != "success":
        sections.extend(
            [
                "",
                f"Status: {payload.status}",
                f"Error category: {payload.error_category or 'unknown'}",
            ]
        )
    return "\n".join(sections)


def render_comparison_llm_panel(
    *,
    summary_df: pd.DataFrame,
    manifest_data: Mapping[str, Any] | None,
    run_key: str,
    prior_summary_df: pd.DataFrame | None = None,
    prior_manifest_data: Mapping[str, Any] | None = None,
    input_fingerprint: str | None = None,
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
        base_url_key = f"{_BASE_URL_KEY}::{run_key}"
        api_version_key = f"{_API_VERSION_KEY}::{run_key}"
        model_key = f"{_MODEL_KEY}::{run_key}"

        provider_default = str(st.session_state.get(provider_key, default_provider())).lower()
        if provider_default not in provider_options:
            provider_default = "openai"
        st.selectbox(
            "Provider",
            provider_options,
            index=provider_options.index(provider_default),
            key=provider_key,
        )
        provider = str(st.session_state.get(provider_key, provider_default)).lower()
        api_key_key = f"{_API_KEY_KEY}::{run_key}::{provider}"
        st.session_state.pop(f"{_API_KEY_KEY}::{run_key}", None)
        saved_key = f"{api_key_key}::saved"
        if api_key_key not in st.session_state:
            st.session_state[api_key_key] = st.session_state.get(saved_key, "")
        override = st.text_input(
            "API Key",
            value="",
            key=api_key_key,
            type="password",
            help="Enter a literal key or env var name.",
        )
        st.session_state[saved_key] = override
        st.text_input(
            "Model (optional)",
            value=st.session_state.get(model_key, ""),
            key=model_key,
        )
        if provider == "azure_openai":
            if not st.session_state.get(base_url_key):
                st.session_state[base_url_key] = default_base_url() or ""
            if not st.session_state.get(api_version_key):
                st.session_state[api_version_key] = default_api_version() or ""
            st.text_input(
                "Azure endpoint",
                value=st.session_state.get(base_url_key, ""),
                key=base_url_key,
            )
            st.text_input(
                "Azure API version",
                value=st.session_state.get(api_version_key, ""),
                key=api_version_key,
            )

    cache = _cache_bucket()
    comparison_id = _comparison_identity(
        summary_df=summary_df,
        manifest_data=manifest_data,
        run_key=run_key,
        prior_summary_df=prior_summary_df,
        prior_manifest_data=prior_manifest_data,
        input_fingerprint=input_fingerprint,
    )
    cached = cache.get(comparison_id) if comparison_id is not None else None
    payload: CompareRunsPayload | None = (
        st.session_state.get(f"comparison_payload::{comparison_id}")
        if comparison_id is not None
        else None
    )

    if st.button("Compare Runs", key=f"compare_runs_btn::{run_key}"):
        with st.spinner("Generating run comparison..."):
            if cached is None or payload is None or payload.status != "success":
                try:
                    prior_manifest, prior_summary, _ = _resolve_prior_inputs(
                        manifest_data,
                        prior_summary_df=prior_summary_df,
                        prior_manifest_data=prior_manifest_data,
                    )
                    if prior_manifest is None or prior_summary is None:
                        st.info(_MISSING_MESSAGE)
                        return
                    questions_key = f"{_QUESTIONS_KEY}::{run_key}"
                    provider_key = f"{_PROVIDER_KEY}::{run_key}"
                    base_url_key = f"{_BASE_URL_KEY}::{run_key}"
                    api_version_key = f"{_API_VERSION_KEY}::{run_key}"
                    model_key = f"{_MODEL_KEY}::{run_key}"

                    provider = str(st.session_state.get(provider_key, default_provider())).lower()
                    api_key_key = f"{_API_KEY_KEY}::{run_key}::{provider}"
                    model = str(st.session_state.get(model_key, "")).strip() or None
                    raw_key = st.session_state.get(api_key_key)
                    resolved_key = resolve_api_key_input(raw_key) or default_api_key(provider)
                    azure_base_url = None
                    azure_api_version = None
                    if provider == "azure_openai":
                        azure_base_url = str(st.session_state.get(base_url_key, "")).strip() or None
                        azure_api_version = (
                            str(st.session_state.get(api_version_key, "")).strip() or None
                        )
                    provider_config = resolve_llm_provider_config(
                        provider=provider,
                        model=model,
                        api_key=resolved_key,
                        base_url=azure_base_url,
                        api_version=azure_api_version,
                    )

                    text, trace_url, payload = compare_runs(
                        current_summary=summary_df,
                        current_manifest=manifest_data,
                        questions=str(
                            st.session_state.get(questions_key, DEFAULT_COMPARISON_QUESTIONS)
                        ),
                        provider_name=provider,
                        model_name=model,
                        api_key=resolved_key,
                        provider_config=provider_config,
                    )
                    if comparison_id is None:
                        comparison_id = _comparison_identity(
                            summary_df=summary_df,
                            manifest_data=manifest_data,
                            run_key=run_key,
                            prior_summary_df=prior_summary,
                            prior_manifest_data=prior_manifest,
                            input_fingerprint=input_fingerprint,
                        )
                    assert comparison_id is not None
                    cached = _new_result(text=text, trace_url=trace_url)
                    cached = ComparisonLLMResult(
                        text=cached.text,
                        trace_url=cached.trace_url,
                        created_at=cached.created_at,
                        config_diff=payload.config_diff,
                        comparison_id=comparison_id,
                    )
                    cache.clear()
                    cache[comparison_id] = cached
                    for existing_key in list(st.session_state):
                        if str(existing_key).startswith("comparison_payload::"):
                            st.session_state.pop(existing_key, None)
                    st.session_state[f"comparison_payload::{comparison_id}"] = payload
                except ModuleNotFoundError:
                    st.info("LLM features unavailable. Install .[llm] to enable LLM Comparison.")
                    return
                except Exception as exc:
                    st.error(str(exc))
                    return

    if cached is None or payload is None:
        st.info(_DEFAULT_TEXT)
        return

    st.markdown(cached.text)
    if payload.status != "success":
        st.caption(f"LLM fallback: {payload.status} ({payload.error_category or 'unknown'}).")
        if payload.error_category in {"ModuleNotFoundError", "missing_extra"}:
            st.info("LLM features unavailable. Install .[llm] to enable LLM Comparison.")
    if cached.trace_url:
        st.caption(f"Trace URL: {cached.trace_url}")
    st.caption(f"Generated: {cached.created_at}")

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
