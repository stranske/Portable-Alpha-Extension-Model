"""Explain Results panel for the Streamlit Results page."""

from __future__ import annotations

import hashlib
import json
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
from pa_core.llm.result_explain import explain_results_details

DEFAULT_QUESTIONS = (
    "Summarize the key drivers of risk and return. Highlight tracking error, CVaR, "
    "and breach probability trade-offs, then suggest one practical next step."
)

_CACHE_KEY = "explain_results_cache"
_PROVIDER_KEY = "explain_results_provider"
_API_KEY_KEY = "explain_results_api_key"
_MODEL_KEY = "explain_results_model"
_QUESTIONS_KEY = "explain_results_questions"


def _cache_bucket() -> dict[str, dict[str, Any]]:
    cache = st.session_state.get(_CACHE_KEY)
    if not isinstance(cache, dict):
        cache = {}
        st.session_state[_CACHE_KEY] = cache
    return cache


def _manifest_seed(manifest: Mapping[str, Any] | None) -> Any:
    if isinstance(manifest, Mapping):
        return manifest.get("seed")
    return None


def _cache_key(
    *,
    xlsx_path: str,
    manifest: Mapping[str, Any] | None,
    provider: str,
    model: str,
    questions: str,
) -> str:
    payload = {
        "xlsx_path": xlsx_path,
        "manifest_seed": _manifest_seed(manifest),
        "provider": provider,
        "model": model,
        "questions": questions,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    ).hexdigest()
    return digest[:16]


def _inputs_summary(
    *,
    xlsx_path: str,
    manifest: Mapping[str, Any] | None,
    provider: str,
    model: str,
    questions: str,
) -> dict[str, Any]:
    return {
        "xlsx_path": str(Path(xlsx_path)),
        "manifest_seed": _manifest_seed(manifest),
        "provider": provider,
        "model": model,
        "questions": questions,
    }


def render_explain_results_panel(
    *,
    summary_df: pd.DataFrame,
    manifest: Mapping[str, Any] | None,
    xlsx_path: str,
) -> None:
    """Render Explain Results controls and output."""

    st.subheader("Explain Results")
    st.text_area(
        "Questions",
        value=st.session_state.get(_QUESTIONS_KEY, DEFAULT_QUESTIONS),
        key=_QUESTIONS_KEY,
        help="Use this to steer what the explanation should focus on.",
    )

    with st.expander("LLM Settings", expanded=False):
        provider_options = ["openai", "anthropic", "azure_openai"]
        default_provider = str(st.session_state.get(_PROVIDER_KEY, "openai")).lower()
        if default_provider not in provider_options:
            default_provider = "openai"

        st.selectbox(
            "Provider",
            provider_options,
            index=provider_options.index(default_provider),
            key=_PROVIDER_KEY,
        )

        current_provider = str(st.session_state.get(_PROVIDER_KEY, default_provider)).lower()
        if not st.session_state.get(_API_KEY_KEY):
            env_key = default_api_key(current_provider)
            if env_key:
                st.session_state[_API_KEY_KEY] = env_key

        st.text_input(
            "API Key",
            value="",
            key=_API_KEY_KEY,
            type="password",
            help="Enter a literal key or an environment variable name.",
        )
        st.text_input(
            "Model (optional)",
            value=st.session_state.get(_MODEL_KEY, ""),
            key=_MODEL_KEY,
        )

    questions = st.session_state.get(_QUESTIONS_KEY) or DEFAULT_QUESTIONS
    provider = str(st.session_state.get(_PROVIDER_KEY, "openai")).lower()
    model = str(st.session_state.get(_MODEL_KEY, "")).strip()

    cache = _cache_bucket()
    cache_key = _cache_key(
        xlsx_path=xlsx_path,
        manifest=manifest,
        provider=provider,
        model=model,
        questions=questions,
    )
    cached = cache.get(cache_key)

    if st.button("Explain Results", key=f"explain_results_btn_{cache_key}"):
        with st.spinner("Generating explanation..."):
            if cached is None:
                try:
                    raw_key = st.session_state.get(_API_KEY_KEY)
                    resolved_key = resolve_api_key_input(raw_key) or default_api_key(provider)
                    config = resolve_llm_provider_config(
                        provider=provider,
                        model=model or None,
                        api_key=resolved_key,
                    )

                    text, trace_url, payload = explain_results_details(summary_df, manifest)
                    cached = {
                        "text": text,
                        "trace_url": trace_url,
                        "payload": payload,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "inputs_summary": _inputs_summary(
                            xlsx_path=xlsx_path,
                            manifest=manifest,
                            provider=config.provider_name,
                            model=config.model_name or "",
                            questions=questions,
                        ),
                    }
                    cache[cache_key] = cached
                except ModuleNotFoundError:
                    st.error("LLM features unavailable. Install .[llm] to enable Explain Results.")
                    return
                except ValueError as exc:
                    st.error(str(exc))
                    return
                except Exception:
                    st.error("Failed to generate explanation.")
                    return

    if cached is None:
        st.info("Click Explain Results to generate a summary.")
        return

    st.markdown(cached["text"])
    if cached.get("trace_url"):
        st.caption(f"Trace URL: {cached['trace_url']}")

    txt_name = f"explain_results_{Path(xlsx_path).stem}.txt"
    json_name = f"explain_results_{Path(xlsx_path).stem}.json"

    json_payload = {
        "text": cached["text"],
        "trace_url": cached.get("trace_url"),
        "created_at": cached["created_at"],
        "inputs_summary": cached["inputs_summary"],
        "payload": cached.get("payload", {}),
    }

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download TXT",
            data=cached["text"],
            file_name=txt_name,
            mime="text/plain",
        )
    with col2:
        st.download_button(
            "Download JSON",
            data=json.dumps(json_payload, sort_keys=True, indent=2, default=str),
            file_name=json_name,
            mime="application/json",
        )


__all__ = ["render_explain_results_panel"]
