"""Tracing helpers for optional LangSmith integration."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator, Literal, Mapping, Sequence
from urllib.parse import quote

DEFAULT_LANGSMITH_TRACE_BASE_URL = "https://smith.langchain.com/r/"
_LANGSMITH_ENABLED: bool | None = None
_TRUTHY = {"1", "true", "yes", "on"}


def _is_nonempty(value: str | None) -> bool:
    return bool(value and value.strip())


def _truthy_env(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY


def maybe_enable_langsmith_tracing() -> bool:
    """Enable LangSmith tracing env wiring when `LANGSMITH_API_KEY` is set."""

    global _LANGSMITH_ENABLED
    if _LANGSMITH_ENABLED is not None:
        return _LANGSMITH_ENABLED

    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not _is_nonempty(api_key):
        _LANGSMITH_ENABLED = False
        return False

    if not _is_nonempty(os.environ.get("LANGCHAIN_API_KEY")):
        os.environ["LANGCHAIN_API_KEY"] = str(api_key).strip()
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    _LANGSMITH_ENABLED = True
    return True


@contextmanager
def langsmith_tracing_context(
    *,
    name: str | None = None,
    run_type: Literal["retriever", "llm", "tool", "chain", "embedding", "prompt", "parser"] = (
        "chain"
    ),
    inputs: Mapping[str, Any] | None = None,
    project_name: str | None = None,
    tags: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Iterator[Any]:
    """Yield a tracing context when LangSmith is configured.

    When `LANGSMITH_API_KEY` is unset or empty, this is a no-op context manager.
    LangSmith SDK imports are deferred until context entry.
    """

    # Keep tests deterministic unless explicitly opted in.
    if os.environ.get("PYTEST_CURRENT_TEST") and not _truthy_env("PA_LANGSMITH_TRACE_TESTS"):
        yield None
        return

    if not maybe_enable_langsmith_tracing():
        yield None
        return

    try:
        from langsmith import run_helpers
    except Exception:
        yield None
        return

    # Trend-compatible path: explicit run object for callers that need URL extraction.
    if name:
        try:
            trace_cm = run_helpers.trace(
                name,
                run_type=run_type,
                inputs=dict(inputs or {}),
                metadata=dict(metadata or {}),
                project_name=project_name,
            )
        except Exception:
            yield None
            return
        with trace_cm as run:
            yield run
        return

    # Lightweight path used by Portable dashboard features.
    context_kwargs: dict[str, Any] = {"enabled": True}
    if project_name:
        context_kwargs["project_name"] = project_name
    if tags:
        context_kwargs["tags"] = list(tags)
    if metadata:
        context_kwargs["metadata"] = dict(metadata)

    with run_helpers.tracing_context(**context_kwargs):
        yield None


def resolve_trace_url(trace: str | Any | None, *, base_url: str | None = None) -> str | None:
    """Construct a LangSmith trace URL without network calls.

    Accepts either a trace id string or a run-like object exposing url helpers.
    """

    if trace is None:
        return None

    # Trend-compatible run object support.
    if not isinstance(trace, str):
        url_attr = getattr(trace, "url", None)
        if isinstance(url_attr, str) and _is_nonempty(url_attr):
            return url_attr.strip()
        if callable(url_attr):
            try:
                value = url_attr()
            except TypeError:
                value = None
            if isinstance(value, str) and _is_nonempty(value):
                return value.strip()
        for method_name in ("get_url", "get_run_url"):
            method = getattr(trace, method_name, None)
            if not callable(method):
                continue
            try:
                value = method()
            except TypeError:
                value = None
            if isinstance(value, str) and _is_nonempty(value):
                return value.strip()
        return None

    if not _is_nonempty(trace):
        return None

    clean_trace_id = quote(trace.strip(), safe="")
    raw_base = base_url or os.getenv("LANGSMITH_TRACE_BASE_URL") or DEFAULT_LANGSMITH_TRACE_BASE_URL
    clean_base = raw_base.rstrip("/")
    return f"{clean_base}/{clean_trace_id}"
