"""Tracing helpers for optional LangSmith integration."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator, Literal, Mapping, Sequence
from urllib.parse import quote

logger = logging.getLogger(__name__)

DEFAULT_LANGSMITH_TRACE_BASE_URL = "https://smith.langchain.com/r/"
_LANGSMITH_ENABLED: bool | None = None


def _is_nonempty(value: str | None) -> bool:
    return bool(value and value.strip())


def maybe_enable_langsmith_tracing() -> bool:
    """Enable LangSmith-related env defaults when API key is present."""

    # The SDK can create a local RunTree while remote tracing is disabled.
    # Such an object must not become a dashboard link to an unpublished run.
    for flag in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        value = os.getenv(flag)
        if value is not None:
            if value.strip().lower() not in {"true", "1"}:
                return False
            break
    global _LANGSMITH_ENABLED
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not _is_nonempty(api_key):
        _LANGSMITH_ENABLED = False
        return False

    if not _is_nonempty(os.getenv("LANGCHAIN_API_KEY")):
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
    LangSmith SDK imports are deferred until context entry. The optional LLM
    extra pins langsmith 0.10.15: run_helpers.trace(name, ...) enters an actual
    RunTree and makes it the parent of nested traced invocations. Read its id
    and get_url(); tracing_context alone only sets context and yields no run.
    SDK source: https://github.com/langchain-ai/langsmith-sdk/blob/v0.10.15/python/langsmith/run_helpers.py
    Local core-only environments may omit the SDK; then this yields None.
    """

    if not maybe_enable_langsmith_tracing():
        yield
        return

    try:
        from langsmith import run_helpers
    except ImportError:
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
                tags=list(tags or []),
            )
        except Exception:
            logger.warning("Failed to start LangSmith trace context", exc_info=True)
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
    """Resolve a trace URL from an id string or LangSmith-like run object."""

    if trace is None:
        return None

    if isinstance(trace, str):
        if not _is_nonempty(trace):
            return None
        clean_trace_id = quote(trace.strip(), safe="")
        raw_base = (
            base_url or os.getenv("LANGSMITH_TRACE_BASE_URL") or DEFAULT_LANGSMITH_TRACE_BASE_URL
        )
        clean_base = raw_base.rstrip("/")
        return f"{clean_base}/{clean_trace_id}"

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


def resolve_trace_reference(run: Any | None) -> tuple[str | None, str | None]:
    """Read identity/link from a created SDK run, never from a request UUID.

    URL construction is the run object's responsibility. This only confirms the
    local SDK identity; authenticated retrieval is a separate acceptance gate.
    """
    if run is None or isinstance(run, str):
        return None, None
    try:
        raw_id = getattr(run, "id", None)
        run_id = str(raw_id).strip() if raw_id is not None else ""
        if not run_id:
            return None, None
        return run_id, resolve_trace_url(run)
    except Exception:
        logger.warning("Unable to resolve SDK trace reference", exc_info=False)
        return None, None


__all__ = [
    "langsmith_tracing_context",
    "maybe_enable_langsmith_tracing",
    "resolve_trace_url",
    "resolve_trace_reference",
]
