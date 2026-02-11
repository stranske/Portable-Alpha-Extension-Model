"""Tracing helpers for optional LangSmith integration."""

from __future__ import annotations

from contextlib import contextmanager
import os
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import quote

DEFAULT_LANGSMITH_TRACE_BASE_URL = "https://smith.langchain.com/r/"


def _is_nonempty(value: str | None) -> bool:
    return bool(value and value.strip())


@contextmanager
def langsmith_tracing_context(
    *,
    project_name: str | None = None,
    tags: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    """Yield a tracing context when LangSmith is configured.

    When `LANGSMITH_API_KEY` is unset or empty, this is a no-op context manager.
    LangSmith SDK imports are deferred until context entry.
    """

    if not _is_nonempty(os.getenv("LANGSMITH_API_KEY")):
        yield
        return

    try:
        from langsmith import run_helpers
    except Exception:
        yield
        return

    context_kwargs: dict[str, Any] = {"enabled": True}
    if project_name:
        context_kwargs["project_name"] = project_name
    if tags:
        context_kwargs["tags"] = list(tags)
    if metadata:
        context_kwargs["metadata"] = dict(metadata)

    with run_helpers.tracing_context(**context_kwargs):
        yield


def resolve_trace_url(trace_id: str | None, *, base_url: str | None = None) -> str | None:
    """Construct a LangSmith trace URL without any network calls."""

    if not _is_nonempty(trace_id):
        return None

    clean_trace_id = quote(trace_id.strip(), safe="")
    raw_base = base_url or os.getenv("LANGSMITH_TRACE_BASE_URL") or DEFAULT_LANGSMITH_TRACE_BASE_URL
    clean_base = raw_base.rstrip("/")
    return f"{clean_base}/{clean_trace_id}"
