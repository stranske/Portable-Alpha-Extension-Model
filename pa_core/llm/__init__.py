"""LLM integration surface for Portable Alpha.

Keep imports lazy so lightweight helpers (for example ``pa_core.llm.tracing``)
do not force heavy optional imports during package initialization.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .prompts import (
        build_comparison_prompt,
        build_config_wizard_prompt,
        build_result_explanation_prompt,
    )
    from .provider import LLMProviderConfig, create_llm
    from .result_explain import explain_results_details
    from .tracing import langsmith_tracing_context, maybe_enable_langsmith_tracing, resolve_trace_url

__all__ = [
    "LLMProviderConfig",
    "build_comparison_prompt",
    "build_config_wizard_prompt",
    "build_result_explanation_prompt",
    "create_llm",
    "explain_results_details",
    "langsmith_tracing_context",
    "maybe_enable_langsmith_tracing",
    "resolve_trace_url",
]

_LAZY_EXPORTS = {
    "LLMProviderConfig": ("pa_core.llm.provider", "LLMProviderConfig"),
    "build_comparison_prompt": ("pa_core.llm.prompts", "build_comparison_prompt"),
    "build_config_wizard_prompt": ("pa_core.llm.prompts", "build_config_wizard_prompt"),
    "build_result_explanation_prompt": ("pa_core.llm.prompts", "build_result_explanation_prompt"),
    "create_llm": ("pa_core.llm.provider", "create_llm"),
    "explain_results_details": ("pa_core.llm.result_explain", "explain_results_details"),
    "langsmith_tracing_context": ("pa_core.llm.tracing", "langsmith_tracing_context"),
    "maybe_enable_langsmith_tracing": ("pa_core.llm.tracing", "maybe_enable_langsmith_tracing"),
    "resolve_trace_url": ("pa_core.llm.tracing", "resolve_trace_url"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
