"""LLM integration surface for Portable Alpha."""

from .prompts import (
    build_comparison_prompt,
    build_config_wizard_prompt,
    build_result_explanation_prompt,
)
from .provider import LLMProviderConfig, create_llm
from .tracing import langsmith_tracing_context, resolve_trace_url

__all__ = [
    "LLMProviderConfig",
    "build_comparison_prompt",
    "build_config_wizard_prompt",
    "build_result_explanation_prompt",
    "create_llm",
    "langsmith_tracing_context",
    "resolve_trace_url",
]
