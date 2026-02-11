"""LLM integration surface for Portable Alpha."""

from .config_patch_chain import ConfigPatchChain, ConfigPatchChainResult
from .prompts import (
    build_comparison_prompt,
    build_config_wizard_prompt,
    build_result_explanation_prompt,
)
from .provider import LLMProviderConfig, create_llm
from .result_explain import explain_results_details
from .tracing import langsmith_tracing_context, resolve_trace_url

__all__ = [
    "LLMProviderConfig",
    "ConfigPatchChain",
    "ConfigPatchChainResult",
    "build_comparison_prompt",
    "build_config_wizard_prompt",
    "build_result_explanation_prompt",
    "create_llm",
    "explain_results_details",
    "langsmith_tracing_context",
    "resolve_trace_url",
]
