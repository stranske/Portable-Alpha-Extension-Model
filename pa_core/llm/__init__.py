"""LLM integration surface for Portable Alpha."""

from .config_patch_chain import (
    ConfigPatchChainResult,
    build_config_patch_prompt,
    parse_chain_output,
    run_config_patch_chain,
)
from .prompts import (
    build_comparison_prompt,
    build_config_wizard_prompt,
    build_result_explanation_prompt,
)
from .provider import LLMProviderConfig, create_llm
from .result_explain import explain_results_details
from .tracing import langsmith_tracing_context, maybe_enable_langsmith_tracing, resolve_trace_url

__all__ = [
    "ConfigPatchChainResult",
    "LLMProviderConfig",
    "build_comparison_prompt",
    "build_config_patch_prompt",
    "build_config_wizard_prompt",
    "build_result_explanation_prompt",
    "create_llm",
    "explain_results_details",
    "langsmith_tracing_context",
    "maybe_enable_langsmith_tracing",
    "parse_chain_output",
    "resolve_trace_url",
    "run_config_patch_chain",
]
