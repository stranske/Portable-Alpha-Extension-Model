"""Prompt builders for LLM-backed explanations and comparisons."""

from __future__ import annotations

import json
from typing import Any


def _require_present(name: str, value: Any) -> Any:
    if value is None:
        raise ValueError(f"{name} is required")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{name} is required")
    return value


def _to_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, default=str)


def build_result_explanation_prompt(result_data: Any) -> str:
    """Build a prompt requesting a concise explanation of model output."""

    _require_present("result_data", result_data)
    return (
        "You are a quantitative analysis assistant. Explain the result data below "
        "in clear, plain language. Include key drivers, risks, and one practical next step.\n\n"
        f"Result Data:\n{_to_json(result_data)}"
    )


def build_comparison_prompt(item_a: Any, item_b: Any) -> str:
    """Build a prompt that compares two inputs objectively."""

    _require_present("item_a", item_a)
    _require_present("item_b", item_b)
    return (
        "Compare Item A and Item B. Highlight the top 3 differences, trade-offs, "
        "and when each is preferable.\n\n"
        f"Item A:\n{_to_json(item_a)}\n\n"
        f"Item B:\n{_to_json(item_b)}"
    )


def build_config_wizard_prompt(config_context: Any) -> str:
    """Build a prompt that asks for configuration guidance."""

    _require_present("config_context", config_context)
    return (
        "Given the configuration context below, propose a safe default configuration "
        "and briefly justify each major field.\n\n"
        f"Configuration Context:\n{_to_json(config_context)}"
    )
