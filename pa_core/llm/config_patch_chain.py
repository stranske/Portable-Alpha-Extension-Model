"""Config patch chain helpers for wizard config chat."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from pa_core.llm.config_patch import ConfigPatch, allowed_wizard_schema, validate_patch_dict
from pa_core.llm.tracing import langsmith_tracing_context, resolve_trace_url

_DEFAULT_SAFETY_RULES: tuple[str, ...] = (
    "Do not include secrets or credentials.",
    "Do not request workflow edits.",
    "Do not request file writes.",
)

_CHAIN_TOP_LEVEL_KEYS: tuple[str, ...] = ("patch", "summary", "risk_flags")


@dataclass(frozen=True)
class ConfigPatchChainResult:
    """Validated and normalized chain result payload."""

    patch: ConfigPatch
    summary: str
    risk_flags: list[str]
    trace_url: str | None = None


def build_config_patch_prompt(
    *,
    current_config: Mapping[str, Any],
    instruction: str,
    safety_rules: Sequence[str] | None = None,
) -> str:
    """Build a config-chat prompt with snapshot, schema, and safety constraints."""

    if not isinstance(current_config, Mapping):
        raise TypeError("current_config must be a mapping")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")

    rules = list(safety_rules or _DEFAULT_SAFETY_RULES)
    schema = allowed_wizard_schema()
    return (
        "You are generating a wizard config patch.\n"
        "Return strict JSON with keys: patch, summary, risk_flags.\n\n"
        f"Current config snapshot:\n{json.dumps(dict(current_config), sort_keys=True, indent=2)}\n\n"
        f"Allowed schema (key -> type):\n{json.dumps(schema, sort_keys=True, indent=2)}\n\n"
        f"Safety rules:\n{json.dumps(rules, ensure_ascii=True, indent=2)}\n\n"
        f"Instruction:\n{instruction.strip()}"
    )


def parse_chain_output(raw_output: str | Mapping[str, Any]) -> ConfigPatchChainResult:
    """Parse and validate chain output into a normalized result."""

    payload = _coerce_output_mapping(raw_output)
    unknown_keys = sorted(set(payload) - set(_CHAIN_TOP_LEVEL_KEYS))
    if unknown_keys:
        payload = {key: value for key, value in payload.items() if key in _CHAIN_TOP_LEVEL_KEYS}

    patch = validate_patch_dict(payload.get("patch", {}))
    summary = str(payload.get("summary", "")).strip()
    risk_flags = _normalize_risk_flags(payload.get("risk_flags", []))
    if unknown_keys:
        risk_flags.append("stripped_unknown_output_keys")
    return ConfigPatchChainResult(patch=patch, summary=summary, risk_flags=risk_flags)


def run_config_patch_chain(
    *,
    current_config: Mapping[str, Any],
    instruction: str,
    invoke_llm: Callable[[str], str | Mapping[str, Any]],
    provider_name: str | None = None,
    model_name: str | None = None,
) -> ConfigPatchChainResult:
    """Run config patch chain and attach LangSmith trace URL when enabled."""

    if not callable(invoke_llm):
        raise TypeError("invoke_llm must be callable")

    prompt = build_config_patch_prompt(current_config=current_config, instruction=instruction)

    trace_id: str | None = None
    with langsmith_tracing_context(
        project_name="portable-alpha-config-chat",
        tags=("wizard", "config-chat"),
        metadata={"provider": provider_name or "unknown", "model": model_name or "unknown"},
    ):
        if os.getenv("LANGSMITH_API_KEY"):
            trace_id = uuid4().hex
        raw_output = invoke_llm(prompt)

    result = parse_chain_output(raw_output)
    trace_url = resolve_trace_url(trace_id) if trace_id else None
    return replace(result, trace_url=trace_url)


def _coerce_output_mapping(raw_output: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(raw_output, Mapping):
        return dict(raw_output)
    if isinstance(raw_output, str):
        parsed = json.loads(raw_output)
        if isinstance(parsed, Mapping):
            return dict(parsed)
    raise ValueError("chain output must be a mapping or JSON object string")


def _normalize_risk_flags(raw_flags: Any) -> list[str]:
    if raw_flags is None:
        return []
    if not isinstance(raw_flags, list):
        return [str(raw_flags)]
    normalized: list[str] = []
    seen: set[str] = set()
    for flag in raw_flags:
        text = str(flag).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


__all__ = [
    "ConfigPatchChainResult",
    "build_config_patch_prompt",
    "parse_chain_output",
    "run_config_patch_chain",
]
