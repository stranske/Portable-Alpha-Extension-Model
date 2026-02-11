"""Prompt + parse helpers for NL-to-config-patch workflow."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from pa_core.llm.config_patch import (
    ALLOWED_WIZARD_PATCH_FIELDS,
    describe_allowed_patch_schema,
    validate_patch,
)
from pa_core.llm.tracing import langsmith_tracing_context, resolve_trace_url

_OUTPUT_KEYS = frozenset({"patch", "summary", "risk_flags"})
_PATCH_OPS = frozenset({"set", "merge", "remove"})
_DEFAULT_SAFETY_RULES: tuple[str, ...] = (
    "Do not request or reveal secrets, API keys, credentials, or tokens.",
    "Do not propose workflow edits or any file writes.",
    "Return only patch operations for allowlisted wizard config fields.",
)


@dataclass(frozen=True)
class ConfigPatchChainResult:
    """Validated chain output payload used by the dashboard config chat."""

    patch: dict[str, Any]
    summary: str
    risk_flags: list[str]
    trace_url: str | None = None


def build_config_patch_prompt(
    *,
    current_config: Mapping[str, Any],
    instruction: str,
    allowed_schema: Mapping[str, Any] | None = None,
    safety_rules: Sequence[str] | None = None,
) -> str:
    """Build prompt text for NL-to-config-patch generation."""

    schema = allowed_schema or describe_allowed_patch_schema()
    rules = tuple(safety_rules) if safety_rules is not None else _DEFAULT_SAFETY_RULES
    return (
        "You convert natural-language instructions into a safe config patch.\n\n"
        "Current wizard config snapshot:\n"
        f"{json.dumps(dict(current_config), indent=2, sort_keys=True, default=str)}\n\n"
        "Allowlisted schema (keys/types/operations):\n"
        f"{json.dumps(schema, indent=2, sort_keys=True, default=str)}\n\n"
        "Safety rules:\n" + "\n".join(f"- {rule}" for rule in rules) + "\n\n"
        "Return strict JSON with keys: patch, summary, risk_flags.\n"
        "Instruction:\n"
        f"{instruction.strip()}"
    )


def _coerce_mapping(payload: Any) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return {str(key): value for key, value in payload.items()}
    if isinstance(payload, str):
        loaded = json.loads(payload)
        if not isinstance(loaded, Mapping):
            raise ValueError("LLM output JSON must decode to an object.")
        return {str(key): value for key, value in loaded.items()}
    raise ValueError("LLM output must be a mapping or JSON object string.")


def _normalize_risk_flags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _sanitize_patch(raw_patch: Any) -> tuple[dict[str, Any], list[str]]:
    data = _coerce_mapping(raw_patch)
    risk_flags: list[str] = []

    unknown_ops = sorted(set(data) - _PATCH_OPS)
    if unknown_ops:
        risk_flags.extend(f"stripped_unknown_patch_ops:{op}" for op in unknown_ops)

    sanitized: dict[str, Any] = {"set": {}, "merge": {}, "remove": []}
    for op in ("set", "merge"):
        block = data.get(op)
        if block is None:
            continue
        if not isinstance(block, Mapping):
            risk_flags.append(f"invalid_patch_op_type:{op}")
            continue
        for key, value in block.items():
            field = str(key)
            if field not in ALLOWED_WIZARD_PATCH_FIELDS:
                risk_flags.append(f"stripped_unknown_patch_field:{op}.{field}")
                continue
            sanitized[op][field] = value

    remove_block = data.get("remove")
    if remove_block is None:
        pass
    elif not isinstance(remove_block, list):
        risk_flags.append("invalid_patch_op_type:remove")
    else:
        for item in remove_block:
            field = str(item)
            if field not in ALLOWED_WIZARD_PATCH_FIELDS:
                risk_flags.append(f"stripped_unknown_patch_field:remove.{field}")
                continue
            sanitized["remove"].append(field)

    return sanitized, risk_flags


def parse_chain_output(output: Any) -> ConfigPatchChainResult:
    """Parse chain output, strip unknown keys, and return validated patch payload."""

    payload = _coerce_mapping(output)
    risk_flags = _normalize_risk_flags(payload.get("risk_flags"))

    unknown_output_keys = sorted(set(payload) - _OUTPUT_KEYS)
    if unknown_output_keys:
        risk_flags.extend(f"stripped_unknown_output_key:{key}" for key in unknown_output_keys)

    patch_payload, patch_flags = _sanitize_patch(payload.get("patch") or {})
    risk_flags.extend(patch_flags)

    validated = validate_patch(patch_payload)
    summary = str(payload.get("summary") or "").strip()
    return ConfigPatchChainResult(
        patch=validated.to_dict(),
        summary=summary,
        risk_flags=risk_flags,
        trace_url=None,
    )


class ConfigPatchChain:
    """Small orchestrator that builds prompt, invokes provider, and parses output."""

    def __init__(
        self,
        response_provider: Callable[[str], Any],
        *,
        project_name: str = "portable-alpha-config-chat",
    ) -> None:
        self._response_provider = response_provider
        self._project_name = project_name

    def run(
        self,
        *,
        current_config: Mapping[str, Any],
        instruction: str,
    ) -> ConfigPatchChainResult:
        prompt = build_config_patch_prompt(current_config=current_config, instruction=instruction)
        trace_id: str | None = None
        with langsmith_tracing_context(
            project_name=self._project_name,
            tags=("wizard", "config-chat"),
            metadata={"operation": "nl_to_patch"},
        ):
            if os.getenv("LANGSMITH_API_KEY"):
                trace_id = uuid4().hex
            raw_output = self._response_provider(prompt)

        result = parse_chain_output(raw_output)
        if trace_id:
            return replace(result, trace_url=resolve_trace_url(trace_id))
        return result


__all__ = [
    "ConfigPatchChain",
    "ConfigPatchChainResult",
    "build_config_patch_prompt",
    "parse_chain_output",
]
