"""Config patch chain helpers for wizard config chat."""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

import yaml

from pa_core.llm.config_patch import (
    ConfigPatch,
    ConfigPatchValidationError,
    allowed_wizard_schema,
    empty_patch,
    validate_patch_dict,
)

try:
    from pa_core.llm.tracing import langsmith_tracing_context as _langsmith_tracing_context
    from pa_core.llm.tracing import resolve_trace_url as _resolve_trace_url
except ImportError:  # pragma: no cover - defensive import fallback
    _langsmith_tracing_context = None
    _resolve_trace_url = None

_DEFAULT_SAFETY_RULES: tuple[str, ...] = (
    "Do not include secrets or credentials.",
    "Do not request workflow edits.",
    "Do not request file writes.",
)

_CHAIN_TOP_LEVEL_KEYS: tuple[str, ...] = ("patch", "summary", "risk_flags")
_TRACE_META_KEYS: tuple[str, ...] = (
    "trace_url",
    "langsmith_trace_url",
    "trace_id",
    "run_id",
    "run",
    "trace",
    "callback",
    "callbacks",
    "langsmith",
    "trace_info",
)


@dataclass(frozen=True)
class ConfigPatchChainResult:
    """Validated and normalized chain result payload."""

    patch: ConfigPatch
    summary: str
    risk_flags: list[str]
    unknown_output_keys: list[str]
    status: str = "accepted"
    error: dict[str, Any] | None = None
    risk_flags_detail: dict[str, list[str]] = field(default_factory=dict)
    rejected_patch_keys: list[str] | None = None
    rejected_patch_paths: list[str] | None = None
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

    patch_payload = payload.get("patch", {})
    validation_error: dict[str, Any] | None = None
    rejected_patch_keys: list[str] = []
    rejected_patch_paths: list[str] = []
    status = "accepted"
    try:
        patch = validate_patch_dict(patch_payload)
    except ConfigPatchValidationError as exc:
        status = "rejected"
        patch = empty_patch()
        rejected_patch_keys = list(exc.unknown_keys or [])
        rejected_patch_paths = list(exc.unknown_paths or [])
        validation_error = _build_validation_error(exc)
    summary = str(payload.get("summary", "")).strip()
    risk_flags = _normalize_risk_flags(payload.get("risk_flags", []))
    risk_flags_detail: dict[str, list[str]] = {}
    if unknown_keys and "stripped_unknown_output_keys" not in risk_flags:
        risk_flags.append("stripped_unknown_output_keys")
    if unknown_keys:
        risk_flags_detail["unknown_keys"] = list(unknown_keys)
    if rejected_patch_keys:
        risk_flags_detail["rejected_patch_keys"] = list(rejected_patch_keys)
    if status == "rejected" and "rejected_unknown_patch_fields" not in risk_flags:
        risk_flags.append("rejected_unknown_patch_fields")
    return ConfigPatchChainResult(
        patch=patch,
        summary=summary,
        risk_flags=risk_flags,
        unknown_output_keys=unknown_keys,
        status=status,
        error=validation_error,
        risk_flags_detail=risk_flags_detail,
        rejected_patch_keys=rejected_patch_keys,
        rejected_patch_paths=rejected_patch_paths,
    )


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

    trace_url: str | None = None
    tracing_context = (
        _langsmith_tracing_context(
            project_name="portable-alpha-config-chat",
            tags=("wizard", "config-chat"),
            metadata={"provider": provider_name or "unknown", "model": model_name or "unknown"},
        )
        if _langsmith_enabled()
        else nullcontext()
    )
    with tracing_context:
        raw_output = invoke_llm(prompt)

    normalized_output, extracted_trace_url = _split_trace_metadata(raw_output)
    if _langsmith_enabled():
        trace_url = extracted_trace_url
    result = parse_chain_output(normalized_output)
    return replace(result, trace_url=trace_url)


def _coerce_output_mapping(raw_output: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(raw_output, Mapping):
        return dict(raw_output)
    if isinstance(raw_output, str):
        text = raw_output.strip()
        if not text:
            raise ValueError("chain output must be a non-empty mapping payload")
        try:
            parsed_json = json.loads(text)
        except json.JSONDecodeError:
            parsed_json = None
        if isinstance(parsed_json, Mapping):
            return dict(parsed_json)
        parsed_yaml = yaml.safe_load(text)
        if isinstance(parsed_yaml, Mapping):
            return dict(parsed_yaml)
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


def _build_validation_error(exc: ConfigPatchValidationError) -> dict[str, Any]:
    error: dict[str, Any] = {
        "kind": "validation_error",
        "message": str(exc),
        "unknown_keys": list(exc.unknown_keys or []),
        "unknown_paths": list(exc.unknown_paths or []),
    }
    if exc.field_name is not None:
        error["field_name"] = exc.field_name
    if exc.expected_type is not None:
        error["expected_type"] = exc.expected_type
    if exc.actual_type is not None:
        error["actual_type"] = exc.actual_type
    return error


def _langsmith_enabled() -> bool:
    return bool(_langsmith_tracing_context and os.getenv("LANGSMITH_API_KEY"))


def _split_trace_metadata(raw_output: Any) -> tuple[str | Mapping[str, Any], str | None]:
    if isinstance(raw_output, Mapping):
        payload_dict = dict(raw_output)
        trace_url = _extract_trace_from_mapping(payload_dict)
        return payload_dict, trace_url
    if (
        isinstance(raw_output, tuple)
        and len(raw_output) == 2
        and isinstance(raw_output[0], (str, Mapping))
    ):
        payload = raw_output[0]
        trace_url = _extract_trace_candidate(raw_output[1])
        if isinstance(payload, Mapping):
            payload_dict = dict(payload)
            trace_from_payload = _extract_trace_from_mapping(payload_dict)
            return payload_dict, trace_from_payload or trace_url
        return payload, trace_url
    return raw_output, None


def _extract_trace_from_mapping(payload: dict[str, Any]) -> str | None:
    direct_url = payload.pop("trace_url", None)
    if isinstance(direct_url, str) and direct_url.strip():
        return direct_url.strip()

    alt_url = payload.pop("langsmith_trace_url", None)
    if isinstance(alt_url, str) and alt_url.strip():
        return alt_url.strip()

    for key in _TRACE_META_KEYS:
        if key in ("trace_url", "langsmith_trace_url"):
            continue
        if key not in payload:
            continue
        candidate = payload.pop(key)
        resolved = _extract_trace_candidate(candidate)
        if resolved:
            return resolved
    return None


def _extract_trace_candidate(candidate: Any) -> str | None:
    if candidate is None:
        return None
    if isinstance(candidate, Mapping):
        for key in ("trace_url", "url", "run_url", "trace_id", "run_id", "run", "trace"):
            if key not in candidate:
                continue
            resolved = _extract_trace_candidate(candidate[key])
            if resolved:
                return resolved
        for value in candidate.values():
            resolved = _extract_trace_candidate(value)
            if resolved:
                return resolved
        return None
    if isinstance(candidate, (list, tuple)):
        for item in candidate:
            resolved = _extract_trace_candidate(item)
            if resolved:
                return resolved
        return None
    if _resolve_trace_url is None:
        return None
    return _resolve_trace_url(candidate)


__all__ = [
    "ConfigPatchChainResult",
    "build_config_patch_prompt",
    "parse_chain_output",
    "run_config_patch_chain",
]
