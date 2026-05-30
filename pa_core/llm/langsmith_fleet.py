"""Dashboard-safe LangSmith fleet records for Portable Alpha LLM surfaces."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

FLEET_SCHEMA = "langsmith-fleet/v1"
FLEET_REPO = "stranske/Portable-Alpha-Extension-Model"
FLEET_SURFACE = "scenario-analysis"
FLEET_GITHUB_ISSUE = "stranske/Portable-Alpha-Extension-Model#1802"
DEFAULT_FLEET_PATH = Path("artifacts/langsmith/langsmith-fleet.ndjson")
MAX_RECORDS = 2000


@dataclass(frozen=True)
class FleetContext:
    """Common trace context for dashboard-compatible fleet records."""

    operation: str
    run_id: str | None = None
    scenario_id: str | None = None
    provider: str | None = None
    model: str | None = None
    trace_id: str | None = None
    trace_url: str | None = None
    latency_ms: int | None = None
    status: str = "success"
    error_category: str | None = None
    config_hash: str | None = None
    seed: int | str | None = None
    metric_delta: float | None = None
    dashboard_surface: str | None = None
    prompt_hash: str | None = None
    output_hash: str | None = None
    artifact_ref: str | None = None
    validation_status: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


def default_fleet_artifact_path() -> Path:
    configured = os.getenv("PAEM_LANGSMITH_FLEET_PATH")
    return Path(configured).expanduser() if configured else DEFAULT_FLEET_PATH


def hash_reference(value: Any) -> str | None:
    """Return a stable non-sensitive reference hash for a value."""

    if value is None:
        return None
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))
    else:
        text = str(value)
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_fingerprint(config: Mapping[str, Any] | None) -> str | None:
    if not isinstance(config, Mapping):
        return None
    return hash_reference(config)


def build_fleet_record(context: FleetContext) -> dict[str, Any]:
    run_id = context.run_id or hash_reference(
        {
            "operation": context.operation,
            "scenario_id": context.scenario_id,
            "config_hash": context.config_hash,
            "seed": context.seed,
            "metric_delta": context.metric_delta,
            "dashboard_surface": context.dashboard_surface,
        }
    )
    scenario_id = context.scenario_id or run_id
    config_hash = context.config_hash or hash_reference(
        {"operation": context.operation, "run_id": run_id, "scenario_id": scenario_id}
    )
    seed = context.seed if context.seed is not None else "unknown"
    metric_delta = context.metric_delta if context.metric_delta is not None else 0.0
    domain = {
        "operation": context.operation,
        "run_id": run_id,
        "scenario_id": scenario_id,
        "config_hash": config_hash,
        "seed": seed,
        "metric_delta": metric_delta,
        "dashboard_surface": context.dashboard_surface,
        "prompt_hash": context.prompt_hash,
        "output_hash": context.output_hash,
        "artifact_ref": context.artifact_ref,
        "validation_status": context.validation_status,
        "error_category": context.error_category,
        **dict(context.extra),
    }
    return _drop_none(
        {
            "schema_version": FLEET_SCHEMA,
            "repo": FLEET_REPO,
            "surface": FLEET_SURFACE,
            "recorded_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "operation": context.operation,
            "run_id": run_id,
            "status": context.status,
            "github_issue": FLEET_GITHUB_ISSUE,
            "provider": context.provider,
            "model": context.model,
            "trace_id": context.trace_id,
            "trace_url": context.trace_url,
            "latency_ms": context.latency_ms,
            "domain": _drop_none(domain),
        }
    )


def append_fleet_records(
    records: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    *,
    path: Path | None = None,
    max_records: int = MAX_RECORDS,
) -> None:
    if not records:
        return
    target = path or default_fleet_artifact_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    existing: list[str] = []
    if target.exists():
        existing = [
            line
            for line in target.read_text(encoding="utf-8").splitlines()
            if _is_current_fleet_record_line(line)
        ]
    incoming = [
        json.dumps(_json_safe(dict(record)), sort_keys=True, separators=(",", ":"))
        for record in records
    ]
    retained = (existing + incoming)[-max_records:]
    target.write_text("\n".join(retained) + "\n", encoding="utf-8")


def record_fleet_event(context: FleetContext, *, path: Path | None = None) -> None:
    append_fleet_records([build_fleet_record(context)], path=path)


def _drop_none(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in payload.items() if value is not None}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _is_current_fleet_record_line(line: str) -> bool:
    if not line:
        return False
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return False
    return (
        isinstance(record, Mapping)
        and record.get("schema_version") == FLEET_SCHEMA
        and record.get("repo") == FLEET_REPO
    )


__all__ = [
    "FLEET_SCHEMA",
    "FLEET_GITHUB_ISSUE",
    "FLEET_REPO",
    "FLEET_SURFACE",
    "FleetContext",
    "append_fleet_records",
    "build_fleet_record",
    "config_fingerprint",
    "default_fleet_artifact_path",
    "hash_reference",
    "record_fleet_event",
]
