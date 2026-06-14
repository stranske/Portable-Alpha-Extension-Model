"""Deterministic fleet-record emission for PA scenario/simulation runs.

Wires the PA core scenario run-completion boundary
(:meth:`pa_core.orchestrator.SimulatorOrchestrator.run` /
:meth:`~pa_core.orchestrator.SimulatorOrchestrator.run_sweep`) into the
dashboard-safe LangSmith fleet-record stream.

This module is intentionally **egress-free**: it never invokes an LLM or a
LangSmith client. It only appends a hashed, domain-only record to the local
``langsmith-fleet.ndjson`` artifact via
:func:`pa_core.llm.langsmith_fleet.record_fleet_event`. With
``LANGSMITH_API_KEY`` unset the record is written with a ``no_secret`` status
so the deterministic, proprietary-data path stays in-perimeter (no raw
holdings, prompts, model output, or PII leave the process).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Mapping

from pa_core.llm.langsmith_fleet import (
    FleetContext,
    config_fingerprint,
    hash_reference,
    record_fleet_event,
)

logger = logging.getLogger(__name__)

SCENARIO_RUN_OPERATION = "scenario-run"
SCENARIO_SWEEP_OPERATION = "scenario-sweep"
SCENARIO_DASHBOARD_SURFACE = "scenario-analysis"
DEFAULT_BENCHMARK = "Base"


def _config_mapping(config: Any) -> Mapping[str, Any] | None:
    """Return a plain mapping view of a ModelConfig (or mapping) for hashing."""

    dump = getattr(config, "model_dump", None)
    if callable(dump):
        try:
            data = dump()
        except Exception:
            logger.warning("Failed to dump config for fleet hashing", exc_info=True)
            return None
        return data if isinstance(data, Mapping) else None
    return config if isinstance(config, Mapping) else None


def _summary_metric_delta(summary: Any, *, benchmark: str = DEFAULT_BENCHMARK) -> float | None:
    """Return a single safe scalar summarising the run's terminal-return spread.

    Prefers the best active sleeve's terminal annual return minus the benchmark
    ("Base") sleeve's; falls back to the cross-agent spread. This is an
    aggregate float only — never a raw distribution.
    """

    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a core dependency
        return None
    if not isinstance(summary, pd.DataFrame) or summary.empty:
        return None
    if "terminal_AnnReturn" not in summary.columns or "Agent" not in summary.columns:
        return None
    col = pd.to_numeric(summary["terminal_AnnReturn"], errors="coerce")
    agents = summary["Agent"].astype(str)
    base_mask = agents == benchmark
    if base_mask.any():
        base_val = col[base_mask].iloc[0]
        active = col[~base_mask]
        if pd.notna(base_val) and active.notna().any():
            return float(active.max() - base_val)
    if col.notna().any():
        return float(col.max() - col.min())
    return None


def _artifact_reference(summary: Any) -> str | None:
    """Return a stable hash of the summary's *shape* — never its raw values."""

    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a core dependency
        return None
    if not isinstance(summary, pd.DataFrame) or summary.empty:
        return None
    agents: list[str] = []
    if "Agent" in summary.columns:
        agents = sorted(summary["Agent"].astype(str).tolist())
    return hash_reference(
        {
            "rows": int(summary.shape[0]),
            "cols": sorted(str(column) for column in summary.columns),
            "agents": agents,
        }
    )


def record_scenario_run(
    config: Any,
    summary: Any,
    *,
    seed: int | str | None,
    latency_ms: int | None = None,
    operation: str = SCENARIO_RUN_OPERATION,
    run_id: str | None = None,
    path: Path | None = None,
) -> None:
    """Emit one dashboard-safe fleet record for a completed scenario run.

    Best-effort and egress-free: any failure is swallowed so deterministic run
    telemetry can never break the simulation path.
    """

    try:
        record_fleet_event(
            FleetContext(
                operation=operation,
                run_id=run_id,
                seed=seed if seed is not None else "unknown",
                latency_ms=latency_ms,
                status="success" if os.getenv("LANGSMITH_API_KEY") else "no_secret",
                config_hash=config_fingerprint(_config_mapping(config)),
                metric_delta=_summary_metric_delta(summary),
                dashboard_surface=SCENARIO_DASHBOARD_SURFACE,
                artifact_ref=_artifact_reference(summary),
                validation_status="deterministic",
            ),
            path=path,
        )
    except Exception:
        # Telemetry is non-critical; never propagate into the run path.
        logger.warning("Failed to record scenario fleet event", exc_info=True)


__all__ = [
    "DEFAULT_BENCHMARK",
    "SCENARIO_DASHBOARD_SURFACE",
    "SCENARIO_RUN_OPERATION",
    "SCENARIO_SWEEP_OPERATION",
    "record_scenario_run",
]
