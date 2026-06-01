"""Deterministic scenario-run fleet-record emission tests (issue #1854).

These exercise the *real* PA core run path
(:class:`pa_core.orchestrator.SimulatorOrchestrator`) and assert that a
dashboard-safe ``langsmith-fleet.ndjson`` record is written with the required
registry domain fields, no raw proprietary data, a ``no_secret`` status when
``LANGSMITH_API_KEY`` is unset, and without any external LLM/LangSmith egress.
"""

from __future__ import annotations

import json

import pandas as pd

from pa_core.config import load_config
from pa_core.llm.langsmith_fleet import FLEET_REPO, FLEET_SCHEMA, FLEET_SURFACE
from pa_core.llm.scenario_fleet import (
    SCENARIO_DASHBOARD_SURFACE,
    SCENARIO_RUN_OPERATION,
    SCENARIO_SWEEP_OPERATION,
    record_scenario_run,
)
from pa_core.orchestrator import SimulatorOrchestrator

_BASIC_CONFIG = {
    "N_SIMULATIONS": 8,
    "N_MONTHS": 6,
    "financing_mode": "broadcast",
    "w_beta_H": 0.6,
    "w_alpha_H": 0.4,
    "risk_metrics": ["terminal_ShortfallProb"],
}

_INDEX_RETURNS = pd.Series([0.01, 0.02, 0.015, 0.03, 0.005, 0.025] * 20)


def _read_last_record(fleet_path) -> dict:
    lines = fleet_path.read_text(encoding="utf-8").splitlines()
    assert lines, "expected at least one fleet record line"
    return json.loads(lines[-1])


def test_scenario_run_emits_fleet_record(monkeypatch, tmp_path) -> None:
    fleet_path = tmp_path / "fleet.ndjson"
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setenv("PAEM_LANGSMITH_FLEET_PATH", str(fleet_path))

    cfg = load_config(_BASIC_CONFIG)
    orch = SimulatorOrchestrator(cfg, _INDEX_RETURNS)
    returns, summary = orch.run(seed=7)

    # The run still returns its normal artifacts unchanged.
    assert "Base" in returns
    assert "terminal_AnnReturn" in summary.columns

    record = _read_last_record(fleet_path)
    assert record["schema_version"] == FLEET_SCHEMA
    assert record["repo"] == FLEET_REPO
    assert record["surface"] == FLEET_SURFACE
    assert record["operation"] == SCENARIO_RUN_OPERATION
    # Registry-required domain fields are present and populated.
    domain = record["domain"]
    assert {"scenario_id", "config_hash", "seed", "metric_delta"}.issubset(domain)
    assert domain["seed"] == 7
    assert domain["config_hash"]
    assert domain["dashboard_surface"] == SCENARIO_DASHBOARD_SURFACE
    assert domain["validation_status"] == "deterministic"
    assert isinstance(domain["metric_delta"], (int, float))
    assert "artifact_ref" in domain and domain["artifact_ref"]


def test_scenario_run_no_secret_status_when_key_unset(monkeypatch, tmp_path) -> None:
    fleet_path = tmp_path / "fleet.ndjson"
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setenv("PAEM_LANGSMITH_FLEET_PATH", str(fleet_path))

    cfg = load_config(_BASIC_CONFIG)
    SimulatorOrchestrator(cfg, _INDEX_RETURNS).run(seed=1)

    record = _read_last_record(fleet_path)
    assert record["status"] == "no_secret"
    # No external trace surface populated on the deterministic path.
    assert "trace_url" not in record
    assert "trace_id" not in record
    assert record.get("provider") is None
    assert record.get("model") is None


def test_scenario_run_record_has_no_external_egress(
    monkeypatch, tmp_path, socket_connect_guard
) -> None:
    """The deterministic emit path must not open any socket."""

    attempts, _ = socket_connect_guard
    fleet_path = tmp_path / "fleet.ndjson"
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setenv("PAEM_LANGSMITH_FLEET_PATH", str(fleet_path))

    cfg = load_config(_BASIC_CONFIG)
    SimulatorOrchestrator(cfg, _INDEX_RETURNS).run(seed=3)

    assert attempts == []
    record = _read_last_record(fleet_path)
    assert record["status"] == "no_secret"


def test_scenario_run_record_excludes_raw_inputs(monkeypatch, tmp_path) -> None:
    """Index-return values and raw config payloads must not leak into the record."""

    fleet_path = tmp_path / "fleet.ndjson"
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setenv("PAEM_LANGSMITH_FLEET_PATH", str(fleet_path))

    cfg = load_config(_BASIC_CONFIG)
    SimulatorOrchestrator(cfg, _INDEX_RETURNS).run(seed=5)

    raw = fleet_path.read_text(encoding="utf-8")
    # Distinct index-return magnitudes never appear verbatim in the record.
    assert "0.015" not in raw
    assert "0.025" not in raw


def test_scenario_sweep_emits_sweep_operation(monkeypatch, tmp_path) -> None:
    fleet_path = tmp_path / "fleet.ndjson"
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setenv("PAEM_LANGSMITH_FLEET_PATH", str(fleet_path))

    cfg = load_config(_BASIC_CONFIG)
    orch = SimulatorOrchestrator(cfg, _INDEX_RETURNS)
    results, summary = orch.run_sweep(seed=2)

    assert summary is not None
    record = _read_last_record(fleet_path)
    assert record["operation"] == SCENARIO_SWEEP_OPERATION
    assert record["domain"]["seed"] == 2


def test_record_scenario_run_swallows_helper_failures(monkeypatch, tmp_path) -> None:
    """A telemetry failure must never propagate into the run path."""

    def _boom(*_args, **_kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr("pa_core.llm.scenario_fleet.record_fleet_event", _boom)
    # Should not raise despite the underlying writer blowing up.
    record_scenario_run(load_config(_BASIC_CONFIG), pd.DataFrame(), seed=0)
