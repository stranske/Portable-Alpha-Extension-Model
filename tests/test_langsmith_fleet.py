"""Tests for dashboard-safe LangSmith fleet records."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pa_core.llm.compare_runs import compare_runs
from pa_core.llm.config_patch_chain import run_config_patch_chain
from pa_core.llm.langsmith_fleet import (
    FLEET_GITHUB_ISSUE,
    FLEET_REPO,
    FLEET_SCHEMA,
    FLEET_SURFACE,
    FleetContext,
    append_fleet_records,
    build_fleet_record,
    config_fingerprint,
    hash_reference,
)
from pa_core.llm.result_explain import explain_results_details


def test_build_fleet_record_hashes_domain_metadata() -> None:
    record = build_fleet_record(
        FleetContext(
            operation="result-explanation",
            run_id="run-1",
            scenario_id="scenario-a",
            provider="openai",
            model="gpt-test",
            status="success",
            config_hash=config_fingerprint({"seed": 7, "weights": {"alpha": 0.2}}),
            seed=7,
            metric_delta=0.05,
            dashboard_surface="results-details",
            prompt_hash=hash_reference("prompt text"),
            output_hash=hash_reference("output text"),
        )
    )

    assert record["schema_version"] == FLEET_SCHEMA
    assert record["repo"] == FLEET_REPO
    assert record["surface"] == FLEET_SURFACE
    assert record["run_id"] == "run-1"
    assert record["github_issue"] == FLEET_GITHUB_ISSUE
    assert "recorded_at" in record
    assert "schema" not in record
    assert "generated_at" not in record
    assert record["provider"] == "openai"
    assert record["domain"]["operation"] == "result-explanation"
    assert record["domain"]["scenario_id"] == "scenario-a"
    assert record["domain"]["config_hash"] == config_fingerprint(
        {"seed": 7, "weights": {"alpha": 0.2}}
    )
    assert record["domain"]["seed"] == 7
    assert record["domain"]["metric_delta"] == 0.05
    assert "prompt text" not in json.dumps(record)
    assert "output text" not in json.dumps(record)


def test_build_fleet_record_contains_required_contract_fields_when_context_sparse() -> None:
    record = build_fleet_record(FleetContext(operation="config-patch", status="no_secret"))

    assert record["schema_version"] == "langsmith-fleet/v1"
    assert record["surface"] == "scenario-analysis"
    assert record["run_id"]
    assert record["github_issue"] == "stranske/Portable-Alpha-Extension-Model#1802"
    assert set(["scenario_id", "config_hash", "seed", "metric_delta"]).issubset(record["domain"])
    assert record["domain"]["scenario_id"]
    assert record["domain"]["config_hash"]


def test_append_fleet_records_retains_recent_records(tmp_path) -> None:
    path = tmp_path / "fleet.ndjson"
    records = [
        build_fleet_record(FleetContext(operation=f"op-{index}", status="success"))
        for index in range(3)
    ]

    append_fleet_records(records, path=path, max_records=2)

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["operation"] == "op-1"
    assert json.loads(lines[1])["operation"] == "op-2"


def test_append_fleet_records_filters_legacy_schema_lines(tmp_path) -> None:
    path = tmp_path / "fleet.ndjson"
    legacy_record = {
        "schema": "paem-langsmith-fleet/v0",
        "repo": FLEET_REPO,
        "generated_at": "2026-05-30T00:00:00Z",
        "operation": "scenario-run",
    }
    current_record = build_fleet_record(
        FleetContext(operation="scenario-run", run_id="current-run")
    )
    path.write_text(
        "\n".join(
            [
                json.dumps(legacy_record, sort_keys=True),
                json.dumps(current_record, sort_keys=True),
                "{not-json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    append_fleet_records(
        [build_fleet_record(FleetContext(operation="config-patch", run_id="new-run"))],
        path=path,
        max_records=10,
    )

    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["run_id"] for record in lines] == ["current-run", "new-run"]
    assert all(record["schema_version"] == FLEET_SCHEMA for record in lines)


def test_config_patch_chain_emits_no_secret_fleet_record(monkeypatch, tmp_path) -> None:
    fleet_path = tmp_path / "fleet.ndjson"
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setenv("PAEM_LANGSMITH_FLEET_PATH", str(fleet_path))

    result = run_config_patch_chain(
        current_config={"n_simulations": 1000, "seed": 42},
        instruction="increase simulations",
        provider_name="openai",
        model_name="gpt-test",
        invoke_llm=lambda _prompt: {
            "patch": {"set": {"n_simulations": 5000}},
            "summary": "Updated n_simulations.",
            "risk_flags": [],
        },
    )

    assert result.status == "accepted"
    record = json.loads(fleet_path.read_text().splitlines()[-1])
    assert record["status"] == "no_secret"
    assert record["operation"] == "config-patch"
    assert record["domain"]["validation_status"] == "accepted"
    assert "increase simulations" not in json.dumps(record)


def test_config_patch_chain_records_llm_failures_before_reraising(monkeypatch, tmp_path) -> None:
    fleet_path = tmp_path / "fleet.ndjson"
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setenv("PAEM_LANGSMITH_FLEET_PATH", str(fleet_path))
    monkeypatch.setattr("pa_core.llm.config_patch_chain._langsmith_enabled", lambda: False)

    def fail_llm(_prompt: str) -> str:
        raise TimeoutError("provider timed out")

    with pytest.raises(TimeoutError):
        run_config_patch_chain(
            current_config={"n_simulations": 1000, "seed": 42},
            instruction="increase simulations",
            provider_name="openai",
            model_name="gpt-test",
            invoke_llm=fail_llm,
        )

    record = json.loads(fleet_path.read_text().splitlines()[-1])
    assert record["status"] == "error"
    assert record["operation"] == "config-patch"
    assert record["domain"]["error_category"] == "TimeoutError"
    assert record["domain"]["validation_status"] == "failed"
    assert "increase simulations" not in json.dumps(record)


def test_result_explain_emits_no_secret_record(monkeypatch, tmp_path) -> None:
    fleet_path = tmp_path / "fleet.ndjson"
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setenv("PAEM_LANGSMITH_FLEET_PATH", str(fleet_path))

    text, trace_url, payload = explain_results_details(
        pd.DataFrame(
            {
                "monthly_TE": [0.02, 0.03],
                "stress_delta": [0.01, -0.04],
                "Agent": ["A", "B"],
            }
        ),
        {"run_name": "demo-run", "seed": 123},
    )

    assert "LLM configuration is required" in text
    assert trace_url is None
    assert payload["trace_url"] is None
    record = json.loads(fleet_path.read_text().splitlines()[-1])
    assert record["status"] == "no_secret"
    assert record["operation"] == "result-explanation"
    assert record["domain"]["run_id"] == "demo-run"
    assert record["domain"]["metric_delta"] == -0.015
    assert "artifact_ref" not in record["domain"]


def test_compare_runs_emits_fallback_record(monkeypatch, tmp_path) -> None:
    fleet_path = tmp_path / "fleet.ndjson"
    prior_output = tmp_path / "prior.xlsx"
    pd.DataFrame({"monthly_TE": [0.01], "monthly_CVaR": [-0.03]}).to_excel(
        prior_output, sheet_name="Summary", index=False
    )
    prior_manifest = tmp_path / "prior.json"
    prior_manifest.write_text(json.dumps({"seed": 1, "cli_args": {"output": str(prior_output)}}))
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setenv("PAEM_LANGSMITH_FLEET_PATH", str(fleet_path))
    monkeypatch.setattr(
        "pa_core.llm.compare_runs._invoke_comparison_llm",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("provider down")),
    )

    text, trace_url, payload = compare_runs(
        current_summary=pd.DataFrame({"monthly_TE": [0.04], "monthly_CVaR": [-0.02]}),
        current_manifest={"previous_run": str(prior_manifest), "seed": 2},
        questions="What changed?",
        provider_name="openai",
        model_name="gpt-test",
    )

    assert "Run-to-run comparison summary" in text
    assert trace_url is None
    assert payload.prior_manifest_path == str(prior_manifest)
    record = json.loads(fleet_path.read_text().splitlines()[-1])
    assert record["status"] == "no_secret"
    assert record["operation"] == "run-comparison"
    assert record["domain"]["metric_delta"] == 0.03
    assert record["domain"]["error_category"] == "RuntimeError"
