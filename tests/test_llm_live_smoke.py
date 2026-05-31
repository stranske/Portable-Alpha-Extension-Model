"""Opt-in live LLM smoke tests for dashboard LLM boundaries."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pytest

from pa_core.llm.compare_runs import _default_provider_api_key, compare_runs
from pa_core.llm.provider import LLMProviderConfig
from pa_core.llm.result_explain import EXPLAIN_RESULTS_DISCLAIMER, explain_results_details


def test_live_llm_smoke_tests_are_opt_in_by_default() -> None:
    """Keep direct file-only pytest invocations from failing with exit code 5."""

    assert True


def _provider_name() -> str:
    return os.environ.get("PA_LLM_PROVIDER", "openai").strip().lower() or "openai"


def _provider_model(provider_name: str) -> str:
    model = os.environ.get("PA_LLM_MODEL")
    if model:
        return model
    if provider_name == "anthropic":
        return "claude-sonnet-4-20250514"
    return "gpt-4o-mini"


def _provider_credentials(provider_name: str, api_key: str) -> dict[str, str]:
    credentials = {"api_key": api_key}
    if provider_name == "azure_openai":
        endpoint = os.environ.get("PA_LLM_BASE_URL", "").strip()
        api_version = (
            os.environ.get("PA_LLM_API_VERSION") or os.environ.get("AZURE_OPENAI_API_VERSION") or ""
        ).strip()
        missing = [
            name
            for name, value in (
                ("PA_LLM_BASE_URL", endpoint),
                ("PA_LLM_API_VERSION or AZURE_OPENAI_API_VERSION", api_version),
            )
            if not value
        ]
        if missing:
            pytest.skip(f"missing Azure OpenAI live verification env vars: {', '.join(missing)}")
        credentials["azure_endpoint"] = endpoint
        credentials["api_version"] = api_version
    return credentials


@pytest.fixture
def live_llm_config() -> tuple[LLMProviderConfig, str]:
    provider_name = _provider_name()
    if os.environ.get("PA_LLM_LIVE") != "1":
        pytest.skip("set PA_LLM_LIVE=1 to run live provider smoke tests")

    api_key = _default_provider_api_key(provider_name)
    if not api_key:
        pytest.skip(f"missing provider key for PA_LLM_PROVIDER={provider_name!r}")

    config = LLMProviderConfig(
        provider_name=provider_name,
        credentials=_provider_credentials(provider_name, api_key),
        model_name=_provider_model(provider_name),
    )
    return config, api_key


def _string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        strings: list[str] = []
        for child in value.values():
            strings.extend(_string_values(child))
        return strings
    if isinstance(value, (list, tuple, set)):
        strings = []
        for child in value:
            strings.extend(_string_values(child))
        return strings
    return []


def _assert_secret_absent(api_key: str, *values: Any) -> None:
    for value in values:
        for text in _string_values(value):
            assert api_key not in text


def _summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Agent": ["Core", "Extension", "Total"],
            "monthly_TE": [0.018, 0.021, 0.02],
            "monthly_CVaR": [-0.041, -0.048, -0.044],
            "monthly_BreachProb": [0.08, 0.11, 0.095],
            "stress_delta_te": [0.002, 0.003, 0.0025],
        }
    )


@pytest.mark.live_llm
def test_explain_results_live(live_llm_config: tuple[LLMProviderConfig, str]) -> None:
    config, api_key = live_llm_config

    text, trace_url, payload = explain_results_details(
        _summary_frame(),
        manifest=None,
        questions="Summarize the most important risk movement in one concise paragraph.",
        llm_config=config,
        tracing_enabled=False,
    )

    assert text.strip()
    assert text.endswith(EXPLAIN_RESULTS_DISCLAIMER)
    assert "Failed to generate explanation" not in text
    _assert_secret_absent(api_key, text, trace_url, payload)


def _run_cli(output_path: Path, *, previous_manifest: Path | None = None) -> Path:
    cmd = [
        sys.executable,
        "-m",
        "pa_core.cli",
        "--config",
        "config/params_template.yml",
        "--index",
        "data/sp500tr_fred_divyield.csv",
        "--output",
        str(output_path),
    ]
    if previous_manifest is not None:
        cmd.extend(["--prev-manifest", str(previous_manifest)])

    subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    manifest_path = output_path.with_name("manifest.json")
    assert manifest_path.exists()
    return manifest_path


@pytest.mark.live_llm
def test_compare_runs_live(tmp_path: Path, live_llm_config: tuple[LLMProviderConfig, str]) -> None:
    config, api_key = live_llm_config
    prior_output = tmp_path / "prior" / "prior.xlsx"
    current_output = tmp_path / "current" / "current.xlsx"
    prior_output.parent.mkdir()
    current_output.parent.mkdir()

    prior_manifest_path = _run_cli(prior_output)
    current_manifest_path = _run_cli(current_output, previous_manifest=prior_manifest_path)

    current_manifest = json.loads(current_manifest_path.read_text())
    current_summary = pd.read_excel(current_output, sheet_name="Summary")

    text, trace_url, payload = compare_runs(
        current_summary=current_summary,
        current_manifest=current_manifest,
        questions="What changed between these two runs?",
        provider_config=config,
    )

    assert text.strip()
    assert not text.startswith("Run-to-run comparison summary:")
    _assert_secret_absent(api_key, text, trace_url, asdict(payload))
