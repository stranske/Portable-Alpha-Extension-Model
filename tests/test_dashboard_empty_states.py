from __future__ import annotations

import pytest

from dashboard.utils import (
    RESULTS_EMPTY_STATE_MESSAGE,
    RUN_LOGS_EMPTY_STATE_MESSAGE,
    friendly_validation_error_message,
)
from pa_core.config import ModelConfig, load_config


def test_results_empty_state_uses_operator_guidance() -> None:
    assert RESULTS_EMPTY_STATE_MESSAGE == (
        "No results yet - complete a run in the Scenario Wizard, then return here "
        "to inspect the generated workbook."
    )
    assert "Outputs.xlsx" not in RESULTS_EMPTY_STATE_MESSAGE
    assert "File " not in RESULTS_EMPTY_STATE_MESSAGE
    assert "Scenario Grid" not in RESULTS_EMPTY_STATE_MESSAGE
    assert "Stress Lab" not in RESULTS_EMPTY_STATE_MESSAGE


def test_run_logs_empty_state_mentions_json_logging_without_cli_flag() -> None:
    assert RUN_LOGS_EMPTY_STATE_MESSAGE == (
        "No run logs yet - enable JSON logging for a run, then return here to inspect its history."
    )
    assert "--log-json" not in RUN_LOGS_EMPTY_STATE_MESSAGE
    assert "JSON logging" in RUN_LOGS_EMPTY_STATE_MESSAGE


def test_validation_error_translator_uses_human_field_label() -> None:
    with pytest.raises(Exception) as exc_info:
        ModelConfig.model_validate({"Number of simulations": 1, "Number of months": 1})

    message = friendly_validation_error_message(exc_info.value)

    assert "Financing draw mode" in message
    assert "financing_mode" not in message
    assert "pydantic.dev" not in message


def test_validation_error_translator_handles_load_config_wrapped_errors() -> None:
    with pytest.raises(ValueError) as exc_info:
        load_config({"Number of simulations": 1, "Number of months": 1})

    message = friendly_validation_error_message(exc_info.value)

    assert "Financing draw mode" in message
    assert "financing_mode" not in message
    assert "pydantic.dev" not in message
