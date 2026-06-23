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
        "No results yet - complete a run in the Scenario Wizard, Scenario Grid, "
        "or Stress Lab to generate output."
    )
    assert "Outputs.xlsx" not in RESULTS_EMPTY_STATE_MESSAGE
    assert "File " not in RESULTS_EMPTY_STATE_MESSAGE


def test_run_logs_empty_state_hides_cli_flag() -> None:
    assert RUN_LOGS_EMPTY_STATE_MESSAGE == (
        "No runs yet - run a scenario and its history will appear here."
    )
    assert "--log-json" not in RUN_LOGS_EMPTY_STATE_MESSAGE


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
