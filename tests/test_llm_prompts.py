"""Basic prompt-builder tests for pa_core.llm.prompts."""

from __future__ import annotations

from pa_core.llm.prompts import (
    build_comparison_prompt,
    build_config_wizard_prompt,
    build_result_explanation_prompt,
)


def test_build_result_explanation_prompt_returns_nonempty_string():
    result = build_result_explanation_prompt({"sharpe": 1.24, "max_drawdown": -0.12})
    assert isinstance(result, str)
    assert len(result) > 0


def test_build_comparison_prompt_returns_nonempty_string():
    result = build_comparison_prompt({"name": "A", "cagr": 0.11}, {"name": "B", "cagr": 0.09})
    assert isinstance(result, str)
    assert len(result) > 0


def test_build_config_wizard_prompt_returns_nonempty_string():
    result = build_config_wizard_prompt({"objective": "maximize_sharpe", "horizon_years": 5})
    assert isinstance(result, str)
    assert len(result) > 0
