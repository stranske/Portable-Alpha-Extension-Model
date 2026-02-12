"""Basic prompt-builder tests for pa_core.llm.prompts."""

from __future__ import annotations

import pytest

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


def test_build_comparison_prompt_contains_both_serialized_items():
    result = build_comparison_prompt(
        {"zeta": 1, "alpha": 2},
        {"scenario": "stress", "cvar": -0.08},
    )

    assert "Item A:" in result
    assert "Item B:" in result
    # json.dumps(..., sort_keys=True) should order keys alphabetically.
    assert '"alpha": 2' in result
    assert '"zeta": 1' in result
    assert result.index('"alpha": 2') < result.index('"zeta": 1')


# ---------- Negative / validation tests ----------


def test_build_result_explanation_prompt_rejects_none():
    with pytest.raises(ValueError, match="result_data is required"):
        build_result_explanation_prompt(None)


def test_build_result_explanation_prompt_rejects_empty_string():
    with pytest.raises(ValueError, match="result_data is required"):
        build_result_explanation_prompt("")


def test_build_comparison_prompt_rejects_none_item_a():
    with pytest.raises(ValueError, match="item_a is required"):
        build_comparison_prompt(None, {"x": 1})


def test_build_comparison_prompt_rejects_none_item_b():
    with pytest.raises(ValueError, match="item_b is required"):
        build_comparison_prompt({"x": 1}, None)


def test_build_config_wizard_prompt_rejects_none():
    with pytest.raises(ValueError, match="config_context is required"):
        build_config_wizard_prompt(None)


def test_build_config_wizard_prompt_rejects_empty_string():
    with pytest.raises(ValueError, match="config_context is required"):
        build_config_wizard_prompt("")
