"""Focused low-level coverage for the first two targets of issue #2415."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pa_core.reporting.excel import (
    _coerce_agent_semantics_df,
    _ensure_agent_semantics_df,
    _records_to_builtin,
    _serialize_metadata_value,
    _to_builtin_scalar,
    normalize_summary_columns,
)
from pa_core.sim.paths import _validate_correlation_matrix, _validate_covariance_matrix


def test_excel_normalizes_and_serializes_nested_agent_semantics() -> None:
    summary = pd.DataFrame({"AnnReturn": [0.1], "monthly_AnnReturn": [0.2]})
    normalized = normalize_summary_columns(summary)
    assert normalized.columns.tolist() == ["terminal_AnnReturn", "monthly_AnnReturn"]
    assert normalize_summary_columns(normalized) is normalized

    assert _coerce_agent_semantics_df(np.array([{"Agent": "A", "Weight": 1.0}])).equals(
        pd.DataFrame([{"Agent": "A", "Weight": 1.0}])
    )
    mixed = _coerce_agent_semantics_df([pd.DataFrame({"Agent": ["A"]}), {"Agent": "B"}])
    assert mixed is not None and mixed["Agent"].tolist() == ["A", "B"]
    assert _to_builtin_scalar({"x": np.array([np.float64(1), np.nan]), "y": (np.inf,)}) == {
        "x": [1.0, None],
        "y": [None],
    }
    assert _records_to_builtin([{"value": np.float64(3)}]) == [{"value": 3.0}]
    assert _serialize_metadata_value({"b": 1, "a": [2]}) == '{"a": [2], "b": 1}'

    inputs = {"_agent_semantics_df": [{"Agent": "A", "Weight": np.float64(1)}]}
    result = _ensure_agent_semantics_df(inputs)
    assert result is not None
    assert inputs["_agent_semantics_df"] == [{"Agent": "A", "Weight": 1.0}]


def test_paths_matrix_validation_accepts_valid_and_rejects_invalid_inputs() -> None:
    correlation = np.eye(2)
    _validate_correlation_matrix(correlation)
    assert np.array_equal(_validate_covariance_matrix(correlation), correlation)

    with pytest.raises(ValueError, match="non-finite"):
        _validate_correlation_matrix(np.array([[1.0, np.nan], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="within"):
        _validate_correlation_matrix(np.array([[1.0, 1.1], [1.1, 1.0]]))
    with pytest.raises(ValueError, match="diagonal"):
        _validate_correlation_matrix(np.array([[0.9, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="symmetric"):
        _validate_covariance_matrix(np.array([[1.0, 0.1], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="variances"):
        _validate_covariance_matrix(np.array([[-1.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="zero variance"):
        _validate_covariance_matrix(np.array([[0.0, 0.1], [0.1, 1.0]]))
