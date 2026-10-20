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
    empty = pd.DataFrame({"AnnReturn": []})
    assert normalize_summary_columns(empty) is empty

    summary = pd.DataFrame({"AnnReturn": [0.1], "monthly_AnnReturn": [0.2]})
    normalized = normalize_summary_columns(summary)
    assert normalized.columns.tolist() == ["terminal_AnnReturn", "monthly_AnnReturn"]
    assert normalize_summary_columns(normalized) is normalized

    # Already-prefixed target present: legacy name must not be renamed away.
    both = pd.DataFrame({"AnnReturn": [0.1], "terminal_AnnReturn": [0.2]})
    assert normalize_summary_columns(both) is both

    assert _coerce_agent_semantics_df(np.array([{"Agent": "A", "Weight": 1.0}])).equals(
        pd.DataFrame([{"Agent": "A", "Weight": 1.0}])
    )
    assert _coerce_agent_semantics_df(pd.DataFrame({"Agent": ["Z"]})).equals(
        pd.DataFrame({"Agent": ["Z"]})
    )
    series_frame = _coerce_agent_semantics_df(pd.Series({"Agent": "S", "Weight": 0.5}))
    assert series_frame is not None and series_frame.to_dict(orient="records") == [
        {"Agent": "S", "Weight": 0.5}
    ]
    concat_frames = _coerce_agent_semantics_df(
        [pd.DataFrame({"Agent": ["A"]}), pd.DataFrame({"Agent": ["B"]})]
    )
    assert concat_frames is not None and concat_frames["Agent"].tolist() == ["A", "B"]
    series_list = _coerce_agent_semantics_df(
        [pd.Series({"Agent": "C"}), pd.Series({"Agent": "D", "Weight": 2})]
    )
    assert series_list is not None and series_list["Agent"].tolist() == ["C", "D"]
    dict_list = _coerce_agent_semantics_df([{"Agent": "P"}, {"Agent": "Q", "Weight": 2}])
    assert dict_list is not None and dict_list["Agent"].tolist() == ["P", "Q"]
    mixed = _coerce_agent_semantics_df([pd.DataFrame({"Agent": ["A"]}), {"Agent": "B"}])
    assert mixed is not None and mixed["Agent"].tolist() == ["A", "B"]
    row_dict = _coerce_agent_semantics_df({"Agent": "R", "Weight": 3.0})
    assert row_dict is not None and row_dict.to_dict(orient="records") == [
        {"Agent": "R", "Weight": 3.0}
    ]
    columnar = _coerce_agent_semantics_df({"Agent": ["X", "Y"], "Weight": [1.0, 2.0]})
    assert columnar is not None and columnar["Agent"].tolist() == ["X", "Y"]
    assert _coerce_agent_semantics_df(42) is None
    assert _coerce_agent_semantics_df({"col_a": [1, 2], "col_b": [3]}) is None

    assert _to_builtin_scalar(np.float64(2.5)) == 2.5
    assert _to_builtin_scalar(float("nan")) is None
    assert _to_builtin_scalar({"x": np.array([np.float64(1), np.nan]), "y": (np.inf,)}) == {
        "x": [1.0, None],
        "y": [None],
    }
    assert _records_to_builtin([{"value": np.float64(3)}]) == [{"value": 3.0}]
    assert _serialize_metadata_value({"b": 1, "a": [2]}) == '{"a": [2], "b": 1}'
    assert _serialize_metadata_value("plain") == "plain"
    assert _serialize_metadata_value(("t", 1)) == '["t", 1]'

    inputs = {"_agent_semantics_df": [{"Agent": "A", "Weight": np.float64(1)}]}
    result = _ensure_agent_semantics_df(inputs)
    assert result is not None
    assert inputs["_agent_semantics_df"] == [{"Agent": "A", "Weight": 1.0}]

    frame_inputs = {
        "_agent_semantics_df": pd.DataFrame({"Agent": ["F"], "Weight": [np.float64(4)]})
    }
    frame_result = _ensure_agent_semantics_df(frame_inputs)
    assert frame_result is not None and frame_result["Agent"].tolist() == ["F"]
    assert frame_inputs["_agent_semantics_df"] == [{"Agent": "F", "Weight": 4.0}]

    mixed_list_inputs = {
        "_agent_semantics_df": [pd.DataFrame({"Agent": ["M"]}), {"Agent": "N", "Weight": 1}]
    }
    mixed_result = _ensure_agent_semantics_df(mixed_list_inputs)
    assert mixed_result is not None and mixed_result["Agent"].tolist() == ["M", "N"]
    assert mixed_list_inputs["_agent_semantics_df"] == [
        {"Agent": "M", "Weight": None},
        {"Agent": "N", "Weight": 1.0},
    ]

    empty_inputs: dict = {"_agent_semantics_df": []}
    empty_result = _ensure_agent_semantics_df(empty_inputs)
    assert isinstance(empty_result, pd.DataFrame) and empty_result.empty
    assert empty_inputs["_agent_semantics_df"] == []


def test_excel_builds_agent_semantics_from_agents_when_frame_missing() -> None:
    agents = [
        {
            "name": "Base",
            "capital": 100.0,
            "beta_share": 1.0,
            "alpha_share": 0.0,
            "extra": {},
        }
    ]
    inputs = {
        "agents": agents,
        "total_fund_capital": 100.0,
        "reference_sigma": 0.15,
        "volatility_multiple": 1.0,
        "financing_model": "flat",
    }
    df = _ensure_agent_semantics_df(inputs)
    assert df is not None
    assert "Agent" in df.columns
    assert inputs["_agent_semantics_df"][0]["Agent"] == "Base"


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
