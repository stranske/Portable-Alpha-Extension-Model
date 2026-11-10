"""Facade branch coverage for issue #2476 remaining exceptions.

Complements tests/test_coverage_round_2381.py (kept at its recorded 52 cases)
by covering the _serialize_agent_semantics_input container branches and the
defensive export branch in pa_core/facade.py that the gate round left as
documented exceptions. See docs/evidence/issue-2476/.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from pa_core import facade


def _serialize(value: Any) -> Any:
    inputs: dict[str, Any] = {"_agent_semantics_df": value}
    facade._serialize_agent_semantics_input(inputs)
    return inputs["_agent_semantics_df"]


def test_serialize_agent_semantics_from_series() -> None:
    result = _serialize(pd.Series({"Agent": "Base", "n": 2}))
    assert result == [{"Agent": "Base", "n": 2}]


def test_serialize_agent_semantics_from_dataframe_list() -> None:
    frames = [pd.DataFrame([{"Agent": "A"}]), pd.DataFrame([{"Agent": "B"}])]
    assert _serialize(frames) == [{"Agent": "A"}, {"Agent": "B"}]


def test_serialize_agent_semantics_from_series_and_dict_list() -> None:
    result = _serialize([pd.Series({"Agent": "A"}), {"Agent": "B"}])
    assert result == [{"Agent": "A"}, {"Agent": "B"}]


def test_serialize_agent_semantics_from_mixed_frame_and_series_list() -> None:
    result = _serialize([pd.DataFrame([{"Agent": "A"}]), pd.Series({"Agent": "B"})])
    assert result == [{"Agent": "A"}, {"Agent": "B"}]


def test_serialize_agent_semantics_from_scalar_tuple_and_list() -> None:
    from_tuple = _serialize((np.int64(1), np.float64(np.nan)))
    assert from_tuple == [1, None]
    assert type(from_tuple[0]) is int
    from_list = _serialize([np.int64(2), float("inf")])
    assert from_list == [2, None]
    assert type(from_list[0]) is int
    assert _serialize([]) == []


def test_serialize_agent_semantics_from_ndarray() -> None:
    assert _serialize(np.array([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]


def test_serialize_agent_semantics_from_column_mapping() -> None:
    result = _serialize({"Agent": ["A", "B"], "n": [1, 2], "w": [0.5, np.nan]})
    assert result == [{"Agent": "A", "n": 1, "w": 0.5}, {"Agent": "B", "n": 2, "w": None}]
    assert type(result[0]["n"]) is int


def test_serialize_agent_semantics_keeps_mapping_when_frame_build_fails() -> None:
    bad = {"a": [1, 2], "b": [1]}
    assert _serialize(bad) is bad


def test_serialize_agent_semantics_empty_mapping_becomes_empty_records() -> None:
    assert _serialize({}) == []


def test_to_builtin_scalar_converts_numpy_and_non_finite_values() -> None:
    assert facade._to_builtin_scalar(np.int64(3)) == 3
    assert type(facade._to_builtin_scalar(np.int64(3))) is int
    assert facade._to_builtin_scalar(float("nan")) is None
    assert facade._to_builtin_scalar(np.array([1.0, np.nan])) == [1.0, None]
    assert facade._to_builtin_scalar((np.int64(1), "x")) == [1, "x"]
    assert facade._to_builtin_scalar({"k": np.float64(0.5)}) == {"k": 0.5}


def test_export_rejects_unsupported_artifacts_type(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unsupported artifacts type"):
        facade.export("not-artifacts", tmp_path / "out.xlsx")  # type: ignore[arg-type]
