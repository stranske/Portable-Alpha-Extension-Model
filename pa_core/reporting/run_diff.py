from __future__ import annotations

from numbers import Real
from typing import Any, Mapping, Tuple, TypeGuard

import pandas as pd
import pandas.api.types as pdt

_HEADLINE_METRICS = (
    "AnnReturn",
    "AnnVol",
    "VaR",
    "CVaR",
    "MaxDD",
    "ShortfallProb",
    "BreachProb",
    "TE",
)
_ID_COLUMNS = ("Agent", "Combination", "Label")


def build_run_diff(
    current_manifest: Mapping[str, Any] | None,
    previous_manifest: Mapping[str, Any] | None,
    current_summary: pd.DataFrame,
    previous_summary: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build config and metric diffs between runs.

    Parameters
    ----------
    current_manifest, previous_manifest:
        Mapping objects representing manifests for current and previous runs.
    current_summary, previous_summary:
        Summary DataFrames for the runs.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        DataFrames representing config diffs and metric deltas.
    """

    def _is_numeric(value: Any) -> TypeGuard[Real]:
        return isinstance(value, Real) and not isinstance(value, bool)

    def _coerce_numeric(value: Any) -> float | None:
        try:
            num = pd.to_numeric(value, errors="coerce")
        except Exception:
            return None
        if pd.isna(num):
            return None
        return float(num)

    def _series_has_numeric(series: pd.Series) -> bool:
        try:
            numeric = pd.to_numeric(series, errors="coerce")
        except Exception:
            return False
        return numeric.notna().any()

    cfg_cur = current_manifest.get("config", {}) if current_manifest else {}
    cfg_prev = previous_manifest.get("config", {}) if previous_manifest else {}

    config_records: list[dict[str, Any]] = []
    for key in sorted(set(cfg_cur) | set(cfg_prev)):
        cur_val = cfg_cur.get(key)
        prev_val = cfg_prev.get(key)
        if cur_val != prev_val:
            delta: float | str = ""
            if _is_numeric(cur_val) and _is_numeric(prev_val):
                try:
                    delta = float(cur_val) - float(prev_val)
                except (TypeError, ValueError):
                    delta = ""
            config_records.append(
                {
                    "Parameter": key,
                    "Current": cur_val,
                    "Previous": prev_val,
                    "Delta": delta,
                }
            )
    cfg_diff_df = pd.DataFrame(config_records)

    metric_records: list[dict[str, Any]] = []
    if not current_summary.empty and not previous_summary.empty:
        id_cols = [
            col
            for col in _ID_COLUMNS
            if col in current_summary.columns and col in previous_summary.columns
        ]

        common = [
            c
            for c in current_summary.columns
            if c in previous_summary.columns and c not in id_cols
        ]
        numeric = [
            c
            for c in common
            if (
                pdt.is_numeric_dtype(current_summary[c])
                or _series_has_numeric(current_summary[c])
            )
            and (
                pdt.is_numeric_dtype(previous_summary[c])
                or _series_has_numeric(previous_summary[c])
            )
        ]
        metrics = [c for c in _HEADLINE_METRICS if c in numeric] or numeric

        common_ids: list[tuple[Any, ...]] = []
        current_keys = None
        if id_cols:
            current_keys = current_summary[id_cols].apply(tuple, axis=1)
            prev_keys = previous_summary[id_cols].apply(tuple, axis=1)
            current_ids = list(dict.fromkeys(current_keys.tolist()))
            prev_ids = set(prev_keys.tolist())
            common_ids = [key for key in current_ids if key in prev_ids]

        if id_cols and common_ids and current_keys is not None:
            prev_map = {}
            for key, row in zip(prev_keys.tolist(), previous_summary.itertuples()):
                if key not in prev_map:
                    prev_map[key] = row
            for key in common_ids:
                mask = current_keys.apply(lambda x, k=key: x == k)
                cur_row = current_summary[mask].iloc[0]
                prev_row = prev_map.get(key)
                if prev_row is None:
                    continue
                for col in metrics:
                    cur_val = _coerce_numeric(cur_row[col])
                    prev_val = _coerce_numeric(getattr(prev_row, col))
                    if cur_val is None or prev_val is None:
                        continue
                    delta = cur_val - prev_val
                    record = {
                        "Metric": col,
                        "Current": cur_val,
                        "Previous": prev_val,
                        "Delta": delta,
                    }
                    for idx, id_col in enumerate(id_cols):
                        record[id_col] = key[idx]
                    metric_records.append(record)
        else:
            for col in metrics:
                cur_val = _coerce_numeric(current_summary[col].iloc[0])
                prev_val = _coerce_numeric(previous_summary[col].iloc[0])
                if cur_val is None or prev_val is None:
                    continue
                delta = cur_val - prev_val
                metric_records.append(
                    {
                        "Metric": col,
                        "Current": cur_val,
                        "Previous": prev_val,
                        "Delta": delta,
                    }
                )
    metric_diff_df = pd.DataFrame(metric_records)
    return cfg_diff_df, metric_diff_df
