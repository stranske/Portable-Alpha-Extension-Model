from __future__ import annotations

from typing import Any, Mapping, Tuple

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

    def _is_numeric(value: Any) -> bool:
        return pdt.is_number(value) and not isinstance(value, bool)

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
        id_col = next(
            (col for col in _ID_COLUMNS if col in current_summary.columns),
            None,
        )
        if id_col and id_col not in previous_summary.columns:
            id_col = None

        common = [
            c
            for c in current_summary.columns
            if c in previous_summary.columns and c != id_col
        ]
        numeric = [
            c
            for c in common
            if pdt.is_numeric_dtype(current_summary[c])
            and pdt.is_numeric_dtype(previous_summary[c])
        ]
        metrics = [c for c in _HEADLINE_METRICS if c in numeric] or numeric

        common_ids = []
        if id_col:
            current_ids = list(dict.fromkeys(current_summary[id_col].tolist()))
            prev_ids = set(previous_summary[id_col].tolist())
            common_ids = [v for v in current_ids if v in prev_ids]

        if id_col and common_ids:
            for ident in common_ids:
                cur_row = current_summary[current_summary[id_col] == ident].iloc[0]
                prev_row = previous_summary[previous_summary[id_col] == ident].iloc[0]
                for col in metrics:
                    try:
                        cur_val = float(cur_row[col])
                        prev_val = float(prev_row[col])
                        delta = cur_val - prev_val
                    except (TypeError, ValueError):
                        continue
                    metric_records.append(
                        {
                            id_col: ident,
                            "Metric": col,
                            "Current": cur_val,
                            "Previous": prev_val,
                            "Delta": delta,
                        }
                    )
        else:
            for col in metrics:
                try:
                    cur_val = float(current_summary[col].iloc[0])
                    prev_val = float(previous_summary[col].iloc[0])
                    delta = cur_val - prev_val
                except (TypeError, ValueError):
                    continue
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
