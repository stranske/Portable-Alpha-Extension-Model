from __future__ import annotations

from typing import Any, Mapping, Tuple

import pandas as pd
import pandas.api.types as pdt


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

    cfg_cur = current_manifest.get("config", {}) if current_manifest else {}
    cfg_prev = previous_manifest.get("config", {}) if previous_manifest else {}

    config_records: list[dict[str, Any]] = []
    for key in sorted(set(cfg_cur) | set(cfg_prev)):
        cur_val = cfg_cur.get(key)
        prev_val = cfg_prev.get(key)
        if cur_val != prev_val:
            config_records.append(
                {"Parameter": key, "Current": cur_val, "Previous": prev_val}
            )
    cfg_diff_df = pd.DataFrame(config_records)

    metric_records: list[dict[str, Any]] = []
    if not current_summary.empty and not previous_summary.empty:
        common = [c for c in current_summary.columns if c in previous_summary.columns]
        for col in common:
            if pdt.is_numeric_dtype(current_summary[col]) and pdt.is_numeric_dtype(
                previous_summary[col]
            ):
                cur_val = current_summary[col].iloc[0]
                prev_val = previous_summary[col].iloc[0]
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
