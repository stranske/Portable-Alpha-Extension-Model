"""Result explanation entrypoint for dashboard Results details."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd


def _manifest_highlights(manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    if manifest is None:
        return {}
    highlights: dict[str, Any] = {}
    seed = manifest.get("seed")
    if seed is not None:
        highlights["seed"] = seed
    cli_args = manifest.get("cli_args")
    if isinstance(cli_args, Mapping):
        # Keep this compact and avoid leaking any key-like values.
        for key in ("output", "benchmark", "config", "n_sims", "horizon_months"):
            value = cli_args.get(key)
            if value is not None:
                highlights[f"cli_{key}"] = value
    return highlights


def explain_results_details(
    details_df: pd.DataFrame,
    manifest: Mapping[str, Any] | None = None,
) -> tuple[str, str | None, dict[str, Any]]:
    """Return an explain-results payload for a summary/details dataframe.

    Parameters
    ----------
    details_df
        Results details table (typically the Summary sheet).
    manifest
        Optional run manifest payload.

    Returns
    -------
    tuple[str, str | None, dict[str, Any]]
        ``(text, trace_url, payload_dict)``.
    """

    if not isinstance(details_df, pd.DataFrame):
        raise TypeError("details_df must be a pandas DataFrame")

    numeric_cols = details_df.select_dtypes(include="number")
    mean_stats: dict[str, float] = {}
    if not numeric_cols.empty:
        means = numeric_cols.mean(numeric_only=True)
        mean_stats = {str(col): float(val) for col, val in means.items()}

    payload: dict[str, Any] = {
        "rows": int(details_df.shape[0]),
        "columns": [str(col) for col in details_df.columns.tolist()],
        "mean_stats": mean_stats,
        "manifest_highlights": _manifest_highlights(manifest),
    }
    text = (
        "Result explanation is ready. "
        f"Processed {payload['rows']} rows across {len(payload['columns'])} columns."
    )
    trace_url = None
    return text, trace_url, payload
