"""Helpers for building and formatting stress delta tables."""

from __future__ import annotations

import io

import pandas as pd
import pandas.api.types as pdt

_PCT_COLUMNS = {
    "AnnReturn",
    "AnnVol",
    "VaR",
    "CVaR",
    "MaxDD",
    "TimeUnderWater",
    "BreachProb",
    "ShortfallProb",
    "TE",
}


def build_delta_table(
    base_summary: pd.DataFrame, stress_summary: pd.DataFrame
) -> pd.DataFrame:
    """Return a delta table (stressed - base) with a Total row."""
    if "Agent" not in base_summary.columns or "Agent" not in stress_summary.columns:
        return pd.DataFrame()

    common_cols = [
        col
        for col in stress_summary.columns
        if col in base_summary.columns and col not in {"Agent"}
    ]
    numeric_cols = [
        col
        for col in common_cols
        if pdt.is_numeric_dtype(stress_summary[col])
        and pdt.is_numeric_dtype(base_summary[col])
    ]
    merged = stress_summary.merge(base_summary, on="Agent", suffixes=("_S", "_B"))
    deltas: dict[str, pd.Series] = {"Agent": merged["Agent"]}
    for col in numeric_cols:
        deltas[col] = merged[f"{col}_S"] - merged[f"{col}_B"]
    delta_df = pd.DataFrame(deltas)

    if not delta_df.empty:
        delta_df["Agent"] = delta_df["Agent"].replace({"Base": "Total"})
        total_rows = delta_df[delta_df["Agent"] == "Total"]
        if not total_rows.empty:
            delta_df = pd.concat(
                [delta_df[delta_df["Agent"] != "Total"], total_rows],
                ignore_index=True,
            )
    return delta_df


def format_delta_table(delta_df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Format deltas for display with clear +/- and percent columns."""
    formatters: dict[str, object] = {}
    for col in delta_df.columns:
        if col == "Agent":
            continue
        if col in _PCT_COLUMNS:
            formatters[col] = lambda x: f"{x:+.2%}" if pd.notna(x) else ""
        else:
            formatters[col] = lambda x: f"{x:+.4f}" if pd.notna(x) else ""
    return delta_df.style.format(formatters)


def format_delta_table_text(delta_df: pd.DataFrame) -> pd.DataFrame:
    """Return deltas formatted as strings for PPTX/table exports."""
    formatted = delta_df.copy()
    for col in formatted.columns:
        if col == "Agent":
            continue
        if not pdt.is_numeric_dtype(formatted[col]):
            continue
        if col in _PCT_COLUMNS:
            formatted[col] = formatted[col].apply(
                lambda x: f"{x:+.2%}" if pd.notna(x) else ""
            )
        else:
            formatted[col] = formatted[col].apply(
                lambda x: f"{x:+.4f}" if pd.notna(x) else ""
            )
    return formatted


def build_stress_workbook(
    base_summary: pd.DataFrame,
    stress_summary: pd.DataFrame,
    delta_df: pd.DataFrame,
    config_diff_df: pd.DataFrame | None = None,
) -> bytes:
    """Return an Excel workbook with base, stressed, and delta summaries."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        base_summary.to_excel(writer, sheet_name="BaseSummary", index=False)
        stress_summary.to_excel(writer, sheet_name="StressedSummary", index=False)
        if not delta_df.empty:
            delta_df.to_excel(writer, sheet_name="Delta", index=False)
        if config_diff_df is not None and not config_diff_df.empty:
            config_diff_df.to_excel(writer, sheet_name="ConfigDiff", index=False)
    buffer.seek(0)
    return buffer.read()
