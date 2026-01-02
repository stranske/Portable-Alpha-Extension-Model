from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Pattern, Sequence, TypedDict

import pandas as pd

# Run directory layout (created when --log-json is enabled)
RUNS_DIR_NAME = "runs"
RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z$")
RUN_LOG_FILENAME = "run.log"
RUN_END_FILENAME = "run_end.json"
RUN_END_MANIFEST_PATH_KEY = "manifest_path"
RUN_DIRECTORY_REQUIRED_FILES: Sequence[str] = (RUN_LOG_FILENAME,)
RUN_DIRECTORY_OPTIONAL_FILES: Sequence[str] = (RUN_END_FILENAME,)


# Explicit run directory contract for dashboard and validation usage.
@dataclass(frozen=True)
class RunDirectoryContract:
    runs_dir_name: str
    run_id_pattern: Pattern[str]
    required_files: Sequence[str]
    optional_files: Sequence[str]


RUN_DIRECTORY_CONTRACT = RunDirectoryContract(
    runs_dir_name=RUNS_DIR_NAME,
    run_id_pattern=RUN_ID_PATTERN,
    required_files=RUN_DIRECTORY_REQUIRED_FILES,
    optional_files=RUN_DIRECTORY_OPTIONAL_FILES,
)

# Manifest contract
MANIFEST_FORMAT = "json"
MANIFEST_FILENAME = "manifest.json"
MANIFEST_REQUIRED_FIELDS: Sequence[str] = (
    "git_commit",
    "timestamp",
    "seed",
    "config",
    "data_files",
    "cli_args",
)
MANIFEST_OPTIONAL_FIELDS: Sequence[str] = (
    "backend",
    "run_log",
    "previous_run",
    "run_timing",
    "substream_ids",
)


@dataclass(frozen=True)
class ManifestContract:
    filename: str
    format: str
    run_end_manifest_key: str
    required_fields: Sequence[str]
    optional_fields: Sequence[str]


MANIFEST_CONTRACT = ManifestContract(
    filename=MANIFEST_FILENAME,
    format=MANIFEST_FORMAT,
    run_end_manifest_key=RUN_END_MANIFEST_PATH_KEY,
    required_fields=MANIFEST_REQUIRED_FIELDS,
    optional_fields=MANIFEST_OPTIONAL_FIELDS,
)

# Summary contract
SUMMARY_SHEET_NAME = "Summary"
ALL_RETURNS_SHEET_NAME = "AllReturns"
DEFAULT_OUTPUT_FILENAME = "Outputs.xlsx"

SUMMARY_AGENT_COLUMN = "Agent"
SUMMARY_ANN_RETURN_COLUMN = "AnnReturn"
SUMMARY_EXCESS_RETURN_COLUMN = "ExcessReturn"
SUMMARY_ANN_VOL_COLUMN = "AnnVol"
SUMMARY_VAR_COLUMN = "VaR"
SUMMARY_TE_COLUMN = "TE"
SUMMARY_TRACKING_ERROR_LEGACY_COLUMN = "TrackingErr"
SUMMARY_CVAR_COLUMN = "CVaR"
SUMMARY_BREACH_PROB_COLUMN = "BreachProb"
SUMMARY_MAX_DD_COLUMN = "MaxDD"
SUMMARY_TIME_UNDER_WATER_COLUMN = "TimeUnderWater"
SUMMARY_BREACH_COUNT_COLUMN = "BreachCount"
SUMMARY_SHORTFALL_PROB_COLUMN = "ShortfallProb"

SUMMARY_REQUIRED_COLUMNS: Sequence[str] = (
    SUMMARY_AGENT_COLUMN,
    SUMMARY_ANN_RETURN_COLUMN,
    SUMMARY_EXCESS_RETURN_COLUMN,
    SUMMARY_ANN_VOL_COLUMN,
    SUMMARY_VAR_COLUMN,
    SUMMARY_CVAR_COLUMN,
    SUMMARY_MAX_DD_COLUMN,
    SUMMARY_TIME_UNDER_WATER_COLUMN,
    SUMMARY_BREACH_PROB_COLUMN,
    SUMMARY_BREACH_COUNT_COLUMN,
    SUMMARY_SHORTFALL_PROB_COLUMN,
    SUMMARY_TE_COLUMN,
)
SUMMARY_NUMERIC_COLUMNS: Sequence[str] = tuple(
    col for col in SUMMARY_REQUIRED_COLUMNS if col != SUMMARY_AGENT_COLUMN
)
SUMMARY_COLUMN_TYPES: Mapping[str, str] = {
    SUMMARY_AGENT_COLUMN: "string",
    SUMMARY_ANN_RETURN_COLUMN: "number",
    SUMMARY_EXCESS_RETURN_COLUMN: "number",
    SUMMARY_ANN_VOL_COLUMN: "number",
    SUMMARY_VAR_COLUMN: "number",
    SUMMARY_CVAR_COLUMN: "number",
    SUMMARY_MAX_DD_COLUMN: "number",
    SUMMARY_TIME_UNDER_WATER_COLUMN: "number",
    SUMMARY_BREACH_PROB_COLUMN: "number",
    SUMMARY_BREACH_COUNT_COLUMN: "number",
    SUMMARY_SHORTFALL_PROB_COLUMN: "number",
    SUMMARY_TE_COLUMN: "number",
}


class RunDirectoryPaths(TypedDict):
    run_id: str
    path: Path
    log_path: Path
    run_end_path: Path


class ManifestPayload(TypedDict, total=False):
    git_commit: str
    timestamp: str
    seed: int | None
    config: Mapping[str, Any]
    data_files: Mapping[str, str]
    cli_args: Mapping[str, Any]
    backend: str | None
    run_log: str | None
    previous_run: str | None
    run_timing: Mapping[str, Any] | None


@dataclass(frozen=True)
class RunDirectoryLayout:
    run_id: str
    path: Path
    log_path: Path
    run_end_path: Path


@dataclass(frozen=True)
class SummaryContract:
    sheet_name: str
    all_returns_sheet_name: str
    default_output_filename: str
    required_columns: Sequence[str]
    numeric_columns: Sequence[str]
    column_types: Mapping[str, str]
    agent_column: str
    te_column: str
    tracking_error_legacy_column: str
    cvar_column: str
    breach_prob_column: str


SUMMARY_CONTRACT = SummaryContract(
    sheet_name=SUMMARY_SHEET_NAME,
    all_returns_sheet_name=ALL_RETURNS_SHEET_NAME,
    default_output_filename=DEFAULT_OUTPUT_FILENAME,
    required_columns=SUMMARY_REQUIRED_COLUMNS,
    numeric_columns=SUMMARY_NUMERIC_COLUMNS,
    column_types=SUMMARY_COLUMN_TYPES,
    agent_column=SUMMARY_AGENT_COLUMN,
    te_column=SUMMARY_TE_COLUMN,
    tracking_error_legacy_column=SUMMARY_TRACKING_ERROR_LEGACY_COLUMN,
    cvar_column=SUMMARY_CVAR_COLUMN,
    breach_prob_column=SUMMARY_BREACH_PROB_COLUMN,
)


def run_directory_layout(path: str | Path) -> RunDirectoryLayout:
    run_path = Path(path)
    return RunDirectoryLayout(
        run_id=run_path.name,
        path=run_path,
        log_path=run_path / RUN_LOG_FILENAME,
        run_end_path=run_path / RUN_END_FILENAME,
    )


def run_directory_paths(path: str | Path) -> RunDirectoryPaths:
    layout = run_directory_layout(path)
    return RunDirectoryPaths(
        run_id=layout.run_id,
        path=layout.path,
        log_path=layout.log_path,
        run_end_path=layout.run_end_path,
    )


def is_valid_run_id(run_id: str) -> bool:
    return bool(RUN_ID_PATTERN.match(run_id))


def validate_run_directory(path: str | Path) -> bool:
    run_path = Path(path)
    if not run_path.exists() or not run_path.is_dir():
        return False
    if not is_valid_run_id(run_path.name):
        return False
    layout = run_directory_layout(run_path)
    for filename in RUN_DIRECTORY_REQUIRED_FILES:
        if not (run_path / filename).is_file():
            return False
    if layout.log_path.exists() and not layout.log_path.is_file():
        return False
    if layout.run_end_path.exists() and not layout.run_end_path.is_file():
        return False
    return True


def manifest_path_for_output(output_path: str | Path) -> Path:
    return Path(output_path).with_name(MANIFEST_FILENAME)


def manifest_path_from_run_end(run_end_path: str | Path) -> Path | None:
    path = Path(run_end_path)
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    manifest_path = payload.get(RUN_END_MANIFEST_PATH_KEY)
    if not isinstance(manifest_path, str) or not manifest_path:
        return None
    return Path(manifest_path)


def validate_manifest_payload(payload: Mapping[str, Any]) -> bool:
    return all(key in payload for key in MANIFEST_REQUIRED_FIELDS)


def validate_summary_frame(summary_df: pd.DataFrame) -> bool:
    if not all(col in summary_df.columns for col in SUMMARY_REQUIRED_COLUMNS):
        return False
    for col, expected in SUMMARY_COLUMN_TYPES.items():
        if col not in summary_df.columns:
            return False
        if expected == "string":
            if not pd.api.types.is_string_dtype(summary_df[col]):
                return False
        elif expected == "number":
            if not pd.api.types.is_numeric_dtype(summary_df[col]):
                return False
        else:
            return False
    return True
