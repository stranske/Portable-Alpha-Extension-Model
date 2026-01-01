from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Mapping, Sequence, TypedDict

import pandas as pd

# Run directory layout (created when --log-json is enabled)
RUNS_DIR_NAME = "runs"
RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z$")
RUN_LOG_FILENAME = "run.log"
RUN_END_FILENAME = "run_end.json"
RUN_DIRECTORY_REQUIRED_FILES: Sequence[str] = (RUN_LOG_FILENAME,)
RUN_DIRECTORY_OPTIONAL_FILES: Sequence[str] = (RUN_END_FILENAME,)

# Manifest contract
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
)

# Summary contract
SUMMARY_SHEET_NAME = "Summary"
ALL_RETURNS_SHEET_NAME = "AllReturns"
DEFAULT_OUTPUT_FILENAME = "Outputs.xlsx"

SUMMARY_AGENT_COLUMN = "Agent"
SUMMARY_TE_COLUMN = "TE"
SUMMARY_TRACKING_ERROR_LEGACY_COLUMN = "TrackingErr"
SUMMARY_CVAR_COLUMN = "CVaR"
SUMMARY_BREACH_PROB_COLUMN = "BreachProb"

SUMMARY_REQUIRED_COLUMNS: Sequence[str] = (
    SUMMARY_AGENT_COLUMN,
    "AnnReturn",
    "ExcessReturn",
    "AnnVol",
    "VaR",
    SUMMARY_CVAR_COLUMN,
    "MaxDD",
    "TimeUnderWater",
    SUMMARY_BREACH_PROB_COLUMN,
    "BreachCount",
    "ShortfallProb",
    SUMMARY_TE_COLUMN,
)
SUMMARY_NUMERIC_COLUMNS: Sequence[str] = tuple(
    col for col in SUMMARY_REQUIRED_COLUMNS if col != SUMMARY_AGENT_COLUMN
)


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


def validate_manifest_payload(payload: Mapping[str, Any]) -> bool:
    return all(key in payload for key in MANIFEST_REQUIRED_FIELDS)


def validate_summary_frame(summary_df: pd.DataFrame) -> bool:
    if not all(col in summary_df.columns for col in SUMMARY_REQUIRED_COLUMNS):
        return False
    if SUMMARY_AGENT_COLUMN in summary_df.columns:
        if not pd.api.types.is_string_dtype(summary_df[SUMMARY_AGENT_COLUMN]):
            return False
    for col in SUMMARY_NUMERIC_COLUMNS:
        if col in summary_df.columns and not pd.api.types.is_numeric_dtype(summary_df[col]):
            return False
    return True
