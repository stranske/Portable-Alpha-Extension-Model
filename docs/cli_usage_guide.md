# CLI Usage Guide

## Overview
The CLI provides the supported entrypoint for running simulations and related workflows.
Use the `pa` console script for canonical invocation. Direct module execution
(`python -m pa_core` or `python -m pa_core.cli`) is deprecated and emits
`DeprecationWarning` via the Python warnings system (ignored by default by Python);
the warning itself does not print to stdout/stderr unless warnings are enabled.

## Canonical Commands
- `pa run --config config.yaml --index index.csv --output Outputs.xlsx`
- `pa validate --config scenario.yaml`
- `pa registry list`
- `pa registry get <scenario-id>`
- `pa calibrate --input asset_library.xlsx --index-id SPX --output asset_library.yaml`
- `pa convert params.csv params.yaml`

## Delegation Flow
- `pa` is the canonical entrypoint (`pa_core.pa.main`). It parses only the
  top-level command (run/validate/registry/calibrate/convert) and uses
  `parse_known_args` to pass run-specific flags through untouched.
- `pa run` forwards the remaining arguments to `pa_core.cli.main` with
  `emit_deprecation_warning=False` so canonical usage stays quiet.
- `pa_core.cli.main` owns the full run-flag parser, constructs `RunOptions`,
  calls `pa_core.facade.run_single`, and then handles export/output.
- `python -m pa_core` remains a legacy entry point with its own minimal argparse
  configuration; it mirrors the same run pipeline for backward compatibility.
- `python -m pa_core.cli` executes the same run parser as `pa run`, but is treated
  as legacy and emits a warning by default.

## Argument Parsing Changes
- The authoritative run-flag parser now lives in `pa_core.cli.main`; this keeps
  argument behavior consistent regardless of whether the run is triggered by
  `pa run` or direct module execution.
- `pa` only parses the top-level subcommand and defers all run-specific flags
  to `pa_core.cli.main`. This avoids duplicated flag definitions and keeps
  `pa --help` focused on subcommands while `pa run --help` exposes run flags.
- The legacy `python -m pa_core` parser intentionally supports a subset of
  flags (config/index/output/backend/seed/return overrides). Unsupported or
  new flags should be provided via `pa run` to avoid argparse errors.

## Deprecation Warnings
Non-canonical invocation paths emit a `DeprecationWarning`:
- `python -m pa_core`
- `python -m pa_core.cli`
- Direct calls to `pa_core.cli.main` outside the `pa` command

Warnings are emitted via the Python warnings system only; they must not be
printed or logged to stdout/stderr unless a user enables them explicitly
(e.g., `-Wd`, `PYTHONWARNINGS=default`, or explicit warning filters). The warning
signal is informational and does not change exit codes.

## Exit Codes and Output
- Success returns exit code `0` so shell scripts can detect a clean run.
- Argument parsing errors follow argparse conventions and exit with code `2`,
  which signals invalid usage rather than runtime failure.
- Validation or runtime failures bubble up as non-zero exit codes (typically via
  `SystemExit`) with error details written to stderr.
- Normal status and summary output is written to stdout; stderr is reserved for
  argument parsing, validation messages, and tracebacks.
- The exit-code split allows callers to distinguish usage errors (fix the command)
  from runtime errors (fix inputs or environment) in automation.
