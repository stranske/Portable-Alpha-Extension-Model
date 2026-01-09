# CLI Usage Guide

## Overview
The CLI provides the supported entrypoint for running simulations and related workflows.
Use the `pa` console script for canonical invocation. Direct module execution
(`python -m pa_core` or `python -m pa_core.cli`) is deprecated and emits
`DeprecationWarning` (ignored by default by Python) without printing to stdout/stderr.

## Canonical Commands
- `pa run --config config.yaml --index index.csv --output Outputs.xlsx`
- `pa validate --config scenario.yaml`
- `pa registry list`
- `pa registry get <scenario-id>`
- `pa calibrate --input asset_library.xlsx --index-id SPX --output asset_library.yaml`
- `pa convert params.csv params.yaml`

## Delegation Flow
- `pa run` parses the top-level subcommand, then forwards the remaining arguments
  to `pa_core.cli.main` with `emit_deprecation_warning=False` so canonical usage
  stays quiet.
- `pa_core.cli.main` owns the full run-flag parser and constructs `RunOptions`
  before calling `pa_core.facade.run_single`.
- `python -m pa_core` remains a legacy entry point with its own minimal argparse
  configuration; it mirrors the same run pipeline for backward compatibility.

## Deprecation Warnings
Non-canonical invocation paths emit a `DeprecationWarning`:
- `python -m pa_core`
- `python -m pa_core.cli`
- Direct calls to `pa_core.cli.main` outside the `pa` command

Warnings are emitted via the Python warnings system only; they must not be
printed or logged to stdout/stderr unless a user enables them explicitly
(e.g., `-Wd` or warning filters).

## Exit Codes and Output
- Success returns exit code `0`.
- Argument errors or validation failures raise `SystemExit` with a non-zero
  exit code and a message on stderr.
- Normal status and summary output is written to stdout; stderr is reserved for
  argument parsing or validation errors.
