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
- `pa run` delegates to `pa_core.cli.main`, which parses arguments and builds
  `RunOptions` for `pa_core.facade`.
- `pa_core.facade.run_single` is the canonical pipeline for simulation runs and
  is used by CLI entrypoints to keep outputs consistent.

## Deprecation Warnings
Non-canonical invocation paths emit a `DeprecationWarning`:
- `python -m pa_core`
- `python -m pa_core.cli`
- Direct calls to `pa_core.cli.main` outside the `pa` command

Warnings are emitted via the Python warnings system and do not appear in
stdout or stderr unless a user enables them explicitly (e.g., `-Wd`).

## Exit Codes and Output
- Success returns exit code `0`.
- Argument errors or validation failures raise `SystemExit` with a non-zero
  exit code and a message on stderr.
- Normal status and summary output is written to stdout.
