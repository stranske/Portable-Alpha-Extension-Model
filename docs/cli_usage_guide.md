# CLI Usage Guide

## Overview
The CLI provides the supported entrypoint for running simulations and related workflows.
Use the `pa` console script for canonical invocation. Direct module execution
(`python -m pa_core` or `python -m pa_core.cli`) is deprecated and emits
`DeprecationWarning` via the Python warnings module (ignored by default) without
printing to stdout/stderr.

## Canonical Commands
- `pa run --config config.yaml --index index.csv --output Outputs.xlsx`
- `pa validate --config scenario.yaml`
- `pa registry list`
- `pa registry get <scenario-id>`
- `pa calibrate --input asset_library.xlsx --index-id SPX --output asset_library.yaml`
- `pa convert params.csv params.yaml`

## Delegation Flow
- `pa` executes `pa_core.pa:main`, which parses the top-level subcommands.
- `pa run` delegates to `pa_core.cli.main` with `emit_deprecation_warning=False`
  and passes through the remaining arguments for full CLI parsing.
- `pa_core.cli.main` parses arguments and builds `RunOptions` for `pa_core.facade`.
- `pa_core.facade.run_single` is the canonical pipeline for simulation runs and
  is used by CLI entrypoints to keep outputs consistent.
- `python -m pa_core` runs `pa_core.__main__`, which provides a narrower argument
  surface and delegates directly to `pa_core.facade.run_single` and `export`.

## Deprecation Warnings
Non-canonical invocation paths emit a `DeprecationWarning`:
- `python -m pa_core`
- `python -m pa_core.cli`
- Direct calls to `pa_core.cli.main` outside the `pa` command when
  `emit_deprecation_warning=True`

Warnings are emitted via the Python warnings system and do not appear in
stdout or stderr unless a user enables them explicitly (e.g., `-Wd`).

## Argument Parsing Details
- `pa` uses `argparse` subparsers and forwards the remaining arguments to
  `pa_core.cli.main` so `pa run` accepts the full simulation flag set.
- `pa_core.cli.main` handles the complete CLI surface for simulations, including
  exports, stress presets, logging, and sleeve suggestion controls.
- `pa_core.__main__` supports a smaller subset of flags (config/index/output,
  backend/seed/legacy RNG and return-distribution overrides) and validates
  volatility regime settings before delegation.

## Exit Codes and Output
- Success returns exit code `0`.
- Argument errors or validation failures raise `SystemExit` with a non-zero
  exit code and a message on stderr (e.g., `argparse` uses exit code `2`).
- Normal status, summaries, and guidance are written to stdout.
