# pa_core.facade

## Purpose
`pa_core.facade` provides the canonical programmatic pipeline for CLI and
library usage. It centralizes run orchestration and export behavior so that
CLI entrypoints stay consistent and regression-safe.

## Key APIs
- `RunOptions`: configuration overrides for CLI and programmatic runs.
- `RunArtifacts`: standardized outputs for a single simulation.
- `SweepArtifacts`: standardized outputs for parameter sweeps.
- `run_single`: canonical simulation pipeline used by CLI entrypoints.
- `run_sweep`: canonical parameter sweep pipeline.
- `export`: shared export helper for artifact bundles.

## Delegation from CLI
- `pa` (the console script) parses the top-level subcommand and forwards run
  arguments to `pa_core.cli.main` so the full run parser stays centralized.
- `pa_core.cli.main` owns the authoritative run argument parser, builds
  `RunOptions`, and calls `run_single` from `pa_core.facade`.
- `python -m pa_core` is a legacy entry point with a smaller argparse surface,
  but it still routes through the same `run_single` pipeline to keep outputs
  aligned with the canonical CLI.
- `python -m pa_core.cli` uses the same run parser as `pa run`, but is treated as
  legacy and emits a deprecation warning by default.

## Deprecation Warning Behavior
Non-canonical invocation paths emit `DeprecationWarning` via the Python
warnings system. These warnings do not appear in stdout/stderr by default
and must not be printed or logged as part of CLI output. Users can surface
them explicitly with `-Wd` or warning filters.

## Argument Parsing and Exit Codes
The facade layer does not parse CLI arguments. Argument parsing lives in CLI
entrypoints (`pa_core.cli`, `pa_core.pa`, and `pa_core.__main__`):
- `pa_core.pa` parses only the top-level subcommand and uses `parse_known_args`
  so run-specific flags are handled by `pa_core.cli.main`.
- `pa_core.cli.main` defines the full run-flag surface and converts the parsed
  namespace into `RunOptions` for the facade.
- `pa_core.__main__` intentionally supports a smaller set of flags to preserve
  legacy behavior while still routing through `run_single`.

Exit code definitions follow CLI conventions:
- `0` for successful runs to simplify automation.
- `2` for usage/argument parsing errors (argparse default), signaling callers to
  fix the invocation.
- Other non-zero exits for validation/runtime failures raised by the CLI layer.
Deprecation warnings are informational only and do not affect exit status.
