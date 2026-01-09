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
- `pa run` delegates to `pa_core.cli.main`.
- `pa_core.cli.main` owns the run argument parser, builds `RunOptions`, and
  calls `run_single` from `pa_core.facade`.
- `python -m pa_core` is a legacy entry point with a smaller argparse surface,
  but it still routes through the same `run_single` pipeline to keep outputs
  aligned with the canonical CLI.

## Deprecation Warning Behavior
Non-canonical invocation paths emit `DeprecationWarning` via the Python
warnings system. These warnings do not appear in stdout/stderr by default
and must not be printed or logged as part of CLI output. Users can surface
them explicitly with `-Wd` or warning filters.

## Argument Parsing and Exit Codes
The facade layer does not parse CLI arguments. Argument parsing lives in CLI
entrypoints (`pa_core.cli` and `pa_core.__main__`). Errors raised there surface
as non-zero exit codes (typically via `SystemExit`), while successful runs
return exit code `0`.
