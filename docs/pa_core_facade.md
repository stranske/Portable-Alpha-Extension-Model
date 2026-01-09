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
- `pa_core.cli.main` builds `RunOptions` and calls `run_single` from
  `pa_core.facade`.
- `python -m pa_core` remains available but is deprecated in favor of `pa run`.

## Deprecation Warning Behavior
Non-canonical invocation paths emit `DeprecationWarning` via the Python
warnings system. These warnings do not appear in stdout/stderr by default.
Users can surface them explicitly with `-Wd` or warning filters.

## Argument Parsing and Exit Codes
The facade layer does not parse CLI arguments. Argument parsing lives in CLI
entrypoints (`pa_core.cli` and `pa_core.__main__`). Errors raised there surface
as non-zero exit codes (typically via `SystemExit`), while successful runs
return exit code `0`.
