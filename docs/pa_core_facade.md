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
- `pa` executes `pa_core.pa:main` and routes `pa run` to `pa_core.cli.main`.
- `pa_core.cli.main` builds `RunOptions` and calls `run_single` from
  `pa_core.facade` after parsing the full set of simulation flags.
- `python -m pa_core` executes `pa_core.__main__`, which parses a smaller
  argument set and delegates directly to `run_single` and `export`.
- `python -m pa_core.cli` runs the full simulation CLI but is deprecated in
  favor of `pa run`.

## Deprecation Warning Behavior
Non-canonical invocation paths emit `DeprecationWarning` via the Python
warnings system. These warnings do not appear in stdout/stderr by default.
`pa run` suppresses the warning by calling `pa_core.cli.main` with
`emit_deprecation_warning=False`. Users can surface warnings explicitly with
`-Wd` or warning filters.

## Argument Parsing and Exit Codes
The facade layer does not parse CLI arguments. Argument parsing lives in CLI
entrypoints (`pa_core.cli` and `pa_core.__main__`). Errors raised there surface
as non-zero exit codes (typically via `SystemExit`), while successful runs
return exit code `0`. Facade functions themselves return artifacts and do not
emit warnings or write to stdout/stderr.
