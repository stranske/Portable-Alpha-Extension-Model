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
- Expected stdout/stderr lines asserted by CLI tests are centralized in
  `tests/expected_cli_outputs.py`; when CLI output changes, update those
  constants alongside any facade or parser adjustments.

## Centralized Expected CLI Outputs
CLI output expectations are defined once in `tests/expected_cli_outputs.py` and
referenced by test assertions in `tests/test_main.py`. Centralizing the strings
keeps regression checks readable and ensures all entry points (`pa run`,
`python -m pa_core.cli`, and legacy `python -m pa_core`) are held to the same
output contract.

Example test usage:

```python
from tests.expected_cli_outputs import BACKEND_USING_NUMPY_LINE

stdout = result.stdout.strip()
assert BACKEND_USING_NUMPY_LINE in stdout
```

## Inline Comment Updates for Parsing/Delegation
The CLI entry points now carry explicit inline comments that connect argument
parsing to the facade delegation path and call out the expected output fixtures.
When adjusting parsing or output sequencing, update:

- `pa_core/cli.py` to keep the canonical argparse surface and output order aligned
  with `tests/expected_cli_outputs.py`.
- `pa_core/__main__.py` to keep the legacy parser synchronized with shared flags
  and ensure delegated output lines still match the constants file.

## Deprecation Warning Behavior
Non-canonical invocation paths emit `DeprecationWarning` via the Python
warnings system. These warnings do not appear in stdout/stderr by default
and must not be printed or logged as part of CLI output. Users can surface
them explicitly with `-Wd` or warning filters.

## Unified Run Record (`run.json`)
Each run is representable as a single JSON envelope, `run.json`, written next to
`manifest.json` (and, when `--log-json` is used, alongside `run_end.json`). It
ties the existing artifacts together and adds two structured fields so an agent
reading the structured output can answer "did this run warn about anything?" and
"what did it cost?" without scraping `run.log`.

`run.json` fields:

- `manifest_path` — path to the run's `manifest.json` (or `null`).
- `run_end_path` — path to the run's `run_end.json` (or `null` when `--log-json`
  is not enabled).
- `bundle_path` — path to the optional `bundle.json` artifact (or `null`).
- `warnings` — a list of normalized warning objects (see below).
- `cost` — `{ "latency_seconds": float, "dollars": float | null }`. `dollars` is
  a deliberate stub (`null`) for the local numpy backend.

Each entry in `warnings` has the four-key shape:

```json
{ "code": "UserWarning", "severity": "warning",
  "message": "Index frequency mismatch: ...",
  "context": { "source": "warnings", "filename": "...", "lineno": 152 } }
```

Warnings are captured during the run by an in-process collector
(`pa_core.cli._WarningCollector`) that hooks both `logging.WARNING`+ records
(e.g. sweep-perturbation and bundle failures) and the `warnings.warn(...)`
channel (e.g. frequency-mismatch warnings from `pa_core.data.loaders`). The same
`warnings` and `cost` fields are mirrored onto `manifest.json` and
`run_end.json` as additive optional keys (`MANIFEST_OPTIONAL_FIELDS`), so
existing consumers that ignore them keep working.

Note: declaring `--index-frequency` explicitly overrides frequency detection;
a mismatch against the expected monthly cadence is then surfaced as a captured
warning rather than a hard error (consistent with the loaders guidance to pass
`--index-frequency {detected}` to skip strict validation).

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
