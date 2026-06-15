# Run Record Contract (`run.json`)

The blueprint `run_contract` standard requires each run to be representable as a
single JSON object so runs replay and test. `run.json` is that envelope. It is
written next to `manifest.json` (and `run_end.json` when `--log-json` is
enabled) and references the existing per-run artifacts instead of duplicating
them.

## Filename

`run.json` (`pa_core.contracts.RUN_RECORD_FILENAME`).

## Shape

| Field           | Type                              | Notes                                              |
| --------------- | --------------------------------- | -------------------------------------------------- |
| `manifest_path` | `str \| null`                     | Path to the run's `manifest.json`.                 |
| `run_end_path`  | `str \| null`                     | Path to `run_end.json` (null without `--log-json`).|
| `bundle_path`   | `str \| null`                     | Path to `bundle.json` when `--bundle` is used.     |
| `warnings`      | `list[Warning]`                   | Captured run-level warnings (see below).           |
| `cost`          | `{latency_seconds, dollars}`      | `dollars` is a stub (`null`) on the numpy backend. |

### Warning shape

Each warning is normalized to four keys
(`pa_core.contracts.RUN_RECORD_WARNING_FIELDS`):

| Field      | Type   | Notes                                                  |
| ---------- | ------ | ------------------------------------------------------ |
| `code`     | `str`  | Warning category (e.g. `UserWarning`) or logger name.  |
| `severity` | `str`  | `warning` / `error` (lower-cased log level).           |
| `message`  | `str`  | Human-readable warning text.                           |
| `context`  | `dict` | `source` (`warnings`/`logging`) plus origin metadata.  |

## Capture sources

`pa_core.cli._WarningCollector` installs before the heavy-import bootstrap and
captures from two channels:

- the `warnings.warn(...)` channel — e.g. frequency-mismatch / unknown-frequency
  warnings in `pa_core.data.loaders`;
- `logging.WARNING`+ records — e.g. sweep-perturbation failures and artifact
  bundle failures.

## Manifest mirroring

`warnings` and `cost` are also appended to `MANIFEST_OPTIONAL_FIELDS` and written
onto `manifest.json` / `run_end.json` as additive optional keys. They are absent
from `MANIFEST_REQUIRED_FIELDS`, so `validate_manifest_payload` remains backward
compatible with legacy manifests that predate these fields.

## Reproducibility

`manifest.json` ties every board number to a versioned config + seed + data
hash:

- `config_hash` (additive optional field) is the SHA-256 of the config file
  bytes, recorded alongside the per-file `data_files` hashes so the exact config
  that produced a run can be verified even if the file is later edited.
- A run without an explicit `--seed` (`seed=None`) uses a non-deterministic RNG
  and cannot be reproduced. `ManifestWriter.write` emits
  `SEED_REPRODUCIBILITY_WARNING` via the `warnings.warn(...)` channel, so the
  warning is also captured into the run-record `warnings` list above.
