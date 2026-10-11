# Issue 2336 — missing previous workbook path fallback

Increment after merged #2344: when `prev_manifest.cli_args.output` points at a path that does not exist, sweep `--packet` must pass an empty `prev_summary_df` (no fabricated metrics) while preserving `prev_manifest`.

The focused regression now covers both a missing workbook in an existing directory and a workbook whose parent directory is missing. It checks that the path remains absent and `pd.read_excel` is never called. The read stub records calls before raising `FileNotFoundError`, so the assertion also detects an unnecessary read that the CLI's error fallback would otherwise hide.

## Verified tasks

- [x] Add a focused sweep `--packet` test for manifests whose `cli_args.output` path is missing on disk.
- [x] Assert export receives an empty `prev_summary_df` with no rows or columns while passing through `prev_manifest`.
- [x] Document the increment in this evidence directory.
- [x] Run the focused acceptance test.
- [ ] Hosted Gate / keepalive on this PR: unverified; this environment cannot reach `api.github.com`.

## Validation

- Focused: `python -m pytest tests/test_cli_packet_diff.py::test_sweep_packet_prev_summary_empty_when_prev_output_missing -q` — **2 passed**.
- Packet regression tests: `python -m pytest tests/test_cli_packet_diff.py tests/test_export_packet.py -q -m 'not slow'` — **17 passed**.
- Formatting: `black --line-length 100 tests/test_cli_packet_diff.py`, then `black --check --line-length 100 --exclude '(\.workflows-lib|node_modules)' .` — **489 files unchanged**, exit code 0. The sandbox blocks the default multiprocessing socket and the single-worker event loop stalls; the successful check used `BLACK_NUM_WORKERS=1` and a temporary `/tmp` startup helper that adds an event-loop heartbeat. The helper changes scheduling only and is not part of the repository.
- Deliberate mutation: temporarily replace `if prev_out and Path(prev_out).exists():` with `if prev_out:` in `pa_core/cli.py`. Both cases fail at the assertion that no workbook reads occur (**2 failed**, pytest exit code 1). The exception fallback still allows packet export, which is why the read-call assertion is necessary. Restore the production source and rerun the focused acceptance command: **2 passed**. No production mutation is retained.

Hosted CI is authoritative for full-suite and coverage deltas on this PR.

The repository's `.git` directory is mounted read-only in this run, so updating its index or branch is blocked. The tested changes remain in the working tree; a commit and portable patch are prepared with isolated Git metadata under `/tmp` for automation handoff.
