# Issue 2336 — missing previous workbook path fallback

Increment after merged #2344: when `prev_manifest.cli_args.output` points at a path that does not exist, sweep `--packet` must pass an empty `prev_summary_df` (no fabricated metrics).

## Validation

- Focused: `python -m pytest tests/test_cli_packet_diff.py::test_sweep_packet_prev_summary_empty_when_prev_output_missing -q`
- Deliberate mutation: remove the `Path(prev_out).exists()` guard in `pa_core/cli.py` (lines ~1011–1015) so a missing path attempts `read_excel` and fails the packet path; restored production yields GREEN on the focused node above.

Hosted CI is authoritative for full-suite and coverage deltas on this PR.
