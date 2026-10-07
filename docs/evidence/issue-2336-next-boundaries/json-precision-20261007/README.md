# CLI JSON timestamp acceptance verification

The formatter regression case now checks both epoch zero and a nonzero timestamp
with fractional seconds. It continues to require interpolated Unicode text and an
explicit UTC offset. This catches timestamp truncation that the original epoch-only
assertion missed, without adding pytest nodes or changing production.

## Verified acceptance criteria

- [x] **Tests**
  - [x] Added focused coverage for CLI error handling, warning collection, snapshot
    isolation, JSON formatting, and invalid UTF-8 configuration.
  - [x] Focused tests pass. Historical comparison results show seven additional
    covered lines, with no increase in failures or errors.
- [x] **Documentation**
  - [x] Added supporting evidence comparing test results and coverage, with
    verification records for test behavior and source integrity.

| Measurement | Baseline | Candidate |
| --- | --- | --- |
| Historical full-suite passed cases | 1594 | 1602 |
| Historical full-suite failures / errors / skips | 0 / 0 / 1 | 0 / 0 / 1 |
| Historical full-suite covered statements | 12133 / 14553 | 12140 / 14553 |
| Historical full-suite coverage | 83.371126% | 83.419226% |
| Current related diagnostic cases | 27 passed | 27 passed |
| Current targeted CLI coverage | 652 / 1106 (59%) | 652 / 1106 (59%) |

All eight current focused cases pass. The matched related runs retain the same named
cases and executed CLI lines, with zero failures, errors, or skips. Each command
explicitly excludes slow and live-LLM tests and targets `--cov=pa_core.cli` when
measuring coverage. Exact commands, interpreter, exits, durations, source/test hashes,
console, JUnit, and coverage reports are retained in the phase receipts and gzip
captures. The related suite comprises `test_cli_observability_boundaries.py`,
`test_cli_edge_cases.py`, `test_cli_packet_diff.py`, and
`test_cli_exception_handling.py`.

`precision-mutation.json` verifies the exact formatter node against a production
mutation that converts `record.created` to `int(record.created)`: the previous test
passes, the strengthened test has exactly one assertion failure, and the strengthened
test passes after byte-identical restoration. Named-node JUnit rejects errors and
skips. The epoch assertion remains, so the new test preserves the prior expectation.

`verification.json` independently verifies the retained historical full-suite JUnit,
coverage totals, all eight original mutation pairs, and all 186 entries across the
two earlier artifact manifests. Historical test hashes are checked against their
recorded Git versions; the new manifest anchors the strengthened file. The full-suite
reports retain the same 186 sources and 181 excluded lines, with no executed-line
regression. These are historical measurements, not a fresh full-suite run of the
strengthened assertion. The prior recheck documents this runner's Kaleido export
stall; the broad coverage initiative remains open.

Black formatted the changed test at line length 100. The required whole-repository
Black check passes for all 486 Python files, and focused Ruff passes. As in the prior
recheck, a sequential check through Black's normal single-file API populated a
temporary cache before running the literal required command with one worker. Scope,
configuration, and source bytes remain unchanged. The checker source and all-source
hashes are retained. The capture, mutation, and verification script sources are also
preserved as gzip; their temporary paths appear in the receipts.

Production `pa_core/cli.py` matches base `d4ed680e` byte for byte, with SHA256
`265d1891f50333a994458086cf0c4260d580971c710ea181f99e68582e5c899f`.
`manifest.json` records stored and decoded hashes for this verification package.
