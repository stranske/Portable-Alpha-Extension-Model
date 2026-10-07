# Interruption recovery after completed replay phases

The prior commits already satisfy the bounded snapshot acceptance criteria.
Before this change, the focused suite passed all 20 nodes, and all 50 entries in
the nested-context manifest matched their stored bytes and declared decoded
hashes. The original six-case/full-suite coverage comparison remains historical;
the broader 90% initiative remains open.

- [x] **Bug Fixes**
  - [x] Warning snapshots preserve nested context independently, so later changes
    to captured data do not affect existing snapshots.
- [x] **Tests**
  - [x] Expanded checks for CLI reporting, configuration snapshots, timer behavior,
    and replay-driver recovery when interrupted.
- [x] **Documentation**
  - [x] Added evidence and replay records for the regression fix, test results,
    and interruption handling.

The two existing timeout/KeyboardInterrupt snapshot-driver nodes now interrupt
three positions each: first RED, restored GREEN, and the next case's RED. They
verify byte-identical source/test restoration, completed phase receipts and source
hashes, and the absence of a completed receipt/JUnit file for the interrupted
phase. Successful child results are simulated only to reach these interruption
positions; the real driver validates their named JUnit results and writes receipts.
No test nodes or production changes were added. The focused suite remains 20 PASS.

```sh
pytest tests/test_cli_observability_boundaries.py tests/test_cli_mutation_replay_isolation.py -m "not slow" -q --cov=pa_core.cli --cov-report=term-missing
```

Targeted coverage reports CLI 102/1107 statements, displayed as 9%. This selection
does not measure full-suite coverage or establish a broader coverage target.

The separate mutation proof discards `case["phases"].append(record)` in the actual
snapshot replay driver in a private tree. Both named interruption nodes fail at
the missing completed receipt assertion, with zero errors/skips. Restoring the
exact driver bytes yields two passes. `controls.json`, RED/GREEN console and JUnit
captures retain this proof. To reproduce it, decompress `mutation-replay.py.gz`
outside the repository and run it from the repository root with a new output
directory as its sole argument. Caller files are never mutated by this proof.

Black formatted the changed test at line length 100; focused Ruff passed. The
first whole-repository Black command stalled and was interrupted (exit 130).
Black's sequential source check verified all 488 files unchanged and populated a
private cache. The required literal command then passed (exit 0):

```sh
BLACK_CACHE_DIR=/tmp/2342-interruption-black-cache BLACK_NUM_WORKERS=1 black --check --line-length 100 --exclude '(\.workflows-lib|node_modules)' .
```

Notebook formatting remains unavailable because optional Jupyter dependencies
are absent. The sequential check and final command captures are retained. The
manifest binds this revision's source/test/driver and stored/decoded evidence
bytes; earlier manifests remain revision-bound and unchanged.

PR #2342 was confirmed open and ready for review. Its release-note bullets were
interpreted as unchecked acceptance by the task prompt, while the PR body had no
explicit acceptance checklist. An explicit, verified checklist was prepared in
`reconciled-pr-body.md.gz`; the connector rejected the update because approval is
required and this run's approval policy is `never`. Remote checkboxes therefore
remain unapplied. The workspace Git metadata is read-only (`.git/index.lock`
creation was denied); a local commit and apply-ready patch are prepared under
`/tmp` using separate writable Git metadata. Neither restriction is a test failure.

The final readiness check found that PR #2342 had merged independently during
this run at commit `c569018fda1433c0696febbb498ff2b522616fac`, with `isDraft=false`.
This supplemental test change is retained in the local commit/patch and is not
included in that merge. `final-pr-state.json` records the observed disposition.
