# Nested warning context acceptance

The existing `deepcopy` snapshot boundary is correct. The two existing warning
regression nodes now exercise dictionaries inside lists and lists inside those
dictionaries, for both logging and Python warnings. Later collector edits must
leave a retained snapshot unchanged. Caller edits must leave both the collector
and another snapshot unchanged. These assertions strengthen the first unchecked
acceptance criterion without adding test nodes or changing production code.

- [x] **Bug Fixes**
  - [x] Warning snapshots preserve nested context independently, so later changes
    to captured data do not affect existing snapshots.
- [x] **Tests**
  - [x] Expanded checks for CLI reporting, configuration snapshots, timer behavior,
    and replay-driver recovery when interrupted.
- [x] **Documentation**
  - [x] Added evidence and replay records for the regression fix, test results,
    and interruption handling.

The focused suite passes all 20 existing nodes on Python 3.14.7, including CLI
comparison error handling, both warning sources, monthly returns and benchmark
summaries, original UTF-8/CRLF configuration bytes, monotonic/UTC timing, timeout,
KeyboardInterrupt, and POSIX SIGTERM isolation. The snapshot driver's timeout and
KeyboardInterrupt tests verify byte-identical restoration and partial receipts
without completed phase claims. The private driver's SIGTERM test verifies the
caller remains unchanged; a temporary private tree may survive termination.
`focused.log.gz` and `focused.xml.gz` retain the results.

The targeted coverage command is:

```sh
pytest tests/test_cli_observability_boundaries.py tests/test_cli_mutation_replay_isolation.py -m "not slow" -q --cov=pa_core.cli --cov-report=term-missing
```

Its limited selection covers 102/1107 CLI statements (9% displayed). The parent
directory's full-suite comparison and coverage measurements remain historical.
No current full-suite measurement or broader 90% achievement is claimed.

`six-mutations/controls.json` and its console/JUnit pairs retain a fresh run of the
existing six-case replay against these strengthened assertions. Each named node
fails under an actual production mutation (exit 1) and passes after byte-identical
restoration (exit 0), with no errors or skips. Replay with:

```sh
python docs/evidence/issue-2336-cli-snapshots/replay.py --output /absolute/new-proof-directory
```

`nested-mutations/controls.json` additionally proves both warning nodes reject a
mutation that copies the outer record **and** the context dictionary, while
sharing their deeper containers. This mutation would pass the prior flat-context
assertions. Both nodes now fail at the retained-snapshot equality check because
the nested filename and tag list changed, then pass after restoring the original
production bytes. `nested-replay.py.gz` retains the standalone replay script;
decompress it outside the repository and run it from the repository root with a
new output directory as its sole argument. Both drivers mutate private copies.

The changed test module was formatted with Black at line length 100. The first
whole-repository check stalled in this sandbox and was interrupted. The retained
sequential check uses Black's own source selection, single-file check and cache
writer; it checked all 488 files unchanged. With that cache populated, the literal
required command passes with exit 0:

```sh
BLACK_CACHE_DIR=/tmp/cli-deep-context-black-cache BLACK_NUM_WORKERS=1 black --check --line-length 100 --exclude '(\.workflows-lib|node_modules)' .
```

`black-final.log.gz`, `black-sequential.py.gz`, `black-sequential-results.json`, and
`black-sequential.log.gz` retain this verification. Optional notebook formatting
is unavailable because Jupyter formatter dependencies are absent. `ruff.log.gz`
records the focused Ruff pass. `manifest.json` binds current source, test and
driver bytes and all evidence members, including decoded gzip hashes. Earlier
evidence directories and manifests are preserved unchanged.

The checkboxes above record locally verified acceptance. GitHub API access failed,
so the remote PR checklist and readiness state could not be inspected or changed.
The workspace Git metadata is read-only and staging failed at `.git/index.lock`.
The source/test changes and evidence are committed in a writable checkout under
`/tmp`, with an apply-ready patch for advancing the workspace branch elsewhere.
