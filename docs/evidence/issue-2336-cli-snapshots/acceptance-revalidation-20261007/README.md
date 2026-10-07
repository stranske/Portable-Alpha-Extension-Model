# Warning snapshot acceptance revalidation

The two existing logging and Python-warning regression nodes now check both directions of isolation. After taking a snapshot, each test edits the live captured context's line number and verifies that the retained snapshot still contains its original context. It then edits the returned snapshot and verifies that the collector retains its updated diagnostic. No new test nodes or production changes were needed; the existing `deepcopy` boundary satisfies both checks.

- [x] **Bug Fixes**
  - [x] Warning snapshots preserve nested context independently, so later changes to captured data do not affect existing snapshots.
- [x] **Tests**
  - [x] Expanded checks for CLI reporting, configuration snapshots, and timer behavior.
- [x] **Documentation**
  - [x] Added evidence and replay records for the regression fix and its test results.

The focused diagnostic and replay-isolation suite passed all 20 nodes before and after the assertion changes on Python 3.14.7. The two additional replay-restoration nodes from commit `193d05df` explain the increase from the earlier historical 18-node focused suite. `focused.log.gz` and `focused.xml.gz` retain the final result. Its targeted `--cov=pa_core.cli` measurement is 102/1107 statements (9% displayed); this narrow diagnostic selection does not measure full-suite coverage or meet the broader 90% target.

`mutations/controls.json` and the twelve console/JUnit pairs retain a fresh replay of all six production mutations against the strengthened tests. Each exact named node failed with exit 1 under its production mutation and passed with exit 0 after byte-identical source restoration, with no errors or skips. In particular, the two shallow-copy mutations fail at the new retained-snapshot assertion. `verification.json` records the current source, test and replay hashes and the independently checked outcomes. The source matches the restored private source. The replay now selects `not slow` explicitly for every phase.

Replay from the repository root with a new output directory:

```sh
python docs/evidence/issue-2336-cli-snapshots/replay.py --output /absolute/new-proof-directory
```

The focused check is:

```sh
pytest tests/test_cli_observability_boundaries.py tests/test_cli_mutation_replay_isolation.py -m "not slow" -q --cov=pa_core.cli --cov-report=term-missing
```

Both changed Python files were individually formatted with Black at line length 100. The initial multi-file invocation failed because the sandbox denied Black's forkserver socket; the single-worker retry stalled and was interrupted. The previously documented sequential-cache workaround checked all 488 files using Black's normal single-file implementation and the required command's source selection. `black-sequential-check.py.gz`, `black-sequential-results.json` and `black-sequential.log.gz` retain that check. With its cache populated, the literal required command passed with exit 0:

```sh
BLACK_CACHE_DIR=/tmp/cli-snapshot-black-cache BLACK_NUM_WORKERS=1 black --check --line-length 100 --exclude '(\.workflows-lib|node_modules)' .
```

`black-check.log.gz` records all 488 files unchanged. Notebook formatting remains unavailable because optional Jupyter dependencies are absent. `ruff.log.gz` records the focused Ruff pass. `manifest.json` anchors these retained artifacts, including decoded gzip bytes. Console and JUnit files are losslessly gzipped; receipt paths refer to their original uncompressed names.

The original full-suite comparison and broader coverage measurements remain historical evidence in the parent directory. They were not rerun or relabeled as results for this assertion-only change. GitHub API access failed in this runner, so the remote PR checklist and readiness state could not be inspected or updated.

The workspace's Git metadata is also mounted read-only: staging failed when Git tried to create `.git/index.lock`. The verified files are committed in an isolated checkout under `/tmp`; an apply-ready patch accompanies the handoff. The workspace branch cannot be advanced from this runner.
