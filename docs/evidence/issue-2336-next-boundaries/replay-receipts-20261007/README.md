# Replay receipt and output preservation verification

Reviewed `7bf98655`, `dd16a4ab`, and `0dba045d` before making this change. The
existing implementation already satisfies the five acceptance items. All 12
focused cases passed before editing; all 133 prior manifest entries, their decoded
hashes, and all eight historical production mutation pairs independently verified.

Verified acceptance checklist for PR #2341:

- [x] **Tests**
  - [x] Added focused checks for CLI error handling, warning collection and snapshot
    isolation, JSON formatting and UTC timestamps, and invalid UTF-8 configuration.
  - [x] Added checks that interrupted mutation replays preserve the active checkout
    and caller data, and that existing output directories are left untouched.
    Focused tests pass; historical comparisons show seven additional covered lines
    with no increase in failures or errors.
- [x] **Documentation**
  - [x] Added evidence and verification records covering test results, coverage
    comparisons, replay safeguards, and source integrity.

The GitHub connector rejected the reconciled PR-body update because it requires
approval and this run's approval policy is `never`. The remote body remains
unchanged; this checklist records verified completion locally.

The active checkout's Git index is read-only: staging fails when creating
`.git/index.lock`. The reviewed source and evidence are committed in a writable
temporary checkout and exported as a delivery patch. Workspace edits remain
available to the automation collector.

The interruption checks now require a retained `controls.json` with the original
source/test hashes, the actual mutant hash, and the restored source hash. An
interrupted phase must not claim completion, and its console artifact must remain.
Both timeout and KeyboardInterrupt exercise these assertions. Existing-output
refusal also checks a nested console artifact and its containing directory, in
addition to the original proof and caller checkout.

Two replay-driver mutations ran only in temporary copies. Recording the mutant hash
as the restored hash passes both previous interruption cases, fails both strengthened
cases, and passes both after byte-identical driver restoration. Deleting nested
output artifacts before refusing reuse passes the previous refusal case, fails the
strengthened case, and passes after restoration. All phases have the expected named
JUnit cases and exits, with zero errors or skips. Console, JUnit, command arguments,
working directories, durations, and hashes are retained here.

All 31 related CLI and replay cases pass with `-m "not slow"`. Targeted
`--cov=pa_core.cli` reports 652/1106 statements (59%). This narrow measurement does
not replace the historical full-suite comparison: 1594 to 1602 passes, 12133 to
12140 covered statements out of 14553, the same 186 source files and 181 exclusions,
no covered-line regressions, zero failures/errors, one skip and two deselections in
each run. These historical reports were independently verified again. The broad
coverage initiative remains open.

Black formatted the changed test with line length 100. A one-worker repository check
stalled and was interrupted. Black's normal sequential single-file API then checked
the exact repository scope, verified unchanged hashes, and populated a writable
temporary cache. The required literal repository-wide Black command subsequently
passed with one worker, and focused Ruff passed. No checks or formatting scope were
disabled. The checker, its source hashes, and both command outputs are retained.

Production and replay-driver bytes are unchanged. Production matches base
`d4ed680e` exactly, with SHA256
`265d1891f50333a994458086cf0c4260d580971c710ea181f99e68582e5c899f`.
The previous manifest is preserved losslessly in `manifest-before-receipts.json.gz`;
the updated parent manifest binds this evidence and the strengthened test.
