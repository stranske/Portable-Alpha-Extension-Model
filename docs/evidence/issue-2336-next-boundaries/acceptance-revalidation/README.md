# Acceptance reconciliation and replay controls

Reviewed commits `3b934060`, `98fbf8ba`, `2800afbc`, and `0dba045d` before extending the replay controls. The eight CLI cases, stronger repeated-installation and fractional-timestamp assertions, private replay checkout, timeout control, and actual SIGTERM control already satisfy the implementation acceptance items. Revalidated the existing ten focused cases and all 110 current-manifest artifact hashes before attempting to reconcile the PR body.

Verified acceptance checklist for PR #2341:

- [x] **Tests**
  - [x] Added focused checks for CLI error handling, warning collection, snapshot isolation, JSON formatting, and invalid UTF-8 configuration.
  - [x] Added checks that mutation replay leaves the active checkout unchanged, including when interrupted. Focused tests pass; historical comparisons show seven additional covered lines with no increase in failures or errors.
- [x] **Documentation**
  - [x] Added evidence and verification records covering test results, coverage comparisons, replay behavior, and source integrity.

The GitHub connector refused the PR-body update because it requires approval and this run's approval policy is `never`. This checked list records verified completion locally; the remote body was not changed. PR #2341 was open and ready for review when inspected.

Staging a commit in the active checkout was also blocked: `.git/index.lock` cannot be created on its read-only filesystem. A separate writable local checkout is used to commit the same reviewed files and export a delivery patch. The active working-tree changes remain available to the automation collector.

This iteration adds automated existing-output refusal before any snapshot or mutation, KeyboardInterrupt cleanup, and whole-fixture byte comparisons that also protect test files and pre-existing caller caches after timeout or SIGTERM. Twelve focused cases pass with zero failures, errors, or skips. Two deliberately broken replay drivers were tested only in temporary copies: allowing output-directory reuse fails the refusal assertion; deleting active caller caches fails the KeyboardInterrupt integrity assertion. These driver controls supplement the previously preserved eight real production RED/GREEN cases.

`verification.json` retains command arguments, working directories, exit codes, durations, and source/test hashes. The lossless gzip console and JUnit records include both failed driver controls and the passing focused suite. Current targeted coverage is deliberately narrow (`--cov=pa_core.cli`): 94/1106 statements, displayed as 8%. It does not replace the historical full-suite measurement. The historical captures were independently checked again: 1594→1602 passes, 12133→12140 covered statements, the same 14553 statements, 186 files and 181 exclusions, no lost covered lines, no failures/errors, and production bytes equal to the base.

Focused Ruff and the required repository-wide Black command pass. Black's default process workers cannot start in this sandbox; a one-worker attempt with the default unwritable cache timed out. The established sequential Black check then verified all 487 selected files unchanged and populated a writable temporary cache. The exact required check subsequently passed with `BLACK_NUM_WORKERS=1` and `BLACK_CACHE_DIR=/tmp/cli-acceptance-black-cache`. No formatter checks or safety checks were disabled. Notebook formatting was skipped by Black because its optional Jupyter dependencies are absent. `black-sequential-results.json.gz` binds all 487 checked file bytes. The prior isolation manifest remains a historical record; this iteration's receipt binds the strengthened current test module.
