# CLI validation acceptance revalidation

This iteration strengthens the same five production CLI validation cases without changing production code, coverage policy, or test node identities. Every invocation now installs a caller-owned logging handler and warning callback and asserts exact restoration after success, advisory results, error results, and caught validator exceptions. The forwarding case distinguishes the loaded configuration from the effective configuration returned by the facade, using a real temporary financing schedule, an 18-month term, 321 simulations, and nondefault values for all eight step sizes. Config loading and facade option application remain stubbed: these assertions protect the CLI boundary rather than proving configuration loading or CLI override parsing end to end.

Verified acceptance checklist:

- [x] **Tests**
  - [x] Added CLI validation tests covering validator exceptions, diagnostic details, error and advisory exit behavior, effective setting forwarding, and preservation of logging and warning handlers.
  - [x] The added tests increased passing test and covered-line counts, with no changes to existing test outcomes. This comparison is verified from the original paired suite captures; the strengthened five cases still pass in the current focused run.
- [x] **Documentation**
  - [x] Added an evidence report documenting test scope, suite and coverage comparisons, production-mutation checks, and archived evidence integrity.

## Current checks

`focused.log` and `focused.xml` record six passing cases: the five boundary cases and the existing validation-only regression. The command explicitly selects `not slow and not live_llm`. Targeted `--cov=pa_core.cli` coverage is 223/1107 statements (20% displayed). This narrow selection does not replace the historical full-suite coverage or satisfy the broader initiative's 90% target.

Eight isolated production mutations each fail the exact intended AssertionError and then pass after byte-identical restoration: drop exception details; reject warning; reject info; accept error; replace the financing term with its default; ignore the facade's returned effective configuration; leak the logging collector handler; and fail to restore the warning callback. `mutation-controls.json` records node identities, source/test hashes, command arguments, exit codes, and restoration checks. The caller's production and test bytes stay unchanged throughout replay. An initial financing-term mutation used `None`; the real schedule validator then failed earlier, so that pilot is retained as `financing-term-initial-red.*` and is excluded from the eight verified pairs. The final default-term mutation reaches the intended forwarding assertion. The original historical five mutation pairs remain intact.

Black formatted the changed Python file at line length 100. The initial repository check stalled and was interrupted. The repository's established sequential-cache workaround then checked all 489 selected files unchanged using Black's normal single-file implementation and populated a writable temporary cache. The exact required check passed with exit 0:

```sh
BLACK_CACHE_DIR=/tmp/pae-validation-black-cache BLACK_NUM_WORKERS=1 black --check --line-length 100 --exclude '(\.workflows-lib|node_modules)' .
```

`black-sequential-results.json` binds all checked file bytes. Black skipped notebooks because optional Jupyter dependencies are absent. Focused Ruff and `git diff --check` also pass.

## Historical suite and integrity checks

All 41 original archive member hashes and the archive hash match the existing manifest. The original source/test/config bindings match the starting commit, and current production/config bytes still match those bindings. The historical test binding is retained as a record of that run; `manifest.json` in this directory binds the strengthened current tests separately.

`check_historical.py` independently reconstructs the paired outcomes and coverage comparison. It verifies 1614 to 1619 passing cases, the same one skip, exactly five added cases, and no changes to existing outcomes. Coverage remains 12143 to 12147 covered statements out of the same 14554, with 186 measured files and 181 exclusions. The only gains are CLI lines 820, 821, 1047, and 1048, with no lost covered lines. Both historical processes exited 0, and all five historical mutation controls passed their intended RED/GREEN verification. These measurements belong to the archived original test snapshot.

A fresh paired-suite attempt in this runner selected `not slow and not live_llm`; the baseline omitted only the five added boundary cases. It stalled at the existing `tests/test_cli.py::test_main_with_png` and was interrupted before the candidate run started. The independent bounded PNG probe timed out after 60 seconds; its 30-second traceback shows Plotly/Kaleido image export waiting on its browser worker. The raw partial suite console, diagnostic traceback, and process receipt are preserved. No new full-suite passing count or aggregate coverage result is claimed for the strengthened test snapshot, and no existing test was changed or disabled to bypass the stall.

## Delivery and archive

`raw-proof.tar.gz` retains the current console/JUnit/targeted-coverage/process/environment/mutation receipts, capture scripts, and independent historical verification. This directory's `manifest.json` hashes every archive member and binds current source/test/config plus the original manifest and archive. Scripts retain capture-time absolute paths and need adaptation before replay in another checkout.

The workspace Git metadata is mounted read-only: staging fails when Git tries to create `.git/index.lock`. The tested source and evidence changes are committed in a writable checkout under `/tmp` and exported as a delivery patch; the workspace branch cannot be advanced here. GitHub API access also failed, so the remote PR checklist and readiness state could not be inspected or updated. The checked acceptance list above records verified implementation locally.
