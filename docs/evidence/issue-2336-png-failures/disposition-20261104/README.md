# Packaging failure diagnostics and task reconciliation

This follow-up starts at PR head
`17ba06520d668c1a39857118c0b65992297286f7`, with main still pinned to
`7613ffd97a450b15e550f499b7386ec4e67f2c51`. Production files are unchanged.

Reviewing both prior implementation commits and validating the retained recheck
manifest confirms **13,493 / 14,626 covered statements (92.25352112676056%)**,
with **2,008 passed, 2 failed, 3 skipped, 4 deselected**. Those are the prior
completed capture's counts, not the counts of this follow-up's focused tests.
The [retained recheck](../recheck-20261104/README.md) contains the complete command,
input hashes, per-module gaps, and current and historical failure dispositions.

## First remaining task: disposition historical failures

The historical wheel failure lost its subprocess stdout/stderr because
`CalledProcessError` does not include them in its default exception message.
The wheel build helper now includes the command, exit, stdout and stderr in its
pytest failure message, which also flows into JUnit. Two regressions exercise
dependency-install and wheel-build failures. Both fail against the old helper;
[diagnostic-red.txt](diagnostic-red.txt) retains that baseline. The repair preserves
the failing outcome and makes its cause available to subsequent investigations.

An [offline wheel build](offline-wheel.txt) succeeds on Python 3.14.10 with local
setuptools 84.2.1, using `pip wheel --no-deps --no-build-isolation --no-index`.
[offline-wheel-command.json](offline-wheel-command.json) records argv, revision,
interpreter and exit. This rules out a universal current-tree wheel-build failure,
but does **not** establish the cause of PR #2475's historical Python 3.12 build
failure. The slow installed-wheel test remains excluded as required.

SciPy is absent in this interpreter. The real optimizer success test consequently
skips; the two existing import-fallback tests pass. No claim is made that this
establishes the historical optimizer assertion's cause. A compatible SciPy/NumPy
integration run and the supported-Python installed-wheel test remain necessary
before conclusively dispositioning all four historical failures or closing #2336.

The focused command below passes **56 tests**, with **1 skipped, 1 deselected**;
the named outcomes are in [regressions.xml](regressions.xml) and console output in
[regressions.txt](regressions.txt). Its two added passing nodes cover the diagnostic
repair. Before edits, the packaging/schema/sleeve subset passed 31 tests with one
skip and one deselection. No fresh full-suite coverage capture is claimed.
Black formatted the changed Python file, then the required whole-tree check passed
for all 534 files. Ruff and source/Markdown whitespace checks also pass.
[verification.json](verification.json) records the checks and input hashes;
[manifest.json](manifest.json) binds the retained evidence bytes.

```sh
python -m pytest tests/test_dashboard_sample_data.py tests/test_sleeve_suggestor.py tests/test_schema_export.py tests/test_cli_png_failures.py tests/test_cli_pdf_failures.py tests/test_cli_sweep_png_export.py -q -m 'not slow' --junitxml=docs/evidence/issue-2336-png-failures/disposition-20261104/regressions.xml
```

[png-audit.json](png-audit.json) records an independent verification of all 41
retained archive members and eight exact-node assertion RED/restored GREEN pairs.
The original, restored, GREEN, caller and main CLI SHA256 values all match
`d979ab73274b096be0f5f0f90edf7644550aba09ac8b4d3b50b058780dba567e`.
This is an audit of existing executions; PDF and next-boundaries evidence is untouched.

## Reconciled checklist

- [x] Completed bounded coverage capture, counts, module totals and largest gaps.
- [ ] Conclusively disposition all four historical failures and native PNG timeout.
- [x] Focused PNG/PDF/sweep run and eight PNG mutation pairs pinned to main's CLI.
- [ ] Post evidence and close #2336 after the historical dispositions are complete.
- [x] Completed coverage JSON committed with exact percent, command and source revision.
- [x] README numerator and denominator match coverage JSON exactly.
- [x] Eight assertion RED/restored GREEN pairs and matching main production hash.
- [ ] Establish a cause or fix for every historical failure as well as current failures.
- [x] Product coverage is at least 90%.

The same statuses apply to repeated copies of these tasks in the PR body. A PR-body
update was attempted after reconciliation. Automatic approval review rejected it:
`MCP tool call requires approval, but approval policy is never`. Remote checkboxes
remain unchanged. No evidence comment was posted and no issue was closed.

The checkout's Git metadata is read-only (`git add --dry-run` cannot create
`index.lock`). The delivery commit is created using temporary Git metadata in
`/tmp`; checkout edits remain available for automation to collect. That commit
does not advance this checkout's branch or publish changes to GitHub.
