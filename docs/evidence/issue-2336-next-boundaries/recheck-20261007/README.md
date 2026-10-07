# CLI warning lifecycle acceptance recheck

The existing warning test verified double installation and hook/handler restoration, but did not exercise reuse after uninstall. It now verifies that logging and Python warnings stop entering the collector after uninstall, that double installation works again for a second run, that prior records remain intact, and that both warnings forward exactly once through the original hook. The eight named cases are unchanged. Production remains byte-identical to the PR base.

## Verified acceptance criteria

- [x] **Tests**
  - [x] Added focused coverage for CLI error handling, warning collection, snapshot isolation, JSON formatting, and invalid UTF-8 configuration.
  - [x] The new tests pass, with no increase in failures or errors.
- [x] **Documentation**
  - [x] Added supporting evidence comparing baseline and candidate test results, including coverage measurements and verification records.

The focused file passes all eight cases on Python 3.14.7. The matched related diagnostic suite passes the same 27 named cases before and after this change, with zero failures, errors, or skips. Its targeted CLI coverage is unchanged at 652/1106 statements (58.951175%, displayed as 59%), with 17 excluded lines and no executed-line regression. These limited runs measure the selected diagnostics, not the entire application's coverage. Exact commands, interpreter, exits, elapsed times, source/test hashes, coverage JSON, JUnit, and console captures are retained here; `comparison-current.json` records the verified comparison.

The related command is:

```sh
python -m pytest tests/test_cli_observability_boundaries.py tests/test_cli_edge_cases.py tests/test_cli_packet_diff.py tests/test_cli_exception_handling.py -q -m 'not slow and not live_llm' --cov=pa_core.cli --cov-report=term-missing --cov-report=json:<capture> --junitxml=<capture>
```

`mutation-recheck/controls.json` records all eight original production mutations against the strengthened file: each explicit named node has exactly one failure/exit 1, then one pass/exit 0 after byte-identical restoration. Errors and skips are rejected. `reinstall-mutation/controls.json` adds a mutation that removes the installed-flag reset in `uninstall`: the original test passes the broken production behavior, the strengthened test fails it, and the strengthened test passes restored production. Every phase retains its console and named-node JUnit capture. Replay script sources are retained as gzip; the eight-case runner differs from the original only by using the current working directory as the repository root and excluding slow/live tests explicitly.

All 106 original manifest entries, including decoded gzip hashes, were verified before this work. The historical full-suite JUnit and coverage reports independently match `../comparison.json`: baseline 1594 passed versus candidate 1602 passed; both have one skip and zero failures/errors. Coverage increases from 12133 to 12140 covered statements out of 14553, across the same 186 source files and 181 excluded lines. Per-file statement/exclusion counts match and no executed lines regress. All eight historical RED/GREEN controls and restoration hashes also verify. `historical-verification.json` records these checks; the original evidence package is unchanged. Its manifest also anchors the original test-file hash; after this deliberate test change, that historical entry is verified against the `3b934060` Git blob, while the new manifest anchors the strengthened file.

## Environment limits and intermediate attempts

This runner's attempted full baseline stalled inside Kaleido PNG export in `tests/test_cli.py::test_main_with_png`, before the changed case ran. It was interrupted with exit 130 after the 120-second faulthandler trace identified the stalled export. The incomplete console and receipt are retained; they provide no complete full-suite coverage or JUnit result. The historical full-suite measurements used the documented Python 3.12 environment and are not represented as a fresh full-suite result for these strengthened assertions. The broad coverage initiative remains open.

The first strengthened related-suite run failed because the assertion expected six forwarded arguments after uninstall. The restored original hook receives the four arguments supplied directly; only the installed collector shim supplies the two optional arguments. The expectation was corrected, and the final 27-case run passes. Both attempts are retained.

The first whole-repository Black check could not start its default process workers because the sandbox denied the forkserver socket. A single-worker retry timed out after 180 seconds; a diagnostic trace showed its idle worker and an event loop waiting for a thread completion wakeup. Black's normal single-file `reformat_one` check was then run sequentially on the exact 486 files returned by its `get_sources` with the required command's scope and repository configuration. All were unchanged, and this populated Black's normal cache. The literal required command then passed with `BLACK_NUM_WORKERS=1` and that temporary cache. The sequential checker source, all-source hashes, diagnostic, failed attempts, final exit-0 gate, and focused Ruff result are retained. No formatter scope, configuration, or source bytes changed.

`manifest.json` hashes every stored file and, for gzip artifacts, its losslessly decoded contents. Captures refer to their original temporary paths; decode with Python gzip or `gzip -dc` without altering the preserved originals. PR #2341 was verified open and ready for review through the GitHub connector (`pr-status.json`); direct shell API access was unavailable.

CLI source SHA256: `265d1891f50333a994458086cf0c4260d580971c710ea181f99e68582e5c899f`.
