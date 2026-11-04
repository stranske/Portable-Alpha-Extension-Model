# Bounded coverage capture with recorded inputs

The completed run reports **2,029 passed, 2 failed, 3 skipped, 4 deselected**, exit 1,
and **13,493 / 14,626 covered statements (92.25352112676056%)**. These totals equal
[coverage.json](coverage.json) exactly. [summary.json](summary.json) records every
module's coverage, missing line numbers, the 15 largest gaps, and named failures.
[full-suite.txt](full-suite.txt) and [full-suite.xml](full-suite.xml) retain the raw
console and JUnit results. The outer 600-second timeout was not reached: elapsed
time was 220.85 seconds, including report generation.

Main was verified through GitHub as `06881dbdf0e915c305669e7fe3f8baa7d8a21ec4`.
The checkout head was `895d15a36281d774e76bd5a6f91b3c2b2be76a28`. All tracked product
files in `pa_core/` and `dashboard/` match that main revision, and neither directory
has untracked product files. This is a candidate-test run on current-main production;
it is not an untouched-main test run. [tests.patch](tests.patch) retains the complete
test overlay relative to main, including the inherited 30-second native PNG child
process bound and seven new capture tests. [command.json](command.json) records the
resolved revision, interpreter, argv, CLI hash, configuration and driver hashes,
and hashes of all test inputs. All input hashes were rechecked after completion.

Reproduce with GNU `timeout` available:

```sh
python docs/evidence/issue-2336-png-failures/capture.py --revision origin/main --timeout 600 --output <fresh-directory>
```

The driver runs `python -m pytest --cov=pa_core --cov=dashboard` with coverage JSON
and JUnit paths inside the fresh directory, followed by
`-m 'not slow and not live_llm' tests/`. This satisfies the required slow-test
exclusion and preserves the repository's default live-provider exclusion. Python
is 3.14.9, pytest 9.2.0, pytest-cov 7.1.0 and coverage 7.17.0. The installed coverage
version differs from main's 7.17.1 development pin; the measurement is identified
with its actual tool version.

| Largest source gap | Missing statements |
| --- | ---: |
| `pa_core/cli.py` | 165 |
| `pa_core/sleeve_suggestor.py` | 115 |
| `dashboard/pages/3_Scenario_Wizard.py` | 101 |
| `dashboard/pages/5_Scenario_Grid.py` | 57 |
| `pa_core/sim/paths.py` | 42 |

Both failing tests are retained as failures:

- `tests/test_cli.py::test_main_with_png`: the native CLI/renderer child exceeds
  its 30-second bound. This reproduces the runner-specific timeout in earlier
  captures. The test remains failed; the capture does not prove the renderer's
  root cause or justify a product-code change.
- `tests/test_entrypoints.py::test_console_scripts_work_in_clean_venv`: pip cannot
  resolve `pypi.org` to install setuptools in the clean environment. The console
  preserves all five DNS retries. This is an environmental network failure before
  installation of the product.

The historical four PR #2475 failures and numerator reconciliation remain in the
[retained reconciliation](../main-20261104/pr2475-reconciliation.json) and
[failure dispositions](../main-06881dbd-20261104/README.md). In particular, the
historical optimizer and wheel-build root causes are still unresolved. Current
environmental limitations do not establish those historical causes. #2336 remains
open despite product coverage exceeding 90%.

The new `capture.py` refuses product drift and existing output directories, records
inputs before launching pytest, retains tracked and untracked test changes, and
uses a separate coverage data file for each capture. Only a completed run accepted
by the existing coverage/JUnit reconciler gets `completed=true` and `summary.json`.
Both graceful timeout and forced-kill exits are rejected even with partial reports.
Seven capture tests pass, including a real stalled executable terminated by the
outer timeout. [regressions.txt](regressions.txt) and [regressions.xml](regressions.xml)
retain **47 passing** capture/summary/replay regressions. The required targeted
[tooling coverage](tooling-coverage.txt) reports the capture helper separately at
85%; the 90% acceptance target measures product code. The whole-tree
[Black check](black-check.txt) passes for all 536 Python files. Black's cache was
primed sequentially in `/tmp` because this sandbox blocks multiprocessing sockets
and makes the default cache directory read-only; the required CLI check then passed.
Ruff also passes for both new Python files.

- [x] Bounded completed coverage capture on current-main production, with disclosed
  test overlay, exact totals, per-module gaps, command and revision.
- [x] README numerator and denominator equal the retained raw coverage JSON.
- [ ] Conclusively classify every historical failure and close #2336.

The eight previously retained PNG restoration hashes still equal main's CLI SHA256
`d979ab73274b096be0f5f0f90edf7644550aba09ac8b4d3b50b058780dba567e`.
No new mutation run is claimed. Existing PNG, PDF and next-boundaries evidence keeps
its original execution identity. PR #2478 was verified open and ready for review.
The checkout's Git metadata is read-only; the delivery commit, patch and bundle
are prepared separately under `/tmp`, rather than advancing the checkout branch.
