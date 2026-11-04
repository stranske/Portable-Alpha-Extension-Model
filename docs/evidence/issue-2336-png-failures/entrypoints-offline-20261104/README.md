# Clean-venv entrypoints without a network prerequisite

The existing `tests/test_entrypoints.py::test_console_scripts_work_in_clean_venv`
now passes on supported Python without downloading setuptools. It builds a wheel
with the test interpreter's installed backend, then installs that wheel with
`--no-index --no-deps` into the same clean venv. The original `pa --help` and
`pa-dashboard --help` checks still execute outside the source tree, with
`PYTHONPATH` removed. The venv does not inherit system packages. Build and install
failures now retain stdout/stderr in assertion messages.

[before.txt](before.txt) and [before.xml](before.xml) reproduce the original
failure: DNS resolution of `pypi.org` fails during the setuptools bootstrap,
before either console script runs. [after.txt](after.txt) and [after.xml](after.xml)
record **7 passed** for the entire entrypoint module after the test repair.
No production code, dependency pins, workflows, PDF evidence or next-boundaries
evidence changed. [entrypoints.patch](entrypoints.patch) is the exact test diff.

## Completed candidate verification

The bounded candidate suite reports **2,040 passed, 1 failed, 3 skipped,
4 deselected**, exit 1, in 188.85 pytest seconds. Coverage is exactly
**13,493 / 14,626 covered statements (92.25352112676056%)**. The printed percentage
is retained in [candidate-summary.txt](candidate-summary.txt); the raw JSON is
losslessly compressed in [candidate-coverage.json.gz](candidate-coverage.json.gz).
[candidate-summary.json](candidate-summary.json) validates the JSON totals against
every module and retains the largest gaps and the failing node.

[candidate.txt](candidate.txt), [candidate.xml](candidate.xml), and
[candidate-command.json](candidate-command.json) retain console output, named-node
JUnit, argv, input hashes, revisions, interpreter/tool versions and elapsed time.
The actual command uses absolute report paths:

```sh
timeout --signal=INT --kill-after=10s 600s python -m pytest --cov=pa_core --cov=dashboard --cov-report=json:candidate-coverage.json --junitxml=candidate.xml -m 'not slow and not live_llm' tests/
```

This measures candidate tests at PR head `202f6728` plus the retained test diff.
It does not replace the [main-only capture](../config-guard-20261104/main-capture/coverage.json).
GitHub main still resolves to `06881dbdf0e915c305669e7fe3f8baa7d8a21ec4`, verified
through the GitHub connector; [main-verification.json](main-verification.json)
records that identity. Product bytes match main. The candidate retains its
coverage 7.17.0 pin; main pins 7.17.1. Actual coverage is 7.17.0 in both captures.

[comparison.json](comparison.json) compares every testcase against the previous
completed candidate capture: the only changed existing outcome is the entrypoint
test, from FAIL to PASS; no node disappeared. Ten additional guard tests were
introduced in the preceding iteration, not in this repair.

## Failure dispositions and remaining work

| Node | Cause or disposition |
| --- | --- |
| `tests/test_cli.py::test_main_with_png` | The only candidate failure: the unchanged native renderer invocation exceeds its existing 30-second subprocess bound. The retained main browser probe establishes sandbox Chromium startup restrictions. Environmental; no native PNG success is claimed. |
| `tests/test_entrypoints.py::test_console_scripts_work_in_clean_venv` | The current failure was an unnecessary network prerequisite in this test, now removed and verified. PR #2475 instead failed the package's Python requirement on unsupported Python 3.12.2. |
| `tests/test_schema_export.py::test_parameter_dictionary_matches_schema` | Historical Python 3.12 schema type formatting differs. The unchanged test passes in both the supported-Python main capture and this candidate run. |
| `tests/test_dashboard_sample_data.py::test_wheel_install_exposes_bundled_dashboard_samples` | PR #2475 lost wheel-build stderr, so its exact historical cause remains unresolved. This slow online test remains deselected by the required marker. The separate existing offline wheel installation test passes in this candidate run. |
| `tests/test_sleeve_suggestor.py::test_suggest_sleeve_sizes_optimize_prefers_best` | Historical `optimizer_success=False` does not distinguish an incompatible SciPy import from nonconvergence. SciPy is absent here, so the existing test skips. The new download attempt in [scipy-download.txt](scipy-download.txt) finds no available distribution in this environment. A compatible optimizer execution remains outstanding. |

The other two skips require absent `langchain_anthropic`. The four deselections
are the slow online wheel test and three live-provider tests. No test was newly
skipped or disabled by this repair.

## Reconciled checklist

[verification.json](verification.json) revalidates the committed main coverage
manifest and all eight retained PNG assertion RED/restored GREEN pairs, including
named-node JUnit and decompressed mutant hashes. Every original/restored/GREEN
CLI hash equals main's
`d979ab73274b096be0f5f0f90edf7644550aba09ac8b4d3b50b058780dba567e`.
This round validates existing receipts; it does not claim a fresh mutation replay.
PR #2475's JSON remains 13,537 / 14,670; its prose numerator 13,441 was incorrect.

The modified Python file was run through Black at line length 100. The required
whole-tree `black --check --line-length 100 --exclude '(\.workflows-lib|node_modules)' .`
passes for **536 files**, and Ruff passes. Two initial whole-tree attempts did
not finish; individual checks then primed a writable cache before the successful
whole-tree check. [black-check.txt](black-check.txt), [black-prime.txt](black-prime.txt)
and the initial attempt logs retain those outcomes. [manifest.json](manifest.json)
binds the retained evidence artifacts.

- [x] Completed bounded main coverage measurement, counts, module totals and gaps.
- [x] Committed main coverage JSON, printed percentage, command and revision.
- [x] README covered/total counts equal the raw coverage JSON.
- [x] Focused export run and eight same-main PNG mutation pairs retained and verified.
- [x] Every current failing test is named with a cause or fix.
- [x] Product coverage exceeds 90%.
- [ ] Conclusively disposition the historical optimizer and wheel-build failures.
- [ ] Publish final acceptance evidence and close #2336 after those dispositions.

These correspond to **14/18** checkboxes in the task prompt (which repeats tasks
and acceptance criteria). Automatic approval review rejected the PR-body update:
`MCP tool call requires approval, but approval policy is never`. The replacement
body is preserved at `/tmp/issue2336-entrypoints-pr-body.md`; the remote checklist
remains unchanged. PR #2478 was verified open and ready for review.

The checkout's `.git` is read-only. The repair is committed in a separate writable
clone and retained in `/tmp/issue2336-entrypoints.bundle`, with delivery metadata
at `/tmp/issue2336-entrypoints-delivery.json`. This preserves a reviewable commit
without advancing the checkout branch or publishing it.
