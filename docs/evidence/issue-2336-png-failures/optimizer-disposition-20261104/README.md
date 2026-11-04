# Distinguish optimizer nonconvergence from an unavailable dependency

Two new parameter cases in
`tests/test_sleeve_suggestor.py::test_feasible_optimizer_result_distinguishes_convergence_from_grid_fallback`
exercise the existing solver handoff with a known feasible optimum. The tests
retain the production linear surrogate, constraints, allocation evaluation and
grid fallback. The injected solver verifies its objective, constraints, bounds,
initial allocation and iteration limit. A successful result returns 80/0/20;
a nonconverged result must use the requested grid and return 50/0/50, with status
`grid_fallback:optimizer_failed`. The existing missing/incompatible import cases
instead report `grid_fallback:missing_scipy`.

This advances the first remaining task, historical failure disposition. It
establishes distinct observable failure paths without requiring SciPy, but does
not determine which path produced PR #2475's uninformative `assert False`.
Production code is unchanged. This is not a compatible-SciPy integration witness.

## Verification

- [optimizer.txt](optimizer.txt) and [optimizer.xml](optimizer.xml): **22 passed,
  1 skipped** for the full optimizer test module. The existing real-SciPy case
  skips because SciPy is absent. Targeted module coverage is **56%**; no module
  coverage target is claimed or marked complete.
- [regressions.txt](regressions.txt) and [regressions.xml](regressions.xml):
  **80 passed** across PNG/PDF/sweep exports and capture/summary/replay helpers.
- [nonconvergence-red.txt](nonconvergence-red.txt) and
  [nonconvergence-red.xml](nonconvergence-red.xml): removing the production
  convergence guard in an isolated writable clone gives **1 failed, 1 passed**.
  The failure is an assertion in the new nonconvergence case.
- [nonconvergence-green.txt](nonconvergence-green.txt) and
  [nonconvergence-green.xml](nonconvergence-green.xml): restoring byte-identical
  production source gives **2 passed**. The main checkout was never mutated.
- [black-check.txt](black-check.txt): the required whole-tree Black check passes
  for **536 files**. A first attempt was interrupted because worker IPC hangs
  in this sandbox; serial checks primed a writable cache before the successful
  exact check. [ruff-check.txt](ruff-check.txt) passes.

[checks.json](checks.json) records argv, outcomes, interpreter and source hashes.
All pytest commands exclude slow tests. No tests were disabled or newly skipped.

## Reconciled main evidence

GitHub's main commit was independently checked as
`06881dbdf0e915c305669e7fe3f8baa7d8a21ec4` through the GitHub connector.
[verification.json](verification.json) records a fresh audit of the existing
main coverage manifest, JSON/JUnit reconciliation and eight retained PNG pairs.

The completed main capture remains exactly **13,493 / 14,626 covered statements
(92.25352112676056%)**, with **1,988 passed, 2 failed, 3 skipped**, and four
documented deselections. Its [command, SHA and test overlay](../config-guard-20261104/README.md)
remain unchanged; this round does not claim another full-suite run. Per-module
coverage and largest gaps remain in that capture's summary. The later candidate
capture remains **2,040 passed, 1 failed, 3 skipped**, after the entrypoint repair.
Historical PR #2475's raw JSON is **13,537 / 14,670 (92.27675528289025%)**;
its prose numerator of 13,441 was incorrect.

Every retained PNG RED fails on an assertion, every GREEN passes after restoration,
and all restored CLI hashes match current main's
`d979ab73274b096be0f5f0f90edf7644550aba09ac8b4d3b50b058780dba567e`.
The audit also verifies decompressed mutant hashes and the named JUnit nodes.
This is an audit of existing same-main receipts, not a fresh mutation replay.
PDF and next-boundaries evidence are untouched.

## Outstanding failure dispositions

| Test | Evidence and remaining uncertainty |
| --- | --- |
| `tests/test_cli.py::test_main_with_png` | Existing 30-second subprocess timeout; the retained browser probe establishes sandbox Chromium startup restrictions. Environmental. |
| `tests/test_entrypoints.py::test_console_scripts_work_in_clean_venv` | Historical unsupported Python 3.12 violates the package's Python >=3.13 requirement. The current network prerequisite was removed in the preceding repair; its seven entrypoint tests pass. |
| `tests/test_schema_export.py::test_parameter_dictionary_matches_schema` | Historical Python 3.12 type rendering differed; the unchanged test passes on supported Python. Environmental. |
| `tests/test_dashboard_sample_data.py::test_wheel_install_exposes_bundled_dashboard_samples` | Historical build stderr was discarded. Unsupported Python is a plausible cause, but the exact cause remains unresolved. Supported-Python offline wheel verification passes; the slow online test remains deselected. |
| `tests/test_sleeve_suggestor.py::test_suggest_sleeve_sizes_optimize_prefers_best` | Historical assertion and SciPy/NumPy warning do not identify import failure versus solver nonconvergence. The new tests distinguish these paths, but compatible-SciPy execution remains unavailable here. |

The verified checklist is still **14/18** (including repeated task/acceptance
items). No additional acceptance checkbox is marked complete. #2336 remains
open pending the two unresolved historical dispositions and final publication.
PR #2478 was verified open and ready for review.

Automatic approval review rejected the PR-body update with
`MCP tool call requires approval, but approval policy is never`. The concrete
reconciled body is preserved at `/tmp/issue2336-optimizer-pr-body.md`; the remote
body remains unchanged. The checkout's `.git` is read-only, so the test and
evidence change is committed in a separate writable clone; the commit is
preserved in `/tmp/issue2336-optimizer.bundle` for handoff.
