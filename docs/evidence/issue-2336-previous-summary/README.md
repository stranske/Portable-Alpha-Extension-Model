# Previous-summary fallback evidence for issue 2336

The real CLI sweep-to-export packet path now covers four reader failures: missing Excel engine, I/O error, permission error and malformed workbook. Export still receives the previous manifest, and an empty previous-summary frame with no invented metric columns. The available-workbook control remains intact. Simulation and export-rendering are stubbed by the existing fixture; this is CLI contract coverage, not a real Excel-engine or rendered PowerPoint acceptance claim. Production code is unchanged.

## Current full-suite comparison

Both linked checkouts start at `994403190daac95bdaf11fad2b1661589d4a9622` and use the same interpreter/environment and literal command `python -m pytest --cov=pa_core --cov=dashboard --cov-report=json:<capture> tests/ --junitxml=<capture>`. Baseline: 1619 passed, one skip and two default deselections. Candidate: 1623 passed, the same skip and default deselections. All 1620 baseline testcase outcomes are retained after mapping the original workbook control to its `[available]` parameter name; the four new named nodes pass. Full console, JUnit, exact commands/process exits, source inventories and coverage JSON are retained losslessly in `raw-proof.tar.gz`; `manifest.json` binds every complete member. `comparison.json` exposes the measured totals and exact new nodes directly.

Coverage: 12147/14554 (83.461591315%) -> 12149/14554 (83.475333242%). Same 186 source files and 181 excluded lines. CLI lines 1014 and 1015 newly execute; no covered line is lost anywhere. Only the test file differs between source/test/config inventories. The broader initiative remains open below 90%.

## Actual production mutations

`pa-mutations.py` in the archive temporarily replaces only the previous-summary exception handler with `except RuntimeError`. Each new exact pytest node fails (RED exit 1), then passes (GREEN exit 0) after byte-identical restoration. Missing-engine and malformed-workbook cases expose the uncaught original exception; I/O and permission cases expose loss of the previous manifest through the outer handler. All eight raw consoles and named-node JUnit files plus `mutations.json` are directly readable beside this README. The caller test hash remains unchanged throughout. The final source hash is 95f0e38b1e2491ff23fe4f278c2d5c5d29a6293bfbf5ef14b64795a4c954906e.

## Selection and limits

CLI ranks first in the current last-500 source-touch history: 27 repair-subject proxy matches, 133 touches, 299 baseline uncovered statements. This is a repair-subject heuristic, not verified escaped-defect history; `ranking.json` and the archived complete history disclose the ordering. Existing earlier CLI chunks are retained and not duplicated.

The isolated Python 3.12.2 environment uses pytest 9.1.1, coverage 7.16.0, NumPy 2.5.0, pandas 3.0.3 and xarray 2026.9.0. It is not CI parity (repository coverage pin 7.16.2); exact tool versions are in `pa-environment.json`. Both full runs complete without changing floor, exclusions, source scope or default selection. Hosted checks, review threads, unchanged-head seven-minute floor and actual post-merge comparison remain keepalive/closer responsibilities; local success is not provider PASS or merge authorization.
