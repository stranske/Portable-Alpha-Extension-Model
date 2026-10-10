# Issue 2336: CLI validation boundaries

This partial coverage increment protects failure diagnostics, severity policy, and effective financing/simulation settings in `pa_core/cli.py`. It does not complete the initiative's 90% target.

## Selection and scope

Base: `c569018fda1433c0696febbb498ff2b522616fac`. Among the last 500 source-touching commits, CLI ranked first by repair-subject proxy (27 matches, 133 touches, 303 uncovered lines); Scenario Wizard had 23/65/338 and config 18/53/41. This proxy is a selection heuristic, not verified escaped-defect history. The full ranking and history are archived.

The five new pytest cases invoke production `cli.main --validate-only`. They protect an exception's original message/details and logging/warning restoration, warning/info/error exit policy, and exact effective financing/simulation keyword forwarding. The fixture injects an already validated configuration and bypasses facade option revalidation so failure injection reaches the CLI validation pass. This is a CLI boundary proof, not an end-to-end config-loading or financing-model proof. Production bytes and coverage policy are unchanged.

## Full-suite comparison

Both runs used the same command, interpreter, source denominator, exclusions, and default pytest selection: `python -m pytest --cov=pa_core --cov=dashboard --cov-report=json:<capture> tests/ --junitxml=<capture>`.

- Baseline: 1,614 passed, 1 skipped, 2 default deselected, exit 0.
- Candidate: 1,619 passed, 1 skipped, 2 default deselected, exit 0.
- Every existing JUnit testcase outcome is unchanged; exactly five tests are added.
- Coverage: 12,143/14,554 (83.434107%) -> 12,147/14,554 (83.461591%). Same 186 measured files and 181 excluded statements. Four newly covered CLI lines (820, 821, 1047, 1048); no lost covered lines.

The interpreter is an existing isolated Python 3.12.2 environment with pytest 9.1.1 and coverage 7.16.0 (repository pin is 7.16.2), NumPy 2.5, pandas 3.0.3, and an xarray 2026.9 overlay. Both suites emit the same 465 warnings, including native ABI/deprecation warnings. These are local paired receipts, not a claim of hosted environment parity. `pa-environment.json` records versions. The early `pa-focused.xml` contains an invalid initial fixture setup and is retained for transparency; `pa-focused-final.xml` is the corrected 5-pass result.

## Deliberate production mutations

`pa-mutations/controls.json` records five actual source mutations in an isolated copy: drop exception details; reject warning; reject info; accept error; discard financing term. Each exact new testcase yields RED (exit 1, one intended AssertionError, no setup error/skip), then GREEN (exit 0) after byte-identical restoration. Every phase has raw console/JUnit, argv, timing, source hashes; the caller's CLI/test bytes remain unchanged. The capture script is archived for audit, not installed as a new generic replay product. This proof establishes these five boundaries only; it is not global mutation adequacy or crash-safe restoration.

## Evidence integrity

`raw-proof.tar.gz` contains lossless paired coverage JSON, JUnit, console output, process metadata, source inventories, ranking, environment, and mutation captures. `manifest.json` hashes every member, archive, and the originally bound source/test/config. `comparison.json` summarizes unchanged outcomes, denominator/exclusions, and exact gained lines. Archive extraction is sufficient to inspect evidence; its capture scripts retain original absolute paths and require explicit adaptation before rerunning elsewhere.

The [acceptance revalidation](acceptance-revalidation-20261010/README.md) strengthens the same five test cases, verifies all 41 original archive members and the historical comparisons, and binds fresh focused and mutation receipts to the updated tests. The original full-suite captures remain historical evidence; the supplemental report records the current runner's PNG export timeout separately.
