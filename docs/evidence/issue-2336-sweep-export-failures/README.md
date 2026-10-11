# Sweep packet export failure boundaries

Related to #2336. Production code is unchanged at `18ac626ed7af301083edca2074b009a5ac3544cd`. Six new named CLI cases exercise runtime, ImportError, ModuleNotFoundError, ValueError, TypeError and KeyError export failures. They assert one export attempt, correct diagnostic/remediation, no success announcement, and a preserved sweep workbook and manifest. Export/rendering itself is stubbed; no real PowerPoint, deployment or physical export claim.

Last500 source-touching commits rank CLI first:21 repair-subject proxy matches,133 touches,297 uncovered statements. These are subject proxies, not verified escaped incidents. Complete ranking/raw history is in the archive.

## Matched full-suite evidence

The same isolated Python3.12.2 interpreter/environment ran `python -m pytest --cov=pa_core --cov=dashboard --cov-report=json:<capture> tests/ --junitxml=<capture>`. Baseline1643PASS/1SKIP/2default-deselected; candidate1649PASS with the same skip/deselections. Exactly six new nodes and zero changed old outcomes. Both commands exit0. Coverage12149→12159/14554 statements (83.47533324172049→83.54404287481105%), identical186source files/181excluded lines, no covered-line regression. CLI810→820/1107; new lines[1487, 1488, 1489, 1490, 1491, 1492, 1493, 1494, 1495, 1496]. No floor, test selection, source scope or exclusion changed. The NumPy/pandas/xarray overlay and complete raw warnings are retained; this is not hosted-environment parity.

## Reproduce

With Python3.12+ and repository test dependencies:

```sh
python3 -m pytest tests/test_cli_sweep_export_failures.py -q -o addopts= -m "not slow"
python3 docs/evidence/issue-2336-sweep-export-failures/replay_sweep_export_failures.py --output /tmp/new-sweep-export-proof
python3 -m pytest --cov=pa_core --cov=dashboard --cov-report=json:coverage.json tests/ -m "not slow and not live_llm"
```

Replay extracts an exact committed Git snapshot, overlays the named test into a private temporary tree, narrows one of three sweep exception handlers to FloatingPointError, and executes all six named cases separately. Every real mutation execution returnsRED1; byte-identical restored source returnsGREEN0 with exact one-node JUnit validation. Three distinct mutant shapes, six named RED/GREEN executions. Caller source/test hashes are unchanged and private trees are cleaned. Reusing an output directory fails before mutation and preserves existing evidence hashes.

`validation.tar.gz` retains complete full/focused/mutation logs, JUnit, coverage, argv/cwd/process exits, source/test/replay snapshots, ranking, and reuse proof. `manifest.json` binds every archive member and original input snapshot. Historical full Black482files, focused Ruff and diff checks pass. Broader source2336 remainsOPEN below90%; no providerPASS or merge authorization. Matchingkeeper owns hosted checks/reviews; closer requires full exact-head topology, zero active threads, seven-minute review floor, guarded merge and actualcompare/bounded disposition.

## Acceptance validation follow-up

The six existing cases now reject conflicting remediation and duplicate failure
diagnostics. Each also checks that the workbook bytes survive the failing export,
the final manifest remains valid JSON, and its config, seed, substream IDs, CLI
arguments and data hashes match the manifest supplied to the exporter. The hashes
are checked against the actual config, index and preserved workbook. Timing,
warnings and cost may legitimately change during finalization.

The full-suite comparison above is historical evidence for the original test
snapshot, not a new full-suite run of these stronger assertions. The original
archive and manifest are preserved: their `inputs_sha256` entries bind the
archived `pa-production.py`, `pa-test.py` and `pa-replay.py` snapshots. The test
and replay hashes consequently differ from the current files. The replay helper
now explicitly selects `-m "not slow"`; production remains byte-identical.
The reproduction command above also excludes slow tests, so its selection need
not match the historical default-selection counts.

Current receipts are in `acceptance-validation.tar.gz`, with archive, member and
current-input hashes in `acceptance-manifest.json`. They retain the focused test
logs/JUnit and six individual real handler mutations followed by restored passes.
Export and rendering remain stubbed; these checks do not demonstrate physical
Office exports or hosted-environment parity.

On Python 3.14.8, all **15** focused cases (the six failure cases plus nine existing
sweep-export cases) pass; the six mutations each fail their exact node and each
restoration passes. Reusing the output directory fails before mutation, preserving
both the receipts and caller files. Focused Ruff and full Black pass, with **493**
current Python files checked. The original 482-file Black receipt belongs to the
historical snapshot.

Targeted coverage of the six failure cases confirms execution of all ten sweep
handler lines, 1487–1496. Its CLI-only total is 361/1107 statements (32.61%,
displayed as 33%); that deliberately narrow run is not comparable to the
historical full-suite CLI or broader source percentages. Reproduce with
`python3 -m pytest tests/test_cli_sweep_export_failures.py --cov=pa_core.cli --cov-report=term-missing -q -o addopts= -m "not slow"`.

Black's multiprocessing mode fails on a sandbox socket, and its thread mode
hangs here. The successful full check uses the unchanged Black CLI discovery,
configuration and `reformat_one` formatter, with sequential worker scheduling.
The temporary adapter, command, exit status and full output are archived. Reproduce
the focused run and the sandbox-compatible formatting check with:

```sh
python3 -m pytest tests/test_cli_sweep_export_failures.py tests/test_cli_sweep_png_export.py -q -o addopts= -m "not slow"
mkdir -p /tmp/sweep-black-repro
tar -xzf docs/evidence/issue-2336-sweep-export-failures/acceptance-validation.tar.gz -C /tmp/sweep-black-repro black-runtime/sitecustomize.py
PYTHONPATH=/tmp/sweep-black-repro/black-runtime black --check --line-length 100 --exclude '(\.workflows-lib|node_modules)' .
```

For the historical comparison, the archive retains the precise baseline/candidate
argv, working directories, Python 3.12.2 environment notes and source/test snapshots.
Use the baseline revision named above in an isolated checkout and overlay the
archived `pa-test.py` as `tests/test_cli_sweep_export_failures.py` for the candidate.
The recorded default selection includes slow tests; the bounded commands in this
document exclude them, so they are not a claim of identical historical totals.

- [x] **Bug Fixes**: verified the bounded sweep-export regression coverage.
- [x] Added six failure cases checking clear diagnostics, no packet-success
  announcement, and preservation of the sweep workbook and manifest.
- [x] **Documentation**: included comparisons, reproduction steps and receipts.
- [x] Added baseline/candidate test and coverage evidence, with validation artifacts.
- [x] Documented the remaining limitation: broader measured source coverage is
  **83.5440%, below 90%**. Issue #2336 remains open; these local receipts establish
  neither hosted-environment parity nor authorization to merge.
