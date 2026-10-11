# Sweep packet export failure boundaries

Related to #2336. Production code is unchanged at `18ac626ed7af301083edca2074b009a5ac3544cd`. Six new named CLI cases exercise runtime, ImportError, ModuleNotFoundError, ValueError, TypeError and KeyError export failures. They assert one export attempt, correct diagnostic/remediation, no success announcement, and a preserved sweep workbook and manifest. The packet exporter is stubbed throughout. The current test also verifies its plotting stub; earlier snapshots could call the real plot builder because they did not patch the parent package attribute. No real PowerPoint, deployment or physical export claim.

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

Receipts for that follow-up are in `acceptance-validation.tar.gz`, with archive, member and
input-snapshot hashes in `acceptance-manifest.json`. They retain the focused test
logs/JUnit and six individual real handler mutations followed by restored passes.
The packet exporter is stubbed; the earlier plotting stub could be bypassed
as explained below. These checks do not demonstrate physical Office exports
or hosted-environment parity.

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

## Independent manifest snapshot validation

The six cases now read the on-disk manifest inside the failing exporter, verify
that it matches the exporter argument, and retain that independent snapshot.
After finalization they compare every field except `run_timing`, `warnings` and
`cost`, which finalization legitimately updates. This also protects provenance,
backend and data-quality fields that the earlier six-field comparison omitted.
Production and the six named nodes remain unchanged.

`snapshot-validation.tar.gz` and `snapshot-manifest.json` bind the latest test
snapshot, targeted coverage, focused JUnit, Black output, and mutation receipts.
The preceding two archives remain historical snapshots; their test input hashes
are not hashes of the current test file. All **15** focused cases pass. Targeted
CLI coverage remains **361/1107 (32.61%)**, with all ten handler lines covered;
this is not a new matched full-suite coverage measurement.

A private source mutation removes `backend` during manifest finalization. All
six parent cases pass with that mutation, all six strengthened cases fail on
their manifest comparison, and all six pass after byte-identical restoration.
The archived `verify_manifest_mutation.py` reproduces this control when run from
the repository root with its output directory set to a fresh location. The
existing handler replay also produces six exact-node failures and six restored
passes; output reuse is refused with caller files and receipts preserved.
Full Black passes for **493** files with the previously documented sequential
adapter; focused Ruff and `git diff --check` pass.

The PR-body reconciliation was attempted after reviewing commits `943d00cc`
and `60f1950e` and verifying their receipts. The GitHub update tool requires
approval, which this run's `never` approval policy blocks. The tests and
documentation criteria are verified locally; the live PR acceptance checkboxes
still need to be applied. Broader source coverage remains **83.5440%, below 90%**;
these local receipts establish neither hosted parity nor merge authorization.

## Rendering isolation and replay dependency validation

Review found that the autouse theme fixture imports the real `pa_core.viz`
package before these cases run. Patching only `sys.modules` left the real
package attached to `pa_core`, so the CLI's relative import could bypass the
plotting stub. The six cases now patch both locations and assert that the
exporter receives the exact stub figure. This corrects the earlier claim that
rendering was fully stubbed; the historical receipts and counts are preserved.

The replay helper uses standard-library ElementTree for JUnit generated by
its own local pytest subprocess, removing its undeclared `defusedxml` dependency.
With that import explicitly blocked, all six real handler mutations fail their
exact nodes and byte-identical restorations pass. Output reuse remains refused
before mutation, preserving both evidence and caller source/test bytes.

`review-validation.tar.gz` and `review-manifest.json` bind the current test/replay
snapshots, focused results, targeted coverage, mutation controls, and formatting
receipts. All **15** focused cases pass. The six failure cases alone cover
**361/1107 CLI statements (32.61%, displayed as 33%)**, including every handler
line 1487–1496. Full Black checks **493** files, focused Ruff passes, and the
production source remains unchanged. The clean committed checkout checks **482**
Python files; the workspace's additional 11 files belong to the local reference
pack and are also covered by its 493-file check.

The archived `verify_viz_isolation.py` runs in a private source tree pinned to
`81a9c7141839704ae004385e64453198383f1e53`, so its previous-test control remains
reproducible after the current change is committed.
It installs a guard that rejects calls to the real plot builder. The previous
test snapshot and the current test with its parent-package patch removed each
fail all six cases at that guard; the guarded current test and its byte-identical
restoration each pass all six. Reproduce from the repository root after extracting
the script to a temporary directory:

```sh
tar -xzf docs/evidence/issue-2336-sweep-export-failures/review-validation.tar.gz -C /tmp verify_viz_isolation.py
python3 /tmp/verify_viz_isolation.py --output /tmp/new-viz-isolation-proof
```

Raw historical JUnit and coverage were rechecked: 1643→1649 passes, one unchanged
skip, exactly six new nodes, zero changed old outcomes, and 12149→12159/14554
statements with no covered-line regression. These are historical measurements,
not a new full-suite run. Broader source coverage remains **83.5440%, below 90%**;
these local receipts establish neither hosted-environment parity nor merge
authorization. PR #2348 was verified open and ready for review. The live
acceptance-checkbox update was rejected again because the GitHub tool requires
approval and this run's policy is `never`; a reconciled PR body is retained
in the review archive for an authorized update.

## Exact diagnostic and remediation validation

The six cases now require one complete diagnostic line, including the exception
message, and the exact remediation line. Dependency guidance must name `plotly`,
`kaleido` and `openpyxl`; runtime failures must have no remediation line. Packet
success announcements are rejected on both stdout and stderr. Production remains
unchanged, and the existing workbook, manifest and rendering assertions remain.

`exact-guidance-validation.tar.gz` and `exact-guidance-manifest.json` retain the
current test, targeted coverage, focused JUnit, handler mutations, formatting
receipts and the reconciled PR body. All **15** focused cases pass. The six cases
cover **361/1107 CLI statements (32.61%, displayed as 33%)**, including all ten
handler lines. Six handler mutations fail their exact nodes and byte-identical
restorations pass; existing output is refused with receipts and caller preserved.
Full Black passes for **493** workspace files with the documented sequential adapter.

An additional private mutation removes `kaleido openpyxl` from the dependency
guidance. All six previous tests pass; the strengthened `import` and `module`
nodes fail specifically on remediation, and all six restored cases pass. Extract
the archived `verify_sweep_guidance.py` and run it from the repository root with
`python3 /tmp/verify_sweep_guidance.py --output /tmp/new-sweep-guidance-proof`.
The previous-test control is pinned to `088f2bbd`.

Recent commits and all four preceding archives were reviewed before continuing;
their hashes and historical baseline/candidate outcomes and coverage were verified.
All five acceptance checkboxes are supported by that evidence. Updating the live
PR body was rejected because the connector requires approval and this run's policy
is `never`; the archived reconciled body records the remaining tracking action.
Historical broader source coverage remains **83.5440%, below 90%**. These local
checks establish neither hosted-environment parity nor authorization to merge.

## Diagnostics on both output streams

The existing six cases now check diagnostic and remediation lines across both
stdout and stderr. Previously, a correct stdout message could conceal contradictory
stderr guidance or a second failure diagnostic. Production remains unchanged.

`stderr-validation.tar.gz` and `stderr-manifest.json` bind the test snapshot,
focused JUnit, targeted coverage, mutation receipts and formatting output.
All **15** focused cases pass. The six cases cover **361/1107 CLI statements
(32.61%, displayed as 33%)**, including all ten handler lines. Full Black passes
for **493** workspace files using the documented sequential adapter; Ruff and
`git diff --check` pass.

Two private production mutations add either a conflicting diagnostic or incorrect
guidance to stderr in each sweep failure handler. Each mutant passes all six
previous cases, fails all six strengthened cases at the relevant assertion, and
passes all six cases after byte-identical restoration. The original handler replay
also yields six exact-node failures and six restored passes. Existing output is
refused before mutation, preserving receipts and caller bytes. Reproduce the new
controls from the repository root:

```sh
tar -xzf docs/evidence/issue-2336-sweep-export-failures/stderr-validation.tar.gz -C /tmp verify_sweep_stderr.py
python /tmp/verify_sweep_stderr.py --output /tmp/new-sweep-stderr-proof
```

Commits `943d00cc` through `29ab2016` and all five preceding archive manifests
were reconciled. All six acceptance checkboxes are supported by the verified
evidence; the reconciled PR body is archived. The connector rejected the live
update because it requires approval under this run's `never` approval policy.
PR #2348 was verified open and ready for review. The full-suite comparisons remain
historical evidence; broader measured coverage remains **83.5440%, below 90%**.
These local checks establish neither hosted parity nor authorization to merge.
