# Optional single-run PDF failures

Related to #2336. Six cases invoke the production CLI with `--pdf --sensitivity`
and stub only optional plotting/export and sensitivity presentation. They check
renderer and module import failures, filesystem/permission errors, and invalid
data/type errors. Each verifies exactly one exporter call, the exact console and
ERROR log diagnostic, matching remediation, an unchanged completed workbook,
no nonexistent PDF, and a finalized manifest containing the failure warning.
No real PDF renderer or physical export is claimed. Production code is unchanged.

## Selection and measured results

The last 500 source-touching commits rank `pa_core/cli.py` first: 21 repair-subject
proxy matches, 133 touches and 287 uncovered statements. These subject matches
are proxies, not confirmed escaped incidents. Raw history and ranking are archived.

The matched complete commands used the same Python 3.12.2 environment and source
revision `395fcd394e185cd5042ca6f90569928f2f037844`:

```sh
python -m pytest --cov=pa_core --cov=dashboard --cov-report=json:coverage.json tests/
```

Baseline: 1648 passed, one failed, one skipped, two default-deselected. Candidate:
1654 passed, the same one failed, one skipped and two deselected. All old JUnit
outcomes are unchanged; exactly six nodes were added. Both process exits are 1:
the pre-existing `test_suggest_sleeve_sizes_optimize_prefers_best` fails importing
SciPy because installed NumPy 1.26.4 lacks `numpy.long`. The environment also has
pandas 3.0.3 and an isolated xarray 2026.9 overlay. Raw warnings and versions are
retained. This is not a green full suite or hosted-environment parity claim.

Covered statements increase 12147 → 12166 / 14554, or 83.46159131510238% →
83.59213961797444%. CLI coverage increases 820 → 838 / 1107. All 186 source files
and 181 excluded lines are unchanged; no covered line regresses. The one extra
covered statement outside CLI belongs to the warning capture exercised by these
failures. No source, dependency, skip, floor or coverage configuration changed.

## Focused validation and mutation proof

```sh
python -m pytest tests/test_cli_pdf_failures.py tests/test_cli_packet_diff.py -q
python docs/evidence/issue-2336-pdf-failures/replay.py --output /tmp/fresh-pdf-proof
```

The focused command passes 17 cases. Replay extracts the pinned committed source
into a private temporary directory and overlays the current six-case test. For
each exact node it breaks the applicable production PDF diagnostic. Each RED
process exits 1 with one named assertion failure; byte-identical restoration
exits 0 with one passing node. Six executions cover four distinct diagnostic
mutation shapes. Caller source and tests remain unchanged; private trees are
cleaned. Reusing the output directory fails before mutation and preserves all
31 proof files. Replay requires Python 3.12+ and the repository test dependencies.

`validation.tar.gz` retains 52 raw members: commands, full/focused logs, JUnit,
coverage, ranking/history, original and mutant source, test/replay snapshots,
restoration hashes and output-reuse proof. `manifest.json` binds archive members
and current source/test/replay inputs. Full Black checks 484 Python files; focused
Ruff and `git diff --check` pass. Notebook formatting dependencies are unavailable
and disclosed in the Black log.

Source #2336 remains open below 90%. Earlier provider NON_PASS dispositions stay
unchanged. Keepalive owns hosted checks and review; closer requires complete
exact-head checks, zero active review threads, the seven-minute floor and actual
post-merge comparison before accepting this bounded component.
