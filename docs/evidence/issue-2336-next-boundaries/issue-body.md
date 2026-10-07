## Why

Continuing the fleet coverage initiative after the previous round closed. Tests must detect real regressions rather than just increase the metric.

## Scope

The current selected low-blast-radius chunk is CLI observability and comparison-error boundaries in `pa_core/cli.py`, ranked first in the last 500 source-touching commits (26 repair-subject proxy matches, 132 touches, 312 uncovered statements). Fresh complete baseline at d4ed680: 1594 passed, 1 skipped, 2 default-deselected, 12133/14553 covered statements (83.37112622826909%). Repair subjects are a proxy, not verified escaped incidents. Preserve the original initiative contract below.

## Non-Goals

No refactors, workflow edits, dependency-policy changes, broad product changes, or artificial fixes when measured coverage is already at least 90%.

## Tasks

- [ ] Run `python -m pytest --cov=pa_core --cov=dashboard --cov-report=json:coverage.json tests/` and rank current measured sources using escaped-defect history, churn and uncovered statements. If coverage is at least 90%, report the evidence and close this issue instead of opening a PR.
- [ ] Add focused tests under `tests/` for the highest-ranked safely bounded behavior; minimally repair any reproducible bug found in that selected module, including negative-cache sentinel confusion if present.
- [ ] Deliberately break each new test's production behavior, run its exact pytest node and capture FAIL; restore the production implementation and capture PASS.
- [ ] Record the selected source, rank evidence, exact pytest nodes, measured coverage delta, break/revert evidence and limitations in one ready-for-review PR for this logical chunk.

## Acceptance Criteria

- [ ] `python -m pytest --cov=pa_core --cov=dashboard --cov-report=json:coverage.json tests/` captures current coverage and source gaps in the PR evidence. Coverage at or above 90% causes an evidence comment and issue closure rather than invented work.
- [ ] The focused command `python -m pytest tests/test_cli_observability_boundaries.py -q` exits 0 on restored production code; each newly added named node exits nonzero against a real deliberate production mutation. Preserve RED and restored GREEN output, named-node JUnit, actual commands and byte-identical restoration hashes under `docs/evidence/issue-2336-next-boundaries/`.
- [ ] The selected module's measured covered lines increase without weakening existing tests, coverage scope, skip ceilings or baseline floors. Any real bug receives a minimal verified production diff in the same PR; unrelated local changes remain untouched.

## Implementation Notes

Original contract: choose escaped-defect priority before churn and uncovered mass, never file size; every new test must actually fail when its production behavior is broken and pass after restoration; repair discovered bugs rather than working around them. Watch for `cache.get(key) is None` confusing never queried with cached absence. Low blast radius only: tests or tightly scoped fixes, no refactors; one PR per logical chunk. The goal is to improve code, not a metric.
