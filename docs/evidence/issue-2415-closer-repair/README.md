# Issue #2415 closer repair evidence

Check date: 2026-11-03 (America/Chicago).
Base: `cab460a72a18d328c717a41090491b767d5a92c9`, current main containing merged #2416.

Outcome 1: the six tests from #2416 pass for their real scenario mappings,
agent inputs, serialization values, distributions, and matrices; each of the
three missing deliberate mutations causes its exact test to fail, and each
same node passes after byte-identical production restoration.

Command that prints the restored outcome:
```sh
/Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python -m pytest -q tests/test_compare_scenarios_coverage_2415.py tests/test_coverage_round_2415.py
```

Stop rule: stop on three failed repair attempts or 45 minutes; report the
first broken step and ask whether to stop. No unrelated repair is authorized.
Each mutation/test operation has a 110-second timeout. Only one mutation is
applied at a time, with unconditional byte-for-byte restoration before the
same node is rerun. No production mutation may enter the final diff.

The existing repository virtualenv is explicitly selected by the Makefile.
Its absolute interpreter path is used here because this linked worktree has
no `.venv`. It reports CPython 3.12.2, pytest 9.1.0, NumPy 2.4.4, pandas 3.0.2.
This is below the declared Python >=3.13 floor and the hosted Gate's 3.14;
local passes alone cannot establish the applicable hosted CI outcome.

Existing PR #2416 proof records and all partial-coverage disclosures are retained.
Target before/after uncovered-unit figures remain UNKNOWN; this repair does not
expand the original scope or claim repository-wide coverage completion.

## Results

| Exact test name | Mutation | Named RED assertion | Restored result |
| --- | --- | --- | --- |
| `test_compare_scenarios_builds_all_optional_figures_from_mapping_inputs` | Suppress `output["regime_timeline"]` assignment | Set equality fails: missing `regime_timeline` (line 47) | 1 passed in 0.85s |
| `test_compare_scenarios_supports_summary_frames_and_reports_invalid_inputs` | Remove `output["risk_metrics"]` assignment | Set equality fails: missing `risk_metrics` (line 61) | 1 passed in 0.95s |
| `test_excel_builds_agent_semantics_from_agents_when_frame_missing` | Return `None` from agents-derived fallback | `assert df is not None` fails (line 122) | 1 passed in 0.85s |

All RED runs exit 1 with the named assertion; all restored runs exit 0.
No test strengthening was required. Production and test files are unchanged.
Full exact commands, mutations, assertion output, restored output and hashes:
[mutation-transcripts.md](mutation-transcripts.md), [mutations.json](mutations.json).
Baseline: [baseline-pytest.log](baseline-pytest.log), 6 passed in 2.08s.
Final required command: [restored-pytest.log](restored-pytest.log), **6 passed in 0.96s**.

The original PR body already records valid deliberate-break/restored-pass proofs
for `test_distribution_helpers_handle_empty_constant_and_large_samples`
(`_downsample`: 6000 != 100),
`test_excel_normalizes_and_serializes_nested_agent_semantics`
(normalization plus empty-frame early-return mutations), and
`test_paths_matrix_validation_accepts_valid_and_rejects_invalid_inputs`
(disabled non-finite guard: NaN does not raise). These historical proofs are
retained, not rerun or represented as newly observed output.

## Applicable hosted CI

Pending on the repair PR. This evidence-only diff would normally skip pytest;
`verify:full` is requested to run the existing full Gate instead. A skipped
pytest job is not treated as a pass. Hosted CI results will be recorded in the
repair PR and appended to #2416's body after inspection.
