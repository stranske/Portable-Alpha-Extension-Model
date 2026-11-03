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

## Target ranking provenance

Issue [#2415](https://github.com/stranske/Portable-Alpha-Extension-Model/issues/2415)
(`coverage-autopilot:chunk-round`, opened 2026-10-20) states the ranked target
order is the Orchestrator's own ranking: escaped defects, then churn, then
uncovered mass (unchanged from prior fleet rounds). The five listed modules and
their CI uncovered statement+branch counts (48 / 43 / 42 / 40 / 33) are copied
from that issue body. PR #2416 verified the first three targets only; ranks
four and five were deferred in the original PR partial-coverage disclosure. No
earlier machine-readable ranking artifact (coverage JSON snapshot or bot comment
with timestamps) was recovered in this repair; the issue body is the authoritative
published provenance available here.

## Results

| Exact test name | Mutation | Named RED assertion | Restored result |
| --- | --- | --- | --- |
| `test_compare_scenarios_builds_all_optional_figures_from_mapping_inputs` | Suppress `output["regime_timeline"]` assignment | Set equality fails: missing `regime_timeline` (line 47) | 1 passed in 0.85s |
| `test_compare_scenarios_supports_summary_frames_and_reports_invalid_inputs` | Remove `output["risk_metrics"]` assignment | Set equality fails: missing `risk_metrics` (line 61) | 1 passed in 0.95s |
| `test_excel_builds_agent_semantics_from_agents_when_frame_missing` | Return `None` from agents-derived fallback | `assert df is not None` fails (line 122) | 1 passed in 0.85s |

All RED runs exit 1 with the named assertion; all restored runs exit 0.
No test strengthening was required. Production and test files are unchanged.
Full exact commands, mutations, assertion output, restored output and hashes
(output trailing whitespace removed):
[mutation-transcripts.md](mutation-transcripts.md), [mutations.json](mutations.json).
Baseline: [baseline-pytest.txt](baseline-pytest.txt), 6 passed in 2.08s.
Final required command: [restored-pytest.txt](restored-pytest.txt), **6 passed in 0.96s**.

The original PR body already records valid deliberate-break/restored-pass proofs
for `test_distribution_helpers_handle_empty_constant_and_large_samples`
(`_downsample`: 6000 != 100),
`test_excel_normalizes_and_serializes_nested_agent_semantics`
(normalization plus empty-frame early-return mutations), and
`test_paths_matrix_validation_accepts_valid_and_rejects_invalid_inputs`
(disabled non-finite guard: NaN does not raise). These historical proofs are
retained, not rerun or represented as newly observed output.

## Publication and hosted pytest

Repair PR [#2462](https://github.com/stranske/Portable-Alpha-Extension-Model/pull/2462)
merged 2026-11-03 (evidence-only diff on main). Follow-up repair PR
`closer-r2/issue-2415` appends the three missing mutation transcripts to
[#2416](https://github.com/stranske/Portable-Alpha-Extension-Model/pull/2416)
(preserving its original body and partial-coverage disclosures) and corrects
#2462's criteria checkboxes to match the published bodies.

**Original hosted pytest success (PR #2416, counts as pytest PASS):**
[Gate run 39454791657](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/39454791657)
on head `a8adf27df9689ea9b592404757ea46327417480b`, job **Python CI / python
3.12** — `platform linux -- Python 3.12.15, pytest-9.1.1` —
`1907 passed, 1 skipped` in 164.87s. This is the last successful full Gate
pytest leg before merge; it is not inferred from skipped docs-only jobs.

Repair PR #2462 Gate run
[41626835314](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/41626835314)
was docs-only classified; pytest there was skipped and is **not** counted as
pytest PASS. Local focused runs (repository `.venv`, Python 3.12.2) confirm the
six #2416 tests only; they do not replace hosted CI.
