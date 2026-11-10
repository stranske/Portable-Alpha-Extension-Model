# Issue #2476 evidence — restored coverage round after closed PR #2484

Restores the reviewed #2476 test/evidence commit after closed PR #2484.
The restored focused module is `tests/test_coverage_round_2381.py`
(49 test functions; the 4-case parametrized
`test_results_main_reports_packet_export_failures` brings it to the
recorded **52 cases**). This round adds
`tests/test_coverage_round_2476_facade.py` for the remaining facade
branches below, without changing the restored file's 52-case count.

## Acceptance records (as recorded for this PR)

```text
/opt/anaconda3/bin/python3 -m pytest -q tests/test_coverage_round_2381.py
52 passed in 3.04s

/opt/anaconda3/bin/python3 -m black --check --line-length 100 \
  --exclude '(\.venv|\.workflows-lib|node_modules)' .
524 files unchanged
```

Gate coverage command completed in 74.97s and emitted coverage
artifacts; it had 2,021 passing tests and 3 expected local-environment
failures because this host uses Python 3.12 while packaging declares
`>=3.13` (`requires-python` in `pyproject.toml`). Hosted Python 3.14 CI
is the authoritative required-CI gate. See `gate-measurements.json`.

## Seven-target round baseline and reduction

The seven ranked targets, in dispatch order (from
`docs/evidence/issue-2381-closer-repair/ranking-provenance.json`), are:

1. `pa_core/llm/result_explain.py`
2. `pa_core/pa.py`
3. `pa_core/facade.py`
4. `pa_core/units.py`
5. `pa_core/sim/internal_pa_financing.py`
6. `pa_core/manifest.py`
7. `dashboard/pages/4_Results.py`

Combined uncovered statements at the round baseline: **209**
(`round_baseline_selected_missing_statements` in
`docs/evidence/issue-2381-closer-repair/gate-measurements.json`).

This round records **207 fewer uncovered statements** (209 → 2
remaining) and **119 fewer uncovered branches** from that seven-target
round baseline, measured with branch coverage as a diagnostic on the
same target set. The authoritative Gate stays line-only
(`branch = false` under `[tool.coverage.run]` in `pyproject.toml`);
the branch figures are the diagnostic comparable to the round's
statement+branch wording, not the Gate percentage.

## Remaining facade and unreachable branch exceptions (documented)

The 2 remaining uncovered statements and the residual partial branches
are confined to the categories below. These are documented exceptions,
not silently skipped targets:

### Facade (`pa_core/facade.py`)

- **numpy-absent fallbacks**: `_to_builtin_scalar` returns the value
  unchanged when `import numpy` raises `ImportError`, and
  `_serialize_agent_semantics_input` proceeds with `np_module = None`
  on the same failure. numpy is a hard dependency
  (`dependencies` in `pyproject.toml`), so these arcs cannot occur in
  the Gate environment; they are exercised only by import mocking
  (`test_facade_scalar_helpers_tolerate_missing_numpy` in the restored
  file).
- **Defensive mapping-serialization failure**: the
  `except (ValueError, TypeError)` warning path in
  `_serialize_agent_semantics_input` fires only when
  `pd.DataFrame(mapping)` rejects a non-row mapping (for example
  mismatched column lengths). Covered from this round by
  `test_serialize_agent_semantics_keeps_mapping_when_frame_build_fails`
  in `tests/test_coverage_round_2476_facade.py`, together with the
  Series / DataFrame-list / mixed Series-dict / mixed frame-Series /
  scalar tuple / scalar list / ndarray / column-mapping container
  branches and the defensive `export()` unsupported-artifacts-type
  raise, which typed callers cannot reach.
- **Type-only block**: the `TYPE_CHECKING` import block is marked
  `# pragma: no cover - type hints only` and is unreachable at runtime
  by construction.

### Unreachable / runtime-only branches on the other targets

- `dashboard/pages/4_Results.py`: the `if __name__ == "__main__":`
  script entrypoint arc is reachable only when the page is executed
  as a script; it is covered through `runpy.run_path(..., run_name=
  "__main__")` in `test_results_script_entrypoint_invokes_main`, and
  the auto-refresh `st.rerun()` arc requires a live Streamlit runtime
  (covered with a fake `st` in
  `test_results_main_auto_refresh_sleeps_then_reruns`).
- Empty-container short-circuits: the `all(...)` row-shape checks in
  `_serialize_agent_semantics_input` are guarded by truthiness, so the
  empty-list and empty-mapping inputs fall through to the scalar /
  empty-records exits by design (asserted in the new facade tests);
  coverage.py reports these guard arcs as partial branches even though
  both exits are asserted.

No other seven-target statements remain uncovered beyond the 2
recorded above. Exact per-line coverage JSON is emitted by the Gate
coverage command (see `gate-measurements.json`); this record documents
the exception categories source-inspected against the current tree.
