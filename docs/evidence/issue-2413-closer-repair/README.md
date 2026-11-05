# Issue #2413 closer repair evidence

Check date: 2026-11-03 (America/Chicago).
Base: `cec2fe3c64fd909e795d0f881a73540cf4c35269` (current `main`, includes merged #2414, #2464).

Outcome: all **23** collected nodes in `tests/test_coverage_round_2413.py` fail on an
isolated production mutation and pass after byte-identical restoration; the missing-index
run gate observes `pa_cli.main` via the module-global spy patched in the test (merged #2464).

## Commands (repository-selected interpreter)

Interpreter (Makefile-selected repo venv; no `.venv` in this linked worktree):

`/Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python`

Collect-only (required):

```sh
$PY -m pytest tests/test_coverage_round_2413.py --collect-only -q
```

Restored module run (required):

```sh
$PY -m pytest tests/test_coverage_round_2413.py -q
```

Outputs: [collect-only.txt](collect-only.txt) (23 collected), [restored-pytest.txt](restored-pytest.txt) (23 passed).

## Executor entry point (#2464)

On `main`, `test_main_step_5_run_gate_opens_index_upload_without_executing` patches
`module["main"].__globals__["pa_cli"]` before calling `main()`. Production
`main()` invokes `pa_cli.main(args)` at `dashboard/pages/3_Scenario_Wizard.py` line 2306
(`from pa_core import cli as pa_cli`). The test asserts `executor_calls == []` when no index
CSV is uploaded.

Deliberate mutation (production): insert `pa_cli.main([])` at the start of the
`if run_simulation:` block. RED: `AssertionError` — `executor_calls == [[]]` vs `[]`.
Restored: pass. Transcript: [mutation-transcripts.md](mutation-transcripts.md) §22.

## Break-and-revert (all collected cases)

Full machine-readable record: [mutations.json](mutations.json) (23 entries, each `fail_exit=1`,
`pass_exit=0`). Human-readable tails: [mutation-transcripts.md](mutation-transcripts.md).

No production or test source files are changed in this repair PR; mutations were transient only.

## Prior branch merge note

Local branch `closer/issue-2413` carried commit `42698028` (executor spy). **Discarded** as
duplicate of merged #2464 (`6abacb50` on `main`). **Kept** all work already on `main` from
#2414 and #2464; this PR adds evidence publication only.

## Gate coverage (authoritative configuration)

Consumer Gate: `.github/workflows/pr-00-gate.yml` → Workflows
`reusable-10-ci-python.yml@main` with `coverage: true`, `enable-soft-gate: true`,
`python-versions: '["3.14"]'` (hosted; historical #2414 Gate used 3.12).

Gate pytest invocation (from `reusable-10-ci-python.yml`; `pyproject.toml` `addopts` applies
`-m 'not live_llm'`):

```sh
pytest -p ci_test_growth \
  --junitxml=pytest-junit.xml \
  --cov --cov-config=pyproject.toml \
  --cov-report=xml:coverage.xml \
  --cov-report=term-missing \
  --cov-report=json:coverage.json \
  -n auto --dist=loadgroup
```

`[tool.coverage.run]` sets `branch = false`, so Gate **statement+branch** counts equal
**statement** misses (`num_branches = 0` in Gate JSON). Supplementary branch diagnostics are
not Gate-authoritative.

### Target ranking (issue #2413)

Issue body order: (1) `dashboard/pages/3_Scenario_Wizard.py` — 164 uncovered stmt+branch;
(2) `pa_core/cli.py` — 165. Ranking rule: escaped defects, then churn, then uncovered mass.

### Measured partial result (#2414 + #2464 on `main`)

| Target | Issue-ranked uncovered (stmt+branch) | Gate stmt misses after #2414* | Δ stmt misses | Disposition |
| --- | ---: | ---: | ---: | --- |
| `dashboard/pages/3_Scenario_Wizard.py` | 164 | **120** | **−59** | Top-level `main()` routing, navigation, reset, and run gates covered; deep simulation execution (`2284–2357`) and step bodies remain. |
| `pa_core/cli.py` | 165 | 165 (unchanged) | 0 | **Deferred** — not started; ranked second but #2414 exhausted the lane on Wizard routing only. |

\*After #2414 merge, PR #2414 body reports Scenario Wizard **120** statement misses on a
wizard-focused `--cov` run (−59 from its **179**-miss baseline on the same basis). Hosted Gate
pytest success for the implementation PR:
[Gate run 39454575215](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/39454575215)
(Python CI / python 3.12, PR #2414 head).

**Chunk goal:** ~180 covered stmt+branch units. **Delivered on Wizard:** 59. **Shortfall:** ~121.
This repair does not expand scope to chase 180; it completes closer acceptance (executor proof,
full mutation matrix, evidence).

### Unsuitable / deferred without padding

| Item | Why deferred |
| --- | --- |
| `pa_core/cli.py` (165 units) | Ranked #2; untouched in #2414/#2464; CLI integration needs separate focused tests, not Wizard routing work. |
| Wizard `main()` simulation success path (`2284–2357`) | Requires uploaded index, temp YAML, `dashboard_cli.build_pa_core_args`, and real `pa_cli.main` execution or extensive fakes; outside the #2414 top-level routing contract. |
| Step renderer bodies (`_render_step_*`) | Exercised indirectly via fakes in this module; full UI coverage belongs to other wizard test modules, not this routing chunk. |

Local full Gate replay (`pytest` over all of `tests/`) was **not** run here per closer dispatch
policy (CI owns the full suite on the PR). This repair PR expects Gate **Python CI** to run the
module above including `tests/test_coverage_round_2413.py`.
