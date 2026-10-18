# Comparison question propagation (#2362)

The Questions widget now reaches the provider prompt through the real comparison helper, alongside both metric catalogs. Blank strings normalize to the existing default question text, shared with the dashboard, heuristic fallback and exported payload. Model selection, metrics and trace handling remain unchanged.

`python -m pytest tests/test_llm_prompts.py tests/test_llm_compare_runs.py tests/test_dashboard_comparison_llm.py -q -m 'not slow and not live_llm'`: **47 passed** offline. The end-to-end panel test uses real manifest/summary loading and captures both provider prompts. It verifies that their data sections are identical, each question section contains exactly the corresponding widget text, both metric catalogs retain their values, and JSON exports match each question and payload. A socket guard verifies that the panel regression makes no network connections. Provider transport is fake and no live LLM quality is claimed.

`dashboard-question-red-green.txt` records actual failure after disconnecting the dashboard forwarding call, then success after exact-byte restoration. Hosted checks, review and original-issue verification remain separate stages.

## Acceptance checklist

- [x] 47 offline tests pass across prompt builders, real run comparison and the dashboard panel.
- [x] The panel-to-provider regression captures both questions and proves changing only the question changes the prompt.
- [x] Disconnecting dashboard forwarding produces an actual test failure; exact-byte restoration passes. Evidence: `dashboard-question-red-green.txt`.
- [ ] Ruff, isort, Black, core mypy and diff checks pass.

## Validation follow-up

The comparison suite remains at **47 passing tests**. Including the existing wizard-render tests gives **55 passing tests** after the two formatting-only adjustments needed for the required repository-wide Black gate.

Ruff passes on `pa_core` and the comparison and wizard-render files. Isort passes on both comparison core modules, the dashboard comparison component, and the four checked test modules. Core mypy passes on all **165 source files**, and `git diff --check` passes.

Repository-wide Black passes: **516 files would be left unchanged**. The required command was run with `BLACK_NUM_WORKERS=1` and a temporary `sitecustomize` outside the repository that adds periodic asyncio wakeups: this sandbox blocks the default multiprocessing socket and can leave asynchronous worker completion notifications waiting indefinitely. The formatter rules and checked file set are unchanged.

The final acceptance checkbox remains open because the broader `isort --check-only pa_core tests dashboard` command reports **18 pre-existing failures** in files outside this comparison change: `pa_core/cli.py`, `pa_core/config.py`, `pa_core/llm/config_chat.py`, `pa_core/sim/covariance.py`, `dashboard/pages/3_Scenario_Wizard.py`, and 13 unrelated test modules. Those files have no changes in this run.
