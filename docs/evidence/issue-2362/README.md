# Comparison question propagation (#2362)

The Questions widget now reaches the provider prompt through the real comparison helper, alongside both metric catalogs. Blank strings normalize to the existing default question text, shared with the dashboard, heuristic fallback and exported payload. Model selection, metrics and trace handling remain unchanged.

`python -m pytest tests/test_llm_prompts.py tests/test_llm_compare_runs.py tests/test_dashboard_comparison_llm.py -q -m 'not slow and not live_llm'`: **47 passed** offline. The end-to-end panel test uses real manifest/summary loading, `compare_runs`, and `_invoke_comparison_llm`, capturing both prompts inside the fake provider's `invoke` method. It verifies successful provider calls, identical data sections, question sections containing exactly the corresponding widget text, both metric catalogs retaining their values, and JSON exports matching each question and payload. A socket guard verifies that the panel regression makes no network connections. The LangChain prompt adapter and provider transport are fake; no live LLM quality is claimed.

`dashboard-question-red-green.txt` records actual failure after disconnecting the dashboard forwarding call, then success after exact-byte restoration. Hosted checks, review and original-issue verification remain separate stages.

## Acceptance checklist

- [x] 47 offline tests pass across prompt builders, real run comparison and the dashboard panel.
- [x] The panel-to-provider regression captures both questions and proves changing only the question changes the prompt.
- [x] Disconnecting dashboard forwarding produces an actual test failure; exact-byte restoration passes. Evidence: `dashboard-question-red-green.txt`.
- [x] Ruff, isort, Black, core mypy and diff checks pass.

## Validation follow-up

The comparison suite remains at **47 passing tests**. The affected test modules, plus config, config-chat, covariance and wizard-render regressions, give **311 passed, 1 skipped, 1 deselected** with `-m 'not slow and not live_llm'`.

`ruff check pa_core tests dashboard` and `isort --check-only pa_core tests dashboard` pass. Core `mypy pa_core` passes on all **165 source files**, and `git diff --check` passes. The 18 pre-existing isort failures are resolved with import ordering and wrapping corrections only.

Repository-wide `black --check --line-length 100 --exclude '(\.workflows-lib|node_modules)' .` passes: **516 files would be left unchanged**. The required command was run with `BLACK_NUM_WORKERS=1` and a temporary `sitecustomize` outside the repository that adds periodic asyncio wakeups: this sandbox blocks the default multiprocessing socket and can leave asynchronous worker completion notifications waiting indefinitely. The formatter rules and checked file set are unchanged.

The remote PR checklist update was attempted after reconciling the prior commits, but the connector rejected the write because it requires approval and this run's approval policy is `never`. The checklist above records the verified task state. PR #2393 was open and ready for review when checked; hosted CI and review remain pending.

Committing was also attempted after validation, but staging failed because `.git/index.lock` cannot be created on the read-only `.git` filesystem. The implementation and evidence remain in the working tree for the automation host to commit.
