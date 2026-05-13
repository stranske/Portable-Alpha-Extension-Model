# Opener Workloop State

## 2026-05-13T20:59:15Z

- Repo: `stranske/Portable-Alpha-Extension-Model`
- Issue: `#1786` `Honor selected provider in Results LLM Comparison`
- PR: `#1787` https://github.com/stranske/Portable-Alpha-Extension-Model/pull/1787
- Branch: `codex/issue-1786-provider-comparison`
- Lane: opener materialization by `codex`
- State: ready PR opened; waiting for keepalive / CI
- Notes:
  - Preserved pre-existing local `.gitignore` modification; it is not part of this issue.
  - Skipped `Counter_Risk#594` because `origin/main` already lists all 7 runner macros/buttons and missing-file behavior in `docs/runner_xlsm_macro_manual_verification.md`.
  - Skipped `Pension-Data#426` as an audit/report follow-up issue rather than a bounded implementation PR.
- Changes:
  - `pa_core/llm/compare_runs.py` now threads `LLMProviderConfig` or structured credentials to `create_llm` without coercing non-OpenAI providers to OpenAI.
  - `dashboard/components/comparison_llm.py` passes the resolved provider config into `compare_runs` and exposes Azure endpoint/API-version fields when Azure OpenAI is selected.
  - Added Anthropic/Azure provider-threading tests and dashboard Azure config coverage.
- Validation:
  - `pytest tests/test_llm_compare_runs.py -k "anthropic or azure" --no-cov` passed.
  - `pytest tests/test_dashboard_comparison_llm.py -k azure --no-cov` passed.
  - `pytest tests/test_dashboard_llm_settings.py tests/test_result_explain.py tests/test_llm_compare_runs.py tests/test_llm_prompts.py tests/test_dashboard_explain_results.py tests/test_dashboard_comparison_llm.py tests/test_reference_packs.py --no-cov` passed: 153 passed, 2 warnings.
  - `ruff check pa_core/llm/compare_runs.py dashboard/components/comparison_llm.py tests/test_llm_compare_runs.py tests/test_dashboard_comparison_llm.py` passed.
  - `black --check pa_core/llm/compare_runs.py dashboard/components/comparison_llm.py tests/test_llm_compare_runs.py tests/test_dashboard_comparison_llm.py` passed.
  - `rg "provider != .openai" pa_core/llm/compare_runs.py` returned no matches.
- PR routing:
  - Opened ready PR with labels `agent:codex`, `agents:keepalive`, and `autofix`.
  - Emitted `pr_opened` relay with `active.source_repo=stranske/Portable-Alpha-Extension-Model`, `active.source_issue=1786`, `active.source_pr=1787`, `active.next_action=wait_for_keepalive`.
  - Initial Gate Followups skipped; opener infra repair added `agent:retry` and dispatched `agents-81-gate-followups.yml`.
  - Post-repair cap-health at `2026-05-13T21:02:25Z`: raw cap 4/5, `raw_cap_reached=false`, `normal_cap_reached=false`; PR `#1787` state `draining` with active Gate evidence.
