# Opener Workloop State

## 2026-05-13T20:59:15Z

- Repo: `stranske/Portable-Alpha-Extension-Model`
- Issue: `#1786` `Honor selected provider in Results LLM Comparison`
- Branch: `codex/issue-1786-provider-comparison`
- Lane: opener materialization by `codex`
- State: implementation complete locally; ready to push and open PR
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

