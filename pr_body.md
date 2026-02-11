<!-- pr-preamble:start -->
> **Source:** Issue #1194

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
Follow-up on PR #1221 for issue #1194, closing remaining gaps around Scenario Wizard helper validation and tests.

#### Tasks
- [x] [#1221](https://github.com/stranske/Portable-Alpha-Extension-Model/issues/1221)
- [x] [#1194](https://github.com/stranske/Portable-Alpha-Extension-Model/issues/1194)

#### Acceptance criteria
- [x] Original PR: #1221
- [x] Parent issue: #1194

## Related Issues
- [x] [#1221](https://github.com/stranske/Portable-Alpha-Extension-Model/issues/1221)
- [x] [#1194](https://github.com/stranske/Portable-Alpha-Extension-Model/issues/1194)
## References
- [ ] _Not provided._

## Notes
- [x] Verified wizard helper tests via `pytest tests/test_wizard_helpers.py -m "not slow"`.
- [x] Verified wizard regime wiring via `pytest tests/test_wizard_regime_wiring.py -m "not slow"`.
- [x] Verified regime model wiring via `pytest tests/test_wizard_regime_wiring.py -m "not slow"`.
- [x] Re-verified regime input validation coverage via `pytest tests/test_wizard_helpers.py -m "not slow"`.
- [x] Re-validated regime mapping key checks via `pytest tests/test_wizard_helpers.py -m "not slow"`.
- [x] Added regime list validation coverage via `pytest tests/test_wizard_helpers.py -m "not slow"`.

<!-- auto-status-summary:end -->

## Task Reconciliation
- [x] Reviewed recent commits for wizard helper tests and formatting updates.
- [x] Updated task checkboxes to reflect completed follow-up work for #1221.
- [x] Synced PR body with Issue #1194 scope and acceptance criteria.
- [x] Confirmed task checkboxes after adding regime mapping key coverage.

## Task Reconciliation (LLM Follow-up #1381)
- [x] Reviewed recent commits (`750b682`, `7d982d4`, `f320f85`, `50496ed`) for missing checkbox reconciliation context.
- [x] Aligned `pyproject.toml` `llm` extras with lockfile pins for deterministic resolution.
- [ ] Confirmed `pip install -e '.[llm]'` in a clean venv with full dependency download (requires networked environment).
- [x] Validated `pa_core/llm/result_explain.py` entrypoint contract via `pytest tests/test_llm_result_explain_entrypoint.py -m "not slow"`.

## Task Reconciliation (Keepalive Next Task #1393)
- [x] Reviewed recent commits (`9c71ab9`, `803a57a`, `6061a6e`) and reconciled unchecked task status.
- [x] Create file `dashboard/components/llm_settings.py` with basic module structure and imports.
- [x] Add docstring documentation to `dashboard/components/llm_settings.py` explaining the module's purpose and relationship to Trend's implementation.
- [x] Implement `sanitize_api_key()` in `dashboard/components/llm_settings.py`.
- [x] Implement `read_secret()` in `dashboard/components/llm_settings.py` to read from `st.secrets`.
- [x] Update `read_secret()` in `dashboard/components/llm_settings.py` to read from environment variables.
- [x] Implement `resolve_api_key_input()` in `dashboard/components/llm_settings.py` to accept raw key input.
- [x] Update `resolve_api_key_input()` in `dashboard/components/llm_settings.py` to accept env-var name input.
- [x] Update `resolve_api_key_input()` in `dashboard/components/llm_settings.py` to resolve env-var names to actual values.
- [x] Define constants for Portable environment variable names in `dashboard/components/llm_settings.py`.
- [x] Implement environment variable reading logic for `PA_LLM_PROVIDER` with appropriate defaults.
- [x] Implement environment variable reading logic for `PA_LLM_MODEL` with appropriate defaults.
- [x] Implement environment variable reading logic for `PA_LLM_BASE_URL` with appropriate defaults.
- [x] Implement environment variable reading logic for `PA_LLM_ORG` with appropriate defaults.
- [x] Implement environment variable reading logic for `PA_STREAMLIT_API_KEY` with appropriate defaults.
- [x] Implement `default_api_key(provider)` in `dashboard/components/llm_settings.py` with Portable defaults.
- [x] Implement fallback logic in `default_api_key()` to check `OPENAI_API_KEY` when provider is openai.
- [x] Implement fallback logic in `default_api_key()` to check `CLAUDE_API_STRANSKE` when provider is anthropic.
- [x] Implement fallback logic in `default_api_key()` to check `LANGSMITH_API_KEY` for LangSmith integration.
- [x] Implement `resolve_llm_provider_config(...)` in `dashboard/components/llm_settings.py` returning a provider config compatible with `pa_core/llm/create_llm()`.
- [x] Write unit tests for `sanitize_api_key()` verifying that API keys are properly masked in output.
- [x] Write unit tests for `resolve_api_key_input()` with direct API key input.
- [x] Write unit tests for `resolve_api_key_input()` with environment variable name input.
- [x] Write unit tests verifying that no secrets appear in test output or logs.
- [x] Write unit tests for `default_api_key()` covering all supported providers.

### Acceptance Criteria (Keepalive Next Task #1393)
- [x] The component can be imported and displayed in a Streamlit app without raising exceptions when environment variables `PA_LLM_PROVIDER`, `PA_LLM_MODEL`, and all API key variables are unset.
- [x] Entering an env var name like `OPENAI_API_KEY` in the UI resolves to the secret value without displaying the secret value in the UI.
- [x] When a key is missing, the error message displayed contains instructions for setting the required environment variable and does not include any API key values (partial or complete).
- [x] All unit tests pass with 100% coverage for `sanitize_api_key()`, `read_secret()`, `resolve_api_key_input()`, and `default_api_key()`.
- [x] Running the test suite produces no output containing API keys or secrets (verified by grep for common key patterns).

### Verification (Keepalive Next Task #1393)
- [x] `pytest tests/test_dashboard_llm_settings.py --cov=dashboard.components.llm_settings --cov-report=term-missing -m "not slow"` (37 passed, module coverage 100%).
- [x] `pytest tests/test_dashboard_explain_results.py -m "not slow"` (2 passed).
- [x] `pytest tests/test_dashboard_llm_settings.py tests/test_dashboard_explain_results.py -m "not slow"` and grep scan for common secret patterns (no matches).

## Task Reconciliation (Keepalive Next Task #1398)
- [x] Reviewed recent commits (`f75c59e`, `8949dbc`) touching `pa_core/llm/compare_runs.py` and `tests/test_llm_compare_runs.py`.
- [x] Confirmed those commits only cover task-01 scope (prior manifest loader) with tests.
- [x] Updated PR checkbox tracking for issue #1378 before continuing implementation.

### Task Progress (Keepalive Next Task #1398)
- [x] Create `dashboard/components/comparison_llm.py` with Streamlit component skeleton using Trend's `streamlit_app/components/comparison_llm.py` as reference.

### Verification (Keepalive Next Task #1398)
- [x] `pytest tests/test_dashboard_comparison_llm.py -m "not slow"` (1 passed).

## Task Reconciliation (Keepalive Next Task #1400)
- [x] Reviewed recent commits (`f9d6cb2`, `ca9d1b7`, `8b80f54`) and reconciled checklist progress for comparison LLM work.
- [x] Added comparison export assertions in `tests/test_dashboard_comparison_llm.py` to verify TXT/JSON payloads include config diff and trace URL.
- [x] Verified readable previous-run path flow produces comparison output and download artifacts.

### Acceptance Criteria (Keepalive Next Task #1400)
- [x] When `manifest_data["previous_run"]` exists and is readable, the comparison panel can produce a coherent explanation and expose trace URL details in UI/export output.

### Verification (Keepalive Next Task #1400)
- [x] `pytest tests/test_llm_compare_runs.py tests/test_dashboard_comparison_llm.py tests/test_dashboard_results_previous_run.py -m "not slow"` (14 passed).
- [x] `pytest tests/test_dashboard_comparison_llm.py -m "not slow"` (2 passed).

## Task Reconciliation (Keepalive Next Task #1401)
- [x] Reviewed recent commits (`ca9d1b7`, `8b80f54`, `ad4c380`) and reconciled unchecked comparison-LLM checklist state.
- [x] Added Results-page integration helper `_render_comparison_panel(...)` in `dashboard/pages/4_Results.py` to centralize previous-run availability gating and panel invocation.
- [x] Added unit test coverage in `tests/test_dashboard_results_previous_run.py` verifying readable `manifest_data["previous_run"]` calls comparison panel with resolved `run_key`.

### Acceptance Criteria (Keepalive Next Task #1401)
- [x] When `manifest_data["previous_run"]` exists and is readable, the comparison panel can produce a coherent explanation and expose trace URL details in UI/export output.

### Verification (Keepalive Next Task #1401)
- [x] `pytest tests/test_dashboard_results_previous_run.py tests/test_dashboard_comparison_llm.py tests/test_llm_compare_runs.py -m "not slow"` (15 passed).
