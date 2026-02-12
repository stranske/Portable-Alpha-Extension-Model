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

## Task Reconciliation (Keepalive Next Task #1391)
- [x] Reviewed recent commits (`a30c46c`, `f24ebd8`, `0b2c8d5`) and reconciled checkbox state with implemented `result_explain` changes.
- [x] Reviewed follow-up commits (`6e5ecb9`, `6fcb282`, `a678b9d`) and confirmed checkbox state remains aligned with implemented metric-catalog and redaction behavior.
- [x] Added explicit acceptance-mapped unit coverage for `analysis_output` required sections and JSON serialization.
- [x] Added dedicated unit test confirming `analysis_output.manifest_highlights` includes sentinel manifest values.
- [x] Expanded `metric_catalog` alias handling/tests to cover human-readable metric column names (e.g., `Tracking Error`, `CVaR`, `Breach Probability`).

### Acceptance Criteria (Keepalive Next Task #1391)
- [x] `analysis_output` returned by `explain_results_details()` is non-empty and includes JSON-serializable sections for: column list, basic statistics, tail sample rows, and key quantiles.
- [x] `analysis_output` includes manifest highlights containing at least one sentinel value derived from the provided manifest (e.g., `run_name='SENTINEL_RUN'`).
- [x] StressDelta behavior is deterministic: with StressDelta-relevant inputs present, `analysis_output` includes a StressDelta summary section; without those inputs, the StressDelta section is absent or explicitly `None`.
- [x] `metric_catalog` exists in the returned `payload_dict` as a structured dict/list and includes TE, CVaR, and breach probability entries for metrics present in the inputs, each with at least `{label, value}`.
- [x] `metric_catalog` generation is resilient to missing metric columns: `explain_results_details()` succeeds and omits (or marks unavailable) only missing metric entries without raising.
- [x] API keys are not leaked: if `create_llm(...)` or the LLM invocation raises an exception containing the `api_key`, the surfaced/returned error text does not contain the secret and includes a redaction token.

### Verification (Keepalive Next Task #1391)
- [x] `pytest -q tests/test_result_explain.py -m "not slow"` (20 passed).
- [x] `coverage run -m pytest -q tests/test_result_explain.py -m "not slow"` and `coverage report -m pa_core/llm/result_explain.py` (20 passed; module coverage 82%).

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

## Task Reconciliation (Keepalive Next Task #1402)
- [x] Define scope for: Add unit tests for key resolution logic in `dashboard/components/llm_settings.py` covering valid (verify: tests pass).
- [x] Implement focused slice for: Add unit tests for key resolution logic in `dashboard/components/llm_settings.py` covering valid (verify: tests pass).
- [x] Validate focused slice for: Add unit tests for key resolution logic in `dashboard/components/llm_settings.py` covering valid (verify: tests pass).
- [x] Define scope for: Add unit tests for sanitization logic in `dashboard/components/llm_settings.py` covering special characters (verify: tests pass).
- [x] Implement focused slice for: Add unit tests for sanitization logic in `dashboard/components/llm_settings.py` covering special characters (verify: tests pass).
- [x] Validate focused slice for: Add unit tests for sanitization logic in `dashboard/components/llm_settings.py` covering special characters edge cases (verify: tests pass).

### Verification (Keepalive Next Task #1402)
- [x] `pytest tests/test_dashboard_llm_settings.py -m "not slow"` (56 passed).

## Task Reconciliation (Keepalive Next Task #1403)
- [x] Reviewed recent commits (`fcaf87b`, `ea76def`, `1360476`) and reconciled checkbox state for llm-settings follow-up test work.
- [x] Added additional diff-formatting unit coverage for `format_config_diff(...)` in `tests/test_llm_compare_runs.py` for matching manifests, missing manifest inputs, and wizard add/remove paths.

### Verification (Keepalive Next Task #1403)
- [x] `pytest tests/test_llm_compare_runs.py -m "not slow"` (11 passed).

## Task Reconciliation (Keepalive Next Task #1406)
- [x] Reviewed recent commits (`0a1e27f`, `95c1279`, `1f473eb`) and reconciled checklist state before continuing.
- [x] Confirmed prior `dashboard/components/llm_settings.py` key-resolution/sanitization subtasks were already completed and reflected in PR tracking.
- [x] Define scope for: Create fixture data with a small fake summary DataFrame for metric extraction tests.
- [x] Implement focused slice for: Create fixture data with a small fake summary DataFrame for metric extraction tests.
- [x] Validate focused slice for: Create fixture data with a small fake summary DataFrame for metric extraction tests.
- [x] Define scope for: Add unit tests for basic metric extraction from summary DataFrames in `pa_core/llm/result_explain.py`.
- [x] Implement focused slice for: Add unit tests for basic metric extraction from summary DataFrames in `pa_core/llm/result_explain.py`.
- [x] Validate focused slice for: Add unit tests for basic metric extraction from summary DataFrames in `pa_core/llm/result_explain.py`.
- [x] Define scope for: Add unit tests for edge cases in metric extraction such as missing columns or null values.
- [x] Implement focused slice for: Add unit tests for edge cases in metric extraction such as missing columns or null values.
- [x] Validate focused slice for: Add unit tests for edge cases in metric extraction such as missing columns or null values.
- [x] Resolved truncated task reference `pa_core/l_...` to `pa_core/llm/tracing.py` based on active LLM module scope and prior task context.
- [x] Added a corresponding unit test in `tests/test_llm_tracing_noop.py` for run-object URL resolution via `get_url()`.

### Verification (Keepalive Next Task #1406)
- [x] `pytest tests/test_llm_result_explain_entrypoint.py -m "not slow"` (5 passed).
- [x] `pytest tests/test_llm_tracing_noop.py -m "not slow"` (13 passed).

## Task Reconciliation (Keepalive Next Task #1407)
- [x] Reviewed recent commits (`3ad6989`, `39ec437`, `b345953`) and reconciled missing checkbox updates.
- [x] Validated key-resolution follow-up for invalid env-var format inputs in `tests/test_dashboard_llm_settings.py`.
- [x] Implemented a focused sanitization hardening slice in `dashboard/components/llm_settings.py` to neutralize non-printable characters before masking.
- [x] Added focused sanitization tests for control/ANSI-character edge cases in `tests/test_dashboard_llm_settings.py`.
- [x] Validated the sanitization slice with targeted test execution.

### Verification (Keepalive Next Task #1407)
- [x] `pytest tests/test_dashboard_llm_settings.py -m "not slow"` (64 passed).

## Task Reconciliation (Keepalive Next Task #1408)
- [x] Reviewed recent commits (`6d33a7e`, `3ad6989`, `39ec437`) and reconciled checklist tracking for the latest llm-settings follow-up.
- [x] Added focused sanitization coverage in `tests/test_dashboard_llm_settings.py` for NUL (`\\x00`) and DEL (`\\x7f`) control-character handling.
- [x] Validated the added slice with targeted test execution.

### Verification (Keepalive Next Task #1408)
- [x] `pytest tests/test_dashboard_llm_settings.py -m "not slow"` (65 passed).

## Task Reconciliation (Keepalive Next Task #1404)
- [x] Reviewed recent commits (`1d1a07c`, `b0828b3`, `a377eea`) and reconciled checkbox state for Config Chat backend groundwork.
- [x] Marked complete in PR tracking: created `pa_core/llm/config_patch.py` schema/patch validation allowlist and tests.
- [x] Marked complete in PR tracking: created `pa_core/llm/config_patch_chain.py` prompt builder, output parsing (`patch`, `summary`, `risk_flags`), and LangSmith trace URL capture with tests.

### Verification (Keepalive Next Task #1404)
- [x] `pytest tests/test_llm_config_patch.py tests/test_llm_config_patch_chain.py -m "not slow"` (pass).

## Task Reconciliation (Keepalive Next Task #1405)
- [x] Implemented `dashboard/components/config_chat.py` with Streamlit controls for instruction input, `Preview`, `Apply`, `Apply+Validate`, and `Revert`.
- [x] Added component tests in `tests/test_dashboard_config_chat.py` covering preview, apply/apply+validate callback wiring, and revert wiring.
- [x] Updated task checkboxes for issue #1403 scope in `Issues.txt` to reflect completed `config_patch_chain` and `config_chat` UI-control items.

### Verification (Keepalive Next Task #1405)
- [x] `pytest tests/test_dashboard_config_chat.py -m "not slow"` (3 passed).
