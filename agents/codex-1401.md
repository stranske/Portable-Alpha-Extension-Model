<!-- autofix diagnostics for PR #1401 -->

## Autofix Attempt Summary

- Gate run: `https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/21926367755`
- Head SHA: `a07df1b8055472fc13d53b9b8551a15afe857206`
- Reported conclusion: `cancelled`
- Reported failing jobs: none

## Local Repro Check

No reproducible failure was found on this head.

Commands run:

- `python -m pytest -q tests/test_dashboard_config_chat.py tests/test_llm_config_patch.py tests/test_llm_config_patch_chain.py tests/test_wizard_config_chat_flow.py`
  - Result: `20 passed, 2 warnings`
- `python -m ruff check dashboard/components/config_chat.py dashboard/pages/3_Scenario_Wizard.py pa_core/llm/config_patch.py pa_core/llm/config_patch_chain.py pa_core/llm/__init__.py tests/test_dashboard_config_chat.py tests/test_llm_config_patch.py tests/test_llm_config_patch_chain.py tests/test_wizard_config_chat_flow.py`
  - Result: `All checks passed!`

## Outcome

No source change required from this autofix attempt because the referenced CI run did not report a code failure.
