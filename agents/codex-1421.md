<!-- autofix diagnostics for PR #1421 -->

## Autofix Attempt Summary

- Gate run: `https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/21950876920`
- Head SHA: `528119c7d96c8458bea0836bde574185e58086cf`
- Reported conclusion: `cancelled`
- Reported failing jobs: none

## Local Repro Check

No reproducible failure was found on this head.

Commands run:

- `python -m pytest -q`
  - Result: `1091 passed, 1 skipped, 335 warnings`
- `python -m pytest -q tests/test_llm_tracing_noop.py tests/test_tracing.py tests/test_result_explain.py`
  - Result: `39 passed`
- `python -m ruff check pa_core/llm/tracing.py pa_core/llm/result_explain.py pa_core/llm/compare_runs.py tests/test_llm_tracing_noop.py tests/test_tracing.py tests/test_result_explain.py`
  - Result: `All checks passed!`

## Outcome

No source change required from this autofix attempt because the referenced CI run did not report a code failure.
