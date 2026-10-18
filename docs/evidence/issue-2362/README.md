# Comparison question propagation (#2362)

The Questions widget now reaches the provider prompt through the real comparison helper, alongside both metric catalogs. Blank strings normalize to the existing default question text, shared with the dashboard, heuristic fallback and exported payload. Model selection, metrics and trace handling remain unchanged.

`.venv/bin/python -m pytest tests/test_llm_prompts.py tests/test_llm_compare_runs.py tests/test_dashboard_comparison_llm.py -q`: **47 passed** offline. Ruff, isort, Black, and mypy on both core source files pass. The end-to-end panel test uses real manifest/summary loading and captures the provider invocation; changing only the question changes the prompt. Provider transport is fake and no live LLM quality is claimed.

`dashboard-question-red-green.txt` records actual failure after disconnecting the dashboard forwarding call, then success after exact-byte restoration. Hosted checks, review and original-issue verification remain separate stages.
