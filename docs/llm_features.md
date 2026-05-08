# Streamlit LLM Features

Portable Alpha keeps dashboard LLM features optional and offline-safe by
default. The Results page can explain a run, compare the current run with a
previous run, and show LangSmith trace links when tracing is enabled. Required
tests use fake clients or heuristic fallbacks and must not call real LLM
providers.

## Install

Install the optional LLM extra before using provider-backed dashboard features:

```bash
python -m pip install -e ".[llm]"
```

The base package can still run the dashboard and tests without this extra. If a
provider package is missing, the Streamlit panels show an install prompt instead
of failing the page import.

## Configuration

Dashboard controls accept either a literal key or an environment-variable name.
Use environment variables for shared or repeated runs:

| Variable | Purpose |
| --- | --- |
| `PA_LLM_PROVIDER` | `openai`, `anthropic`, or `azure_openai`; defaults to `openai`. |
| `PA_LLM_MODEL` | Optional model override for the selected provider. |
| `PA_STREAMLIT_API_KEY` | Dashboard-specific provider key. |
| `OPENAI_API_KEY` | OpenAI fallback key. |
| `CLAUDE_API_STRANSKE` | Anthropic fallback key. |
| `PA_LLM_BASE_URL` | Azure OpenAI endpoint or custom OpenAI-compatible base URL. |
| `PA_LLM_API_VERSION` | Azure OpenAI API version. |
| `AZURE_OPENAI_API_VERSION` | Azure API-version fallback. |
| `LANGSMITH_API_KEY` | Enables LangSmith trace-link resolution. |
| `LANGSMITH_PROJECT` | Optional LangSmith project name. |

Provider keys are masked before display, excluded from TXT/JSON exports, and
redacted from user-visible error messages. Do not log raw key values in tests,
Streamlit messages, or exported artifacts.

## Reference Pack

Agent runs use the curated Trend reference pack declared in
`.github/reference_packs.json`. Workflows materializes it at
`.reference/trend_streamlit_llm/` during keepalive runs so agents can inspect
Trend's Streamlit LLM settings, comparison, tracing, and natural-language config
patterns without copying the entire Trend repository.

Validate the reference-pack contract before relying on it:

```bash
python scripts/reference_packs.py --format self-check
python -m pytest tests/test_reference_packs.py -q
```

The `.reference/` directory is runtime-only and must not be committed.

## Offline Checks

Use these checks after touching LLM dashboard helpers:

```bash
python -m pytest \
  tests/test_dashboard_llm_settings.py \
  tests/test_result_explain.py \
  tests/test_llm_compare_runs.py \
  tests/test_llm_prompts.py \
  tests/test_dashboard_explain_results.py \
  tests/test_dashboard_comparison_llm.py \
  tests/test_reference_packs.py \
  --no-cov
```
