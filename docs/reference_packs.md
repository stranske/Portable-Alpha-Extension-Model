# Reference Packs for Agent Runs

Portable Alpha defines curated reference packs in `.github/reference_packs.json`.

The `trend_streamlit_llm` pack pins a full Trend commit SHA and a small set of
files that Codex can inspect as implementation patterns during keepalive runs.
The pack is intentionally limited to Streamlit LLM settings, results
explanation, run comparison, chain invocation, tracing, natural-language config
patching, diff/replay surfaces, and the related operator documentation. This
keeps prompts focused and avoids pulling unrelated source trees into context.

In keepalive automation, Workflows materializes the pack under
`.reference/trend_streamlit_llm/` at runtime. That directory is runtime-only and
must not be committed; `.gitignore` excludes `.reference/` for that reason.

Validate the local pack contract before relying on it in an agent run:

```bash
python scripts/reference_packs.py --format self-check
python -m pytest tests/test_reference_packs.py -q
```

After Workflows reference-pack support is available in a keepalive run, confirm
the run workspace contains `.reference/trend_streamlit_llm/` and that the
materialized files match the paths declared in `.github/reference_packs.json`.
