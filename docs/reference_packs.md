# Reference Packs for Agent Runs

Portable Alpha defines curated reference packs in `.github/reference_packs.json`.

The `trend_streamlit_llm` pack pins a specific Trend commit and a small set of
files that Codex can copy as implementation patterns during keepalive runs. This
keeps prompts focused and avoids pulling unrelated source trees into context.

In keepalive automation, Workflows materializes the pack under
`.reference/trend_streamlit_llm/` at runtime. That directory is runtime-only and
should not be committed.
