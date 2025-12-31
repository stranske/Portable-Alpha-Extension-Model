# Contributing

Thanks for contributing. Keep changes small, tested, and scoped to the issue/PR.

## Agent quickstart

1. Create a virtual environment and install dev dependencies:
   - `python -m venv .venv && source .venv/bin/activate`
   - `python -m pip install -e ".[dev]"`
2. Run the quickstart verifier:
   - `python scripts/agent_quickstart.py`
3. Run the focused agent tests or full CI check:
   - `python -m pytest tests/test_agents.py -v`
   - `./dev.sh ci`

Alternative quick commands:
- Set up environment: `./setup.sh` or `make install-dev`
- Quick debug: `make debug-codex`
- Full validation: `make validate-pr`
- Auto-fix and commit: `make debug-codex-fix`
- Details: see `docs/development/AUTOMATION_QUICK_START.md`

## Key files to examine

- `pa_core/reporting/excel.py` - `export_to_excel()` function

## Helpful references

- `docs/INSTRUCTIONS_FOR_CODEX.md` for workflow expectations and test guidance.
- `docs/AGENT_ISSUE_FORMAT.md` for the issue template used by the agent pipeline.
