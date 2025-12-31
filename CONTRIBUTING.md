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

## Helpful references

- `docs/INSTRUCTIONS_FOR_CODEX.md` for workflow expectations and test guidance.
- `docs/AGENT_ISSUE_FORMAT.md` for the issue template used by the agent pipeline.
