# Workloop State

## 2026-05-31T04:08Z - codex opener materialized issue #1837 golden gate hard-fail

- Automation: `pd-workloop-resume` opener lane from neutral Code workspace.
- Source repo: `stranske/Portable-Alpha-Extension-Model`.
- Source issue: [#1837](https://github.com/stranske/Portable-Alpha-Extension-Model/issues/1837), branch `codex/issue-1837-golden-hardfail`.
- Worktree: `/Users/teacher/.codex/automations/pd-workloop-resume/worktrees/paem-1837-golden-hardfail`.
- Implementation: removed the golden-test soft-fail fallback from `.github/workflows/pr-00-gate.yml` so `pytest tests/golden/ -v --tb=short` fails the `integration-tests` job directly; documented in `tests/golden/README.md` that golden failures hard-fail the required `Gate / gate` status instead of becoming warning annotations.
- Validation: `python -m pytest tests/golden/test_scenario_smoke.py -q --tb=short` passed; deliberate-break check changed `EXPECTED["terminal_AnnReturn"]` to `0.123` and confirmed the same test failed, then restored it; parsed `.github/workflows/pr-00-gate.yml` with PyYAML and asserted `summary.needs` contains `integration-tests`; `grep -c '|| echo "::warning::Some golden tests failed"' .github/workflows/pr-00-gate.yml` returned `0`; `git diff --check` passed.
- Local environment note: full `python -m pytest tests/golden/ -v --tb=short` is blocked on this machine by global NumPy 2.4.6 incompatibility with compiled pandas/pyarrow/plotly dependencies (`np.unicode_`/`_ARRAY_API` errors), not by the workflow change. CI installs repo requirements in a fresh environment.
- Next action: push and open ready-for-review PR with `agent:codex`, `agents:keepalive`, and `autofix`; keepalive owns CI/review follow-up after PR creation.
