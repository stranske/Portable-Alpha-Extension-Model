# Issue #2360 closer repair evidence

Records the missing acceptance evidence for merged
[PR #2372](https://github.com/stranske/Portable-Alpha-Extension-Model/pull/2372)
and follow-on wizard render tests on `main` (`#2397`, `#2411`). Production
`dashboard/pages/3_Scenario_Wizard.py` is unchanged in this repair.

## Focused tests

```text
$ .venv/bin/python -m pytest -q tests/test_wizard_render_coverage.py
17 passed, 2 warnings in 21.02s
exit=0
```

## Formatter

```text
$ .venv/bin/ruff check tests/test_wizard_render_coverage.py
All checks passed!
$ .venv/bin/ruff format --check tests/test_wizard_render_coverage.py
1 file already formatted
```

## Gate coverage pairing

Baseline: `10ac6839518576d4dc5199a24392eadfc388c713` (parent of #2372 merge).
Head: `cc7042594111cf6dc7ca4728c72e7ab271a0330f` (current `main` at repair time).

Gate pytest (local replay, macOS, Python 3.14.3; omits CI-only observers):

```sh
.venv/bin/python -m pytest -m "not live_llm" --cov --cov-config=pyproject.toml \
  --cov-report=json:coverage.json -q
```

| Metric | Baseline | Head | Reduction |
| --- | ---: | ---: | ---: |
| Repo missing statements (Gate `branch=false`) | 2368 | 1381 | 987 |
| `3_Scenario_Wizard.py` missing statements | 338 | 101 | **237** |
| `3_Scenario_Wizard.py` missing branches (`--cov-branch` overlay) | 125 | 41 | **84** |
| Wizard statement + branch units | 463 | 142 | **321** |

The authoritative Gate reads `coverage.xml` `line-rate` with `branch=false`
(`pyproject.toml`). The branch column is a supplemental measurement using the
same test selection and answers the issue's blended units without changing Gate
policy.

## Ranking

See [ranking-provenance.json](./ranking-provenance.json). Issue #2360 names a
single target (`3_Scenario_Wizard.py`, 338 blended units). Archived
`docs/evidence/issue-2336-next-boundaries/ranking.json` corroborates high churn
and uncovered mass for the same path.

## Mutations

17/17 nodes fail on deliberate break and pass after restoration. Details:
[mutation-transcripts.md](./mutation-transcripts.md), [mutations.json](./mutations.json).
