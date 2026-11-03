# Issue #2360 closer repair evidence

Records the missing acceptance evidence for merged
[PR #2372](https://github.com/stranske/Portable-Alpha-Extension-Model/pull/2372)
and follow-on wizard render tests on `main` (`#2397`, `#2411`). Production
`dashboard/pages/3_Scenario_Wizard.py` is unchanged in this repair.

## Acceptance contract (checked 2026-11-03)

Outcome 1: successful final-head pytest and formatter CI, with accurate ranking
and coverage attribution. Read real run input with:

```sh
gh run view <run-id> --repo stranske/Portable-Alpha-Extension-Model \
  --json headSha,status,conclusion,url,jobs
```

Stop after two repair attempts or 60 minutes; report the first broken step.
Blocking/drainable pair: final-head CI pending / correct documentation and run
focused checks while GitHub executes the full suite. No local full-suite run.

## Focused tests (prior repair, retained)

```text
$ .venv/bin/python -m pytest -q tests/test_wizard_render_coverage.py
17 passed, 2 warnings in 21.02s
exit=0
```

## Formatter (prior repair, retained)

```text
$ .venv/bin/ruff check tests/test_wizard_render_coverage.py
All checks passed!
$ .venv/bin/ruff format --check tests/test_wizard_render_coverage.py
1 file already formatted
```

## Historical cumulative coverage pairing

Baseline: `10ac6839518576d4dc5199a24392eadfc388c713` (parent of #2372 merge).
Head: `cc7042594111cf6dc7ca4728c72e7ab271a0330f` (historical pre-#2461 `main`, not the current repair head).

Previously recorded Gate pytest replay (macOS, Python 3.14.3; omits CI-only
observers). Retained as historical evidence, not rerun by this repair:

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

These are **cumulative** gains over
`10ac6839518576d4dc5199a24392eadfc388c713..cc7042594111cf6dc7ca4728c72e7ab271a0330f`,
not an isolated measurement of #2372. The range contains:

- [#2372](https://github.com/stranske/Portable-Alpha-Extension-Model/pull/2372):
  `10ac6839518576d4dc5199a24392eadfc388c713..e8967edb2aa1bdbc1a68ea58ed241cf205915d03`
  (original eight test cases).
- [#2397](https://github.com/stranske/Portable-Alpha-Extension-Model/pull/2397):
  `a0849defa30f73a7e6afbe5bb2ccb677e40494ab..d82092b74c248cd73a8834e2b67d7d78f3754b6f`
  (Black formatting only; no added coverage logic).
- [#2411](https://github.com/stranske/Portable-Alpha-Extension-Model/pull/2411):
  `a2c4233ca57d8c02711973df2d6a1ffed2de0d73..63e93a89f9b159c844ddab9a3842e4bc5a56bf7a`
  (nine more test cases).

Other intervening changes also belong to the cumulative range. No isolated
full-suite before/after measurement for #2372 is available here. Its PR reports
54 previously missing lines exercised by the added tests, explicitly a crosswalk
projection, not a measured full-suite gain. Do not credit it with all 321 units
or infer its isolated blended gain from these cumulative results.

The authoritative Gate reads `coverage.xml` `line-rate` with `branch=false`
(`pyproject.toml`). The branch column is a supplemental measurement using the
same test selection and answers the issue's blended units without changing Gate
policy.

## Ranking

See [ranking-provenance.json](./ranking-provenance.json). The live
[#2360 dispatch](https://github.com/stranske/Portable-Alpha-Extension-Model/issues/2360)
reports escaped-defect priority, then churn, then uncovered mass and lists values
1 / 1 / 338 for the wizard. This is a dispatch claim, not independent historical
selection proof. The original candidate set, identified fix commit, incident
mapping, history window and executable ranking receipt are **UNKNOWN** here.

The independently retained [ranking archive](../issue-2336-next-boundaries/ranking.json)
uses the last 500 commits touching `pa_core` or `dashboard`, sorted by repair-subject
count, churn, then uncovered statements. Wizard ranks **second**, with **23 / 65 /
338**; CLI ranks first with 26 / 132 / 312. Repair subjects are only an escaped-defect
proxy, not verified incidents. This older archive corroborates risk but cannot
reconstruct #2360's later dispatch (different counts and no matching snapshot).
The former 312 wizard count was copied from the CLI row and is corrected.

## Mutations

17/17 nodes fail on deliberate break and pass after restoration. Details:
[mutation-transcripts.md](./mutation-transcripts.md), [mutations.json](./mutations.json).

## Reported CI failure and current validation

Inspected `.github/workflows/ci.yml`, the reusable workflow, and the actual
[failed lint-format log](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/38977412115/job/117230149240)
before editing. At merge `e8967edb2aa1bdbc1a68ea58ed241cf205915d03`, Black reported:

```text
would reformat .../tests/test_wizard_render_coverage.py
1 file would be reformatted, 496 files would be left unchanged.
Process completed with exit code 1.
```

#2397 already fixed the two line wraps. Its
[dispatched CI](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/39140761094)
passed on `23da41632fd7607f2d49d7547de1d74a5aefb517`, including lint-format and
pytest (then Python 3.12). No formatter failure reproduces on current main, so
this repair changes no Python. Gate disables Black (`format_check: false`);
passing Ruff alone does not verify that historical Black failure.

Local check on 2026-11-03: reused the repository-selected opener `.venv`
(CPython 3.14.3, Black 26.10.1) through a temporary symlink (removed after validation); no installation
or changes to that environment. Exact formatter command:

```text
$ .venv/bin/python -m black --check --line-length 100 --exclude '(\.venv|\.workflows-lib|node_modules)' .
All done!
521 files would be left unchanged.
$ .venv/bin/python -m pytest -q tests/test_wizard_render_coverage.py
17 passed, 2 warnings in 2.30s
```

Black skipped notebooks because its optional Jupyter dependencies are absent;
the failed file is Python and was checked. The two pytest warnings concern legacy
return inputs. The full workflow pytest command runs only in GitHub, respecting
the prohibition on local full-suite/release verification. The reusable workflow
builds this invocation (default marker is empty; pyproject excludes live LLM):

```sh
pytest -p ci_test_growth --junitxml=pytest-junit.xml --cov \
  --cov-config=pyproject.toml --cov-report=xml:coverage.xml \
  --cov-report=term-missing --cov-report=json:coverage.json \
  -n auto --dist=loadgroup
```

### Successful CI receipt (2026-11-03)

Validated evidence head: `d98fed55d7cfa17ce40171282db4a7a60b6deaf0`.
[Dispatched CI run 41660173230](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/41660173230)
completed **successfully** on that exact SHA:

```text
Python CI / lint-format: success (Black; 521 files unchanged)
Python CI / lint-ruff: success
Python CI / typecheck-mypy: success
Python CI / python 3.14: success (runner CPython 3.14.9)
1969 passed, 1 skipped, 995 warnings in 92.78s
Coverage 91.25% meets minimum 60.00%
```

[Test job](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/41660173230/job/126001402506)
and [Black job](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/41660173230/job/126001094995).
[PR Gate 41660160636](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/41660160636)
and [dispatched Gate 41660180292](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/41660180292)
also passed on this SHA, but both classified the change as docs-only and skipped
pytest. They are not the source of the pytest result above. CI's 60% floor is
unchanged; observed coverage also exceeds the configured PR Gate floor of 84%.

The receipt-only commit changes this README after that tested SHA. Its own SHA
cannot be embedded in itself. Final-head CI is rerun after the receipt push;
the exact final SHA, immutable run URL and results are recorded in
[PR #2467](https://github.com/stranske/Portable-Alpha-Extension-Model/pull/2467).
Resolve that final-head binding directly (do not treat a prior-head run as current):

```sh
gh pr view 2467 --repo stranske/Portable-Alpha-Extension-Model \
  --json headRefOid,state,isDraft,body
gh run list --repo stranske/Portable-Alpha-Extension-Model \
  --workflow ci.yml --branch closer-r2/issue-2360 \
  --json databaseId,headSha,status,conclusion,url
```

Require a successful completed CI run whose `headSha` equals `headRefOid`.
The historical cumulative coverage table above is retained unchanged; this CI
percentage is a current result, not an isolated #2372 coverage gain.
