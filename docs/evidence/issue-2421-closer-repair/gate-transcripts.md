# Gate transcripts

Check date: 2026-11-02 America/Chicago. Commands run from unmodified Git checkouts at the exact SHAs below. Warning sections omitted.

```sh
/tmp/2421-repair/historical-env/bin/python -m pytest --junitxml=pytest-junit.xml --cov --cov-config=pyproject.toml --cov-report=xml:coverage.xml --cov-report=term-missing --cov-report=json:coverage.json -n 2 --dist=loadgroup
```

CI source: final #2422 Gate run 39497784103, job 118887469164; reusable workflow d2def1901357855222046823b609a8f87b95ccfb. CI adds a receipt observer (`-p ci_test_growth`); see the environment limits in gate-measurements.json.

## before

543b53597014100c69d12635b428c85ac791aaeb

```text
platform darwin -- Python 3.12.2, pytest-9.1.1, pluggy-1.6.0
2 workers [1922 items]
pa_core/llm/result_explain.py                227      3    99%   445-446, 456
pa_core/sim/internal_pa_financing.py          35      1    97%   81
pa_core/sim/metrics.py                       288     29    90%   58, 69, 117, 152, 203, 270, 299, 342, 345, 375, 402, 404, 420, 422, 428, 431, 493, 517, 519, 521, 523, 525, 530-534, 567, 569
pa_core/units.py                              81      2    98%   200, 214
TOTAL                                      14775   1392    91%
========== 1921 passed, 1 skipped, 953 warnings in 170.85s (0:02:50) ===========
exit=0
```

## after

127f111898a90ee46fb6538f05ede021dd4541d9

```text
platform darwin -- Python 3.12.2, pytest-9.1.1, pluggy-1.6.0
2 workers [1929 items]
pa_core/llm/result_explain.py                227      0   100%
pa_core/sim/internal_pa_financing.py          35      0   100%
pa_core/sim/metrics.py                       288      1    99%   270
pa_core/units.py                              81      0   100%
TOTAL                                      14775   1358    91%
========== 1928 passed, 1 skipped, 953 warnings in 170.86s (0:02:50) ===========
exit=0
```

## Restored current-main baseline

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2421.py
7 passed, 14 warnings in 2.21s
exit=0
```

## Discarded first replay

Archive snapshots under Python 3.14.3 had four missing-baseline-kit collection errors and four failures (Git metadata, two ensurepip subprocesses, Python-dependent schema formatting) in each revision. Those incomplete runs are not used for the coverage table. Repeated with real checkouts, the historical lockfile, matching baseline-kit commit and Python 3.12.2; both corrected runs pass. No product code was repaired.
