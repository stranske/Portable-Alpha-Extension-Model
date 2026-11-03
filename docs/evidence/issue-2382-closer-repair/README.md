# Issue #2382 closer repair evidence

Checked **2026-11-02, America/Chicago**. Completes the remainder after merged
[PR #2385](https://github.com/stranske/Portable-Alpha-Extension-Model/pull/2385)
without reverting its tests.

## Acceptance (real input)

Outcome: focused module tests plus full Gate-style coverage replay show **70**
fewer missing statements across the five dispatch targets (**123 → 53**), **150**
removed from the issue baseline **203**, **30** short of the **~180** statement
goal. Supplementary `--cov-branch` diagnostic on the same target set: **233 → 107**
combined missing statements+branches (**126** reduction).

Focused command (under two minutes):

```text
$ .venv/bin/python -m pytest -q tests/test_config.py tests/test_llm_config_patch_chain.py tests/test_coverage_round_2382.py
85 passed, 8 warnings in 0.93s
exit=0
```

Stop rule: three failures at the same step or 90 minutes; mutation subprocess
timeout 120s; restore production in `finally`.

## Ranking

[Issue #2382](https://github.com/stranske/Portable-Alpha-Extension-Model/issues/2382)
lists targets in order: `config_patch_chain.py`, `config.py`, `loaders.py`,
`sweep.py`, `calibration.py` (~203 combined uncovered units at dispatch). This
repair adds `tests/test_coverage_round_2382.py` for the ranked tail after #2385
finished `config_patch_chain.py` and partial `config.py`. See
[ranking-provenance.json](ranking-provenance.json).

## Gate measurements

[gate-measurements.json](gate-measurements.json) records before/after JSON from
paired full-suite runs on base `69303735`. Authoritative Gate uses line-only
coverage (`branch = false` in `pyproject.toml`); the branch JSON is a
diagnostic comparable to the issue’s “statement+branch” wording, not the Gate
percentage.

## Mutations

[mutation-transcripts.md](mutation-transcripts.md) — RED/GREEN for repair tests
and representative #2385 tests. [mutation-transcripts-append.md](mutation-transcripts-append.md)
— additional spot checks.

## Production change

Minimal fix in `pa_core/calibration.py`: `corr_target="constant"` no longer calls
`np.fill_diagonal` on a read-only pandas buffer (NumPy 2.x / pandas 3.x).

## Follow-ups (out of scope)

Local full suite still reports two unrelated failures
(`test_dashboard_sample_data`, `test_entrypoints`); not modified here. Remaining
missing lines on the five targets are listed in the repair PR body.
