# PR #2498 continuation evidence — 2026-11-11

The pre-edit acceptance outcome was that pandas would retain its native all-Series
construction and duplicate-label error, mixed Series/dict rows with duplicate
Series labels would raise before dictionary conversion, and mixed rows with unique
labels would retain their existing records. The focused limit was 120 seconds per
invocation. Each hosted revision and coverage configuration gets one attempt
with a 180-second test limit.

## Focused proof

Repository-selected Python 3.14 environment available on this host was
`/private/tmp/paemv/bin/python` (3.14.3, pandas 3.1.1, pytest 9.2.0).
The named regression command, with that interpreter as `python`:

```sh
timeout -k 5s 120s python -m pytest -q tests/test_coverage_round_2476_facade.py -k 'duplicate_label' -m 'not slow and not live_llm'
```

The production mutation replaced the repaired facade with the exact file from
`496cf6c7884ada74bc1e0586075af28d91a3d486`, reversing
[the facade fix](facade-fix.diff). Both named regressions printed
`DID NOT RAISE` in [red](red.txt). Restoring the repaired bytes produced
[matching SHA-256 values](hashes.txt) and [two passing regressions](green.txt).

```sh
timeout -k 5s 120s python -m pytest -q tests/test_coverage_round_2381.py tests/test_coverage_round_2476_facade.py tests/test_coverage_round_2476_evidence.py tests/test_coverage_round_2476_mutation_transcripts.py -m 'not slow and not live_llm'
```

[Focused acceptance](focused.txt) printed 80 passed and four warnings in 1.68 seconds.

GitHub keepalive advanced the existing branch to
`913ae5cdd655459d25604390403c22a258cc82bf` during the focused work. The
continuation fast-forwarded to retain its commits. The same command then
[printed 87 passed and four warnings in 1.92 seconds](focused-current.txt).

## Hosted whole-suite comparison

[GitHub Actions run 43113742075](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/43113742075)
checked out the [measurement contract](../measurement-20261111/contract.json)
baseline `def0eec035bdccfdebce62c0116aa468ed6fdd2f` and candidate source
`4b1f5e4c4b5b1135fecf84adb04b44b8d15d7fe3`. The candidate includes the
keepalive tests and the facade repair. The subsequent evidence commit adds
receipts under this directory. The hosted runner definition lived on a
temporary measurement ref.

```sh
python -m pytest --cov --cov-config=pyproject.toml --cov-report=xml:coverage.xml --cov-report=term-missing --cov-report=json:coverage.json -n auto --dist=loadgroup -m 'not slow and not live_llm'
```

The separate branch diagnostic appended `--cov-branch`. Each of the four jobs
ran once with `timeout -k 5s 180s` and `PYTEST_ADDOPTS=-vv` for named
progress. All ran on the same hosted image and Python 3.14.10 with two xdist
workers, matching coverage configuration and dependency input hashes, and
identical resolved packages apart from the revision of the editable project.

| Revision | Configuration | Pytest result |
| --- | --- | --- |
| Baseline | Gate | 2,045 passed, 1 skipped in 133.77s |
| Baseline | Branch supplement | 2,045 passed, 1 skipped in 76.15s |
| Candidate | Gate | 2,057 passed, 1 skipped in 120.45s |
| Candidate | Branch supplement | 2,057 passed, 1 skipped in 126.59s |

Each run printed 1,014 warnings. The previously unfinished
`tests/test_cli.py::test_main_with_png` passed in all four logs. Commands,
revision and input hashes, environments, exits, named logs, and
[seven-target coverage extracts](hosted/seven-targets.json) are in [hosted/](hosted/).
The extracts identify the source JSON by SHA-256. The full JSON and XML
reports are available in the Actions artifacts.

| Target | Gate uncovered / total statements, baseline → candidate | Diagnostic uncovered / total branches, baseline → candidate |
| --- | ---: | ---: |
| `pa_core/llm/result_explain.py` | 0/227 → 0/227 | 0/102 → 0/102 |
| `pa_core/pa.py` | 0/222 → 0/222 | 1/68 → 1/68 |
| `pa_core/facade.py` | 2/333 → 2/337 | 2/120 → 2/124 |
| `pa_core/units.py` | 0/81 → 0/81 | 0/30 → 0/30 |
| `pa_core/sim/internal_pa_financing.py` | 0/35 → 0/35 | 0/18 → 0/18 |
| `pa_core/manifest.py` | 0/61 → 0/61 | 0/4 → 0/4 |
| `dashboard/pages/4_Results.py` | 0/210 → 0/210 | 1/70 → 1/70 |
| **Total** | **2/1,169 → 2/1,173** | **4/412 → 4/416** |

### Round attribution

The historical [round snapshots](../gate-coverage.json) record the progression
from the parent of #2384 through its merge, the original #2476 bootstrap,
and the earlier delivered test revision. Their combined change was 326 fewer
uncovered units, exceeding the approximately 180-unit round objective under
that historical test policy.

| Historical step | Fewer uncovered statements | Fewer uncovered branches |
| --- | ---: | ---: |
| `bf424ec6` → `1b7f97a1` (#2384) | 58 | 14 |
| `1b7f97a1` → `cb7143b4` (intervening changes) | 131 | 79 |
| `cb7143b4` → `54219bd7` (earlier #2476 work) | 18 | 26 |
| **Historical round total** | **207 (209 → 2)** | **119 (123 → 4)** |

The historical snapshots included slow tests, had some environment-related
test failures, and used a different worker setting for the #2384 branch run.
The hosted comparison above excludes slow tests and measures an already
improved baseline. It independently confirms the same final uncovered counts
and **zero additional reduction** from the contract baseline to this
candidate. Four new facade statements and four new facade branches were
covered. A uniform hosted remeasurement of the historical attribution is
outside the two-revision measurement contract. #2384 exercised target 7
before targets 1–6. Its reason for that selection remains unknown.

### Residual exceptions at the candidate

- `pa_core/facade.py` statements 751 and 757 and arcs 750→751 and 756→757.
  The export path receives the already serialized value, so the DataFrame
  body and tuple conversion cannot be reached through that path.
- `pa_core/pa.py` arc 349→exit. Argument parsing requires a registry
  subcommand and accepts only `list` and `get`.
- `dashboard/pages/4_Results.py` arc 178→180. Every unavailable previous-run
  result carries a message.
- All other ranked targets had zero uncovered statements and branches.

The explicit format-enabled [CI run 43113578331](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/43113578331)
on the measured candidate executed and passed both the pytest and lint-format
jobs. The evidence commit receives its own exact-head CI dispatch.
