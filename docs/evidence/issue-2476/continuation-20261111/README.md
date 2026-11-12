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

## Current-main PR delta (2026-11-11, second hosted run)

The baseline above, `def0eec0`, is a commit on this PR branch, not main. To
measure the PR's own delta, [run 43122605282](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/43122605282)
used the same [runner definition](hosted-main/workflow.yml.txt). Only the
matrix revisions changed. It measured current main
`5be77e15c925f5354ab67e2a1749fe6187d2038c` and the prospective merge. Main's
two commits since the merge base add tests (`tests/test_issue_2360_evidence.py`,
`tests/test_ci_python_standard.py`), so the merge changes the candidate's
coverage inputs. The prospective merge tree is
`ab96b5b495547fcad317da7c135d9b082aa0e150` (tree `c9713981`, parents `4b1f5e4c`
and `5be77e15`). It is a measurement-only commit and was not pushed to this
branch. Later commits on this branch change only `docs/evidence/` and
`tests/test_coverage_round_2476_evidence.py`, which reads evidence text and
executes no target module.

The command, Python 3.14.10, two-worker image, input hashes and resolved
packages matched run 43113742075. Each job had one attempt with the
180-second limit. The receipts are in [hosted-main/](hosted-main/), including
the [seven-target extracts](hosted-main/seven-targets.json).

| Revision | Configuration | Pytest result |
| --- | --- | --- |
| Main `5be77e15` | Gate | 2,120 passed, 1 skipped in 117.49s |
| Main `5be77e15` | Branch supplement | 2,120 passed, 1 skipped in 114.54s |
| Merge `ab96b5b4` | Gate | 2,182 passed, 1 skipped in 80.02s |
| Merge `ab96b5b4` | Branch supplement | 2,182 passed, 1 skipped in 119.14s |

| Target | Gate uncovered / total statements, main → merge | Δ | Diagnostic uncovered / total branches, main → merge | Δ |
| --- | ---: | ---: | ---: | ---: |
| `pa_core/llm/result_explain.py` | 0/227 → 0/227 | 0 | 4/102 → 0/102 | −4 |
| `pa_core/pa.py` | 0/222 → 0/222 | 0 | 2/68 → 1/68 | −1 |
| `pa_core/facade.py` | 2/332 → 2/337 | 0 | 8/120 → 2/124 | −6 |
| `pa_core/units.py` | 0/81 → 0/81 | 0 | 0/30 → 0/30 | 0 |
| `pa_core/sim/internal_pa_financing.py` | 0/35 → 0/35 | 0 | 0/18 → 0/18 | 0 |
| `pa_core/manifest.py` | 0/61 → 0/61 | 0 | 0/4 → 0/4 | 0 |
| `dashboard/pages/4_Results.py` | 18/210 → 0/210 | −18 | 16/70 → 1/70 | −15 |
| **Total** | **20/1,168 → 2/1,173** | **−18** | **30/412 → 4/416** | **−26** |

The merge's seven-target counts, lines and arcs equal the candidate receipts
above, so main's added tests do not change the delivered coverage.

### Reconciliation with the historical round totals

Hosted main reproduces the historical `before_this_work` snapshot at
`cb7143b4` target by target: 20 statements and 30 branches. The measured PR
delta of 18 statements and 26 branches therefore equals the historical
"earlier #2476 work" step under the matched policy.

| Step | Statements | Branches | Policy of the source |
| --- | ---: | ---: | --- |
| `bf424ec6` → `1b7f97a1` (#2384) | 58 | 14 | Historical snapshots only |
| `1b7f97a1` → `cb7143b4` (intervening changes) | 131 | 79 | Historical snapshots only |
| Main `5be77e15` → merge `ab96b5b4` (this PR) | 18 | 26 | Hosted Gate and branch receipts (current) |
| **Round total** | **207 (209 → 2)** | **119 (123 → 4)** | Endpoints: historical start, hosted finish |

The PR alone removes 44 uncovered units, well short of the ~180-unit round
objective. The objective is met only with the 189 statements and 93 branches
that #2384 and the intervening changes removed. Those amounts, and the round
baseline of 209 statements and 123 branches, still come from the historical
test policy: slow tests included, environment-related failures, and a
different worker setting for the #2384 run.

To anchor the round baseline under the matched policy,
[run 43123043013](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/43123043013)
made one attempt per configuration at `bf424ec6ff58bb88b6431acf7809d42b2c59f95d`
([runner definition](hosted-anchor/workflow.yml.txt)). That revision predates
`ci-python-versions.env`, so the runner fixed the matched tool versions
explicitly. Both jobs failed at **Install the repository test environment**:
`Because you require coverage==7.17.2 and coverage==7.16.2, we can conclude
that your requirements are unsatisfiable`. The anchor's `requirements.lock`
pins `coverage==7.16.2`
([log excerpt](hosted-anchor/failed-install.txt)). No tests ran. Per the
one-attempt rule the anchor was not retried under that forced `7.17.2` pin, so
that forced-pin round baseline remains **unmeasured**. A matched anchor would
need either the anchor's own coverage pin, which is no longer the matched
policy, or an overridden lock.

Subsequently, [run 43154194839](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/43154194839)
used `.github/workflows/issue-2360-paired-ci-attribution.yml` with runner-local
dependency resolution (`source .github/workflows/autofix-versions.env` plus
`requirements.lock` from each checked-out revision on `ubuntu-latest` Python
`3.14.10`) to execute the exact Gate command and `--cov-branch` supplement on
baseline `bf424ec6ff58bb88b6431acf7809d42b2c59f95d`, PR `#2384` merge
`1b7f97a1c640ef69efe6a6750cf93f23a105000e`, and prospective delivered merge
`d0e8f8058a0c8f32babf4460390bd2073ed5389a`. All six jobs completed on the first
attempt within 180 seconds; receipts and per-step target reconciliations are in
[hosted-main/seven-targets.json](hosted-main/seven-targets.json).

The measured residual exceptions at the merge are unchanged from the candidate:
facade statements 751 and 757 with arcs 750→751 and 756→757, `pa.py` arc
349→exit, and `4_Results.py` arc 178→180.

After these evidence and test updates, the focused command printed
[90 passed and four warnings](focused-final.txt) under the 120-second limit.
