# Unblocked local Gate measurement, 2026-11-12 UTC

Pre-edit baseline: `6d4d137aa10ec0b4b8153d8b2141c92723058a59`.
Measured source commit: `6053f77ee4fecb4ee4659f57f1bb4b021331fc42`.
Available local main: `9d083d69796126c41d3a05aa3499cb4fe8cc11eb`.
Interpreter: `/opt/hostedtoolcache/Python/3.14.10/x64/bin/python`, Python 3.14.10.
This continuation changes only `tests/test_cli.py`; no production change is retained.

## First unchecked task: discovery and execution contract

Read the repository instructions, target-environment and issue-format guides,
the current tests, main references, prior evidence and the five-file #2473 diff.
The earlier candidate review remains applicable: #2473 merged as `7613ffd9`;
its description as unlanded is stale. Before editing, the six-module focused
candidate scope recorded in [measurements.json](measurements.json) passed
**156 cases in 8.00s**, including target 1 at 100% statement/branch coverage.

Repeated `rg -n 'coverage|pytest|lint-format|Gate' .github pyproject.toml`,
then excluded pre-existing untracked node_modules from follow-up searches.
`ci.yml` delegates to Workflows; `pr-00-gate.yml` selects Python 3.14, coverage
and the same reusable workflow. The coverage configuration in `pyproject.toml`
selects `pa_core`, `dashboard`, and `design-system`, with `branch=false`.
The discovered invocation remains:

```sh
python -m pytest --cov --cov-config=pyproject.toml --cov-report=xml:coverage.xml --cov-report=term-missing --cov-report=json:coverage.json -n auto --dist=loadgroup -m 'not slow and not live_llm'
```

[contract.json](contract.json) was written before the test edit. It records
the exact baseline, ranked paths, configuration hash, prior measured reduction,
command and attempt/time stop rule. The pre-edit reduction evidence remains the
hosted main/merge [receipts](../continuation-20261111/hosted-main/seven-targets.json):
**18 statements and 26 branches**, separately attributed to this PR. No new
reduction was assumed when choosing this test repair.

The prior unchanged full-suite attempts were not repeated. One targeted
35-second diagnostic enabled pytest's 10-second faulthandler dump. It timed out
with the CLI inside `Figure.write_image`, waiting in Kaleido's synchronous
thread join; [png-diagnostic.txt](png-diagnostic.txt) preserves the full output.
This establishes the blocking call, not its renderer/OS root cause.

After isolating that call, the first complete full-suite run finished in
82.44s and emitted coverage, with one clean-venv failure caused by PyPI DNS
resolution. Its complete output is in [network-failure.log.gz](network-failure.log.gz).
A targeted rerun passed with pip using the existing virtualenv-bundled
setuptools wheels. The final statement and branch runs use that same offline
environment; wheel hashes and exact commands are recorded in measurements.
No dependency declarations or packaging tests were changed.

Stop rule: focused checks and each mutation have a 120-second cap, with at most
three attempts per failing step. Complete runs have a 180-second cap. The only
full-suite retry followed the verified offline-environment fix; both final
runs passed. Do not repeat the former renderer-dependent run unchanged.

## Test change and mutation checks

`test_main_with_png` retains the real CLI, simulation, XLSX export, populated
chart construction, and image backend. An autospecced `Figure.write_image`
replacement writes sentinel bytes. Assertions check one renderer call, a
populated Plotly figure, the PNG destination, keyword arguments, and the
written bytes. This follows the existing export-backend tests' boundary;
it does not claim to validate the external browser renderer.

[mutations.md](mutations.md) includes complete production diffs and red/green
outputs for three independent mutations: omit the image write, use a PDF
destination, and pass an empty figure. Each fails the named PNG test, then
passes after restoring the exact original production bytes. [mutations.json](mutations.json)
records commands, exits, timings, and before/mutated/restored hashes.

## Current seven-target measurement

Both complete runs use the default tests collection, identical Gate coverage
configuration and xdist scheduler; the branch diagnostic adds `--cov-branch`.
The generated hosted `ci_test_growth` observer and JUnit option are not loaded.
These are local Gate-configured measurements, not hosted required-check results.

| Rank / target | Statements | Missing statements | Branches | Missing branches |
| --- | ---: | ---: | ---: | ---: |
| 1. `pa_core/llm/result_explain.py` | 227 | 0 | 102 | 0 |
| 2. `pa_core/pa.py` | 222 | 0 | 68 | 1 |
| 3. `pa_core/facade.py` | 337 | 2 | 124 | 2 |
| 4. `pa_core/units.py` | 81 | 0 | 30 | 0 |
| 5. `pa_core/sim/internal_pa_financing.py` | 35 | 0 | 18 | 0 |
| 6. `pa_core/manifest.py` | 61 | 0 | 4 | 0 |
| 7. `dashboard/pages/4_Results.py` | 210 | 0 | 70 | 1 |

The complete raw reports are [statements](head-gate.coverage.json.gz) and
[branches](head-branch.coverage.json.gz), with their full [statement](head-gate.log.gz)
and [branch](head-branch.log.gz) run output. Measurements retain uncompressed
report hashes, metadata, totals, missing lines/arcs, and source hashes.
All seven production files match the prior statistics continuation's recorded
source hashes. The table confirms the previously documented saturation before
further ranked test work; no target is skipped on historical counts alone.

The only missing statements remain facade lines 751/757, with arcs
750→751 and 756→757. Serialization has already converted nonempty dataframes
and tuples before those export checks. The pa arc 349→exit would require an
argparse-validated registry subcommand outside its choices. Results' arc
178→180 requires unavailable comparison data with no diagnostic, which its
availability helper does not return. These exceptions remain measured rather
than covered by replacing their own prerequisite implementations. Results now
has **zero missing statements and one missing branch**, not its historical
136 missing units. The original reason #2384 selected target 7 first is unknown.

This test repair contributes **zero ranked coverage units**. The prior PR step
remains 18 statements/26 branches. Earlier attribution remains #2384 at 58/14
and intervening changes at 131/79. Their historical 326-unit round exceeds
approximately 180 under the historical test policy; a matched-policy round
baseline is still unmeasured. The pre-edit focused baseline here is not paired
with the complete head to manufacture a reduction. No complete new baseline
report exists, so matched baseline/head and round-goal acceptance remain open.

## Validation and task disposition

[checks.txt](checks.txt) preserves focused acceptance, the CLI/export suite,
offline packaging verification, Ruff, and Black output:

- Statement suite: **2062 passed, 3 skipped in 65.47s**; branch suite:
  **2062 passed, 3 skipped in 71.55s**.
- Results acceptance: `timeout -k 5s 120s /usr/bin/time -p python -m pytest tests/test_coverage_round_2381.py -m 'not slow and not live_llm' -q`:
  **60 passed in 2.57s**, **4.79s wall**, below two minutes.
- CLI/sweep/image-backend suite: **19 passed in 4.80s**.
- Existing evidence/transcript guards: **18 passed in 1.59s** after the README update.
- `ruff check .`: passes. Black formatted `tests/test_cli.py` at line length 100.
  The exact required `black --check --line-length 100 --exclude '(\.workflows-lib|node_modules)' .`
  passes, **538 files unchanged**, using the previously documented temporary
  multiprocessing/asyncio runtime workaround and `BLACK_NUM_WORKERS=1`.
  The plain invocation's socket permission error and the complete workaround
  source are retained; formatter discovery, safety checks and formatting remain active.
- The original lint-format failure named `tests/test_wizard_render_coverage.py`
  (run 39044091242/job 117429268297), fixed on main in `d82092b7` (#2397).
  Its [archived output](../measurement-20261111/original-lint-format.txt) remains
  authoritative. No unrelated formatting repair was needed here.

Verified local checklist (not a claim that all PR acceptance is complete):

- [x] Gate discovery, pre-edit contract, ranked order, prior reduction evidence,
  stop rule, and runnable acceptance command below two minutes.
- [x] Focused Results acceptance for the unchanged Results test module.
- [x] Production mutation/restoration receipts for this strengthened CLI test.
- [ ] Complete matched-policy baseline/head and approximately 180-unit round evidence.
- [ ] Required hosted CI, including lint-format, on the exact delivered head.

The checkout's `.git` is read-only. Source commit `6053f77e` was created in
`/tmp/issue-2476-delegation/delivery.git` with the pre-edit baseline as parent.
An evidence-only follow-up and `/tmp/issue-2476-delegation/delivery.bundle`
preserve the commits for a writable checkout. Working-tree changes remain
available to automation. No push or workflow edit was performed. PR #2498
was verified open with `draft=false` at remote head `6d4d137a`; its hosted
checks cannot establish CI for the local source commit.

Automatic approval review rejected the attempted PR checklist reconciliation:
`MCP tool call requires approval, but approval policy is never`. The PR body
was not changed. The locally verified checklist above records the disposition;
the matched-policy baseline/head and hosted-CI tasks were left unchecked.
