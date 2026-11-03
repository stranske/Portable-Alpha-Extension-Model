# Coverage round #2418 acceptance evidence

Checked **2026-11-03 (America/Chicago)**. Refs #2418.

This is an evidence-only repair of merged [PR #2420](https://github.com/stranske/Portable-Alpha-Extension-Model/pull/2420).
The five tests and Pillow declaration remain unchanged. No production changes,
threshold changes, exclusions, or deliberate defects are delivered.

## Outcome and bounded execution

Outcome 1: all five merged tests pass on real generated PDF-fallback/Excel/Plotly
and Wizard inputs; each detects its nominated production defect, then passes
following byte-for-byte restoration. Print command:

```sh
/opt/anaconda3/bin/python3 -m pytest -q tests/test_coverage_round_2418.py
```

Stop rule, recorded before experiments: 120 seconds per local experiment, two
failed repair attempts or 75 minutes total. The stop rule did not trip.
Blocking/drainable pair: local full-suite execution is prohibited / recover the
original Gate logs and run configured full CI remotely on the one repair PR.
No local full-suite or release/verification pipeline was executed.

## Revisions and environment

| Role | Exact revision |
| --- | --- |
| Historical baseline: immediate parent of #2420 squash | `982430ef26973ea75b64da5991f78bad161c5c0c` |
| Historical measured head: #2420 squash | `543b53597014100c69d12635b428c85ac791aaeb` |
| Original PR metadata base (older than squash parent) | `80e956cefc75bf3f8033b77b2205f3e51f4ba45e` |
| Original final PR head | `fbad5f414a398b75cbae63fe425e1d45e02fb852` |
| Current main when repair started | `6abacb50a941ce1286e74afb5782d3c804427418` |
| Current local acceptance code after prior-branch merge | `58429899368c56ae80e66bdc387066a1bbf10cfc` |

The last revision includes documentation only beyond main. Its tested
`tests/test_coverage_round_2418.py` is byte-identical to #2420 and main. The
repair PR's final documentation commit is recorded by the PR head; it does not
change the tested code.

No `.venv` exists in this provisioned worktree or its owning repository.
Following the owner runtime rule, local runs used `/opt/anaconda3/bin/python3`
(Python **3.12.2**, pytest **9.1.1**, coverage.py **7.16.2**, macOS/Darwin).
The single-node commands resolved `pytest` through `/opt/anaconda3/bin` first on
PATH; the snapshot runs used that interpreter explicitly and set PYTHONPATH to
the respective snapshot. No dependency installation or runtime fallback was needed.
Local pandas emitted two optional accelerator-version warnings (numexpr and
bottleneck); neither caused a test failure.

The historical Gate runs both actually used **Ubuntu, Python 3.12.15, pytest
9.1.1, coverage.py 7.16.2**, xdist `auto` on four CPUs, as their execution-context
and session logs show. This differs from the consumer's declared Python 3.14
input; job names alone do not prove runtime. New remote repair runs are recorded
in the PR body with their actual result, separately from historical measurements.

## Actual CI/Gate commands and retained runs

Inspected `.github/workflows/ci.yml`, `.github/workflows/pr-00-gate.yml`,
`pyproject.toml`, and `config/coverage-baseline.json` at current main. Both CI and
Gate call `stranske/Workflows/.github/workflows/reusable-10-ci-python.yml@main`.
CI is push-main/workflow_dispatch, coverage minimum **60**. Gate is PR/dispatch,
minimum **84**, soft reporting enabled. Both request Python **3.14**. Gate's
Integration Tests job runs `pytest tests/golden/ -v --tb=short` remotely.

The reusable workflow was read at Workflows
`c3f3fc5dca7cd4a86e11b8d56c70961d0219f5e6` (2026-11-03).
Its `Pytest (unit tests with coverage)` step constructs this invocation when,
as here, markers/extra args are empty and xdist is installed:

```sh
pytest -p ci_test_growth --junitxml=pytest-junit.xml --cov \
  --cov-config=pyproject.toml --cov-report=xml:coverage.xml \
  --cov-report=term-missing --cov-report=json:coverage.json \
  -n auto --dist=loadgroup
```

`ci_test_growth` is a temporary plugin written by the workflow; it is not a
repository-local executable. No invented local Gate command is used. The
coverage-min step is the workflow's inline Python XML check: read
`coverage.xml` root `line-rate`, multiply by 100, exit 1 if
`rate + 1e-6 < float(COVERAGE_MIN)`, otherwise print
`Coverage {rate:.2f}% meets minimum {target:.2f}%` and exit 0.
The repository's default pytest marker is `-m 'not live_llm'`.

Historical workflow revisions were respectively Workflows
`df108f233ea978e1788dc6b2b60e641e69fa5794` and
`d2def1901357855222046823b609a8f87b95ccfb`. Both retained logs show the same
pytest arguments and coverage-min algorithm above. The checkout logs, rather
than the run API's PR head SHA, bind these results to the squash-parent/squash
pair in the revision table.

| Original Gate run | Checkout | pytest output | pytest / coverage-min / Gate |
| --- | --- | --- | --- |
| [39472616859](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/39472616859) | `982430ef26973ea75b64da5991f78bad161c5c0c` | `1916 passed, 1 skipped, 953 warnings in 126.49s` | 0 / 0 / success |
| [39480415598](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/39480415598) | `543b53597014100c69d12635b428c85ac791aaeb` | `1921 passed, 1 skipped, 953 warnings in 161.29s` | 0 / 0 / success |

The historical step logs report `PYTEST_OUTCOME=success` and
`COVERAGE_MIN_OUTCOME=success`; no `Process completed with exit code` error is
present. Exit 0 above follows those successful steps. This is the passing CI
pytest evidence for the merged five tests, distinct from the local reruns.

Original coverage JSON/ZIP artifacts were not present in either run's artifact
listing on the check date. The measurements below were **recovered from original
Gate terminal coverage tables**, not presented as newly reconstructed CI runs.

## Comparable Gate coverage and shortfall

`[tool.coverage.run]` selects product code only (`pa_core`, `dashboard`,
`design-system`) and explicitly sets **`branch = false`**. The issue's dispatch
counts of 105/32/31/30 exactly match the baseline's **missing statements**;
calling them blended statements+branches was a selection-record error.
Historical branch exits were **not measured**, rather than measured as zero.

| Target, frozen issue order | Statements | Gate missing before | Gate missing after | Reduction | Historical missing branches before / after |
| --- | ---: | ---: | ---: | ---: | --- |
| `dashboard/pages/3_Scenario_Wizard.py` | 1052 | 105 | 101 | 4 | unmeasured / unmeasured |
| `pa_core/viz/pdf_report.py` | 54 | 32 | 0 | 32 | unmeasured / unmeasured |
| `pa_core/reporting/sweep_excel.py` | 89 | 31 | 0 | 31 | unmeasured / unmeasured |
| `pa_core/reporting/export_packet.py` | 220 | 30 | 6 | 24 | unmeasured / unmeasured |
| **Selected total** | **1415** | **198** | **107** | **91** | **unmeasured / unmeasured** |
| **Whole product** | **14775** | **1495** | **1392** | **103** | **unmeasured / unmeasured** |

Whole-product line coverage: **89.8816% → 90.5787% (+0.6971 percentage
points)**, reported by the XML minimum check as **89.88% → 90.58%**. Both exceed
84%. Twelve statements of whole-product improvement are outside the selected
four modules; they are incidental execution from the tests, not additional
selected-target delivery. The prior local document's **90.81% / +0.93** result
has no supporting comparable run and is discarded.

On the actual Gate/dispatch statement basis, the selected-target reduction is
**91**, leaving **89 short of the approximate 180 goal**. A true historical
blended Gate reduction cannot be derived because branch execution was not
recorded. It must not be substituted with a focused-suite percentage.

### Newly reconstructed focused statement+branch diagnostic

To supply actual branch measurements without a local full-suite run, clean
`git archive` snapshots of the exact historical baseline/head were tested with
the same focused related suites used in #2420, adding the new module only at
head. Both snapshots use the same local environment and coverage config;
`--cov-branch` is an explicit diagnostic override. These are **reconstructed
2026-11-03 focused runs**, not original Gate measurements and not a claim about
whole-suite branch coverage.

| Target | Missing statements before → after | Missing branch exits before → after | Combined before → after | Combined reduction |
| --- | ---: | ---: | ---: | ---: |
| `dashboard/pages/3_Scenario_Wizard.py` | 164 → 160 | 75 → 71 | 239 → 231 | 8 |
| `pa_core/viz/pdf_report.py` | 32 → 0 | 11 → 0 | 43 → 0 | 43 |
| `pa_core/reporting/sweep_excel.py` | 36 → 0 | 18 → 3 | 54 → 3 | 51 |
| `pa_core/reporting/export_packet.py` | 30 → 6 | 15 → 9 | 45 → 15 | 30 |
| **Total** | **262 → 166** | **119 → 83** | **381 → 249** | **132** |

The focused combined reduction is **132**, **48 short of 180** on that separate
basis. It cannot fill the missing historical Gate branch denominator. For
example, focused sweep baseline has 36 missing statements whereas Gate has 31;
focused Wizard head has 160 missing statements whereas Gate has 101. Those
suite-selection differences are why the two tables are kept separate.

Exact reconstructed invocations (cwd is the respective archive snapshot):

```sh
/opt/anaconda3/bin/python3 -m pytest -q tests/test_reporting_exports.py tests/test_export_packet.py tests/test_viz.py tests/test_wizard_config_chat_acceptance.py tests/test_wizard_config_chat_flow.py tests/test_wizard_config_wiring.py tests/test_wizard_getattr_fix.py tests/test_wizard_helpers.py tests/test_wizard_regime_wiring.py tests/test_wizard_render_coverage.py tests/test_wizard_schema.py tests/test_wizard_session_state.py tests/test_wizard_sleeve_suggestion_apply.py tests/test_dashboard_advanced_settings.py tests/test_dashboard_dead_ux_1907.py tests/test_dashboard_session_handoff.py tests/test_config_chat_preview_apply_revert.py tests/test_config_patch_roundtrip.py --cov --cov-config=pyproject.toml --cov-branch --cov-report=json:/private/tmp/coverage-2418-repair/before-focused-branches.json
```

Before `982430ef26973ea75b64da5991f78bad161c5c0c` output: `174 passed, 26 warnings in 22.76s`; exit **0**.

```sh
/opt/anaconda3/bin/python3 -m pytest -q tests/test_reporting_exports.py tests/test_export_packet.py tests/test_viz.py tests/test_wizard_config_chat_acceptance.py tests/test_wizard_config_chat_flow.py tests/test_wizard_config_wiring.py tests/test_wizard_getattr_fix.py tests/test_wizard_helpers.py tests/test_wizard_regime_wiring.py tests/test_wizard_render_coverage.py tests/test_wizard_schema.py tests/test_wizard_session_state.py tests/test_wizard_sleeve_suggestion_apply.py tests/test_dashboard_advanced_settings.py tests/test_dashboard_dead_ux_1907.py tests/test_dashboard_session_handoff.py tests/test_config_chat_preview_apply_revert.py tests/test_config_patch_roundtrip.py tests/test_coverage_round_2418.py --cov --cov-config=pyproject.toml --cov-branch --cov-report=json:/private/tmp/coverage-2418-repair/after-focused-branches.json
```

After `543b53597014100c69d12635b428c85ac791aaeb` output: `179 passed, 26 warnings in 21.01s`; exit **0**.

### Meaningful gaps left by #2420

- **Wizard:** only four of the 105 Gate-missing statements were gained. The
  normalizer helper test does not drive most page callbacks or the upload →
  temporary CSV/YAML → CLI simulation → saved-scenario path (historical lines
  2284–2357), which dominates the remaining gap. Additional controlled UI/CLI
  input fixtures were not supplied by #2420. These paths are testable; there is
  no evidence they require a browser or are dead code. The focused branch
  diagnostic still misses 71 exits in this page.
- **Export packet:** Gate leaves six executable statements at historical lines
  68, 111, 214–215, 218–220 (the latter includes a comment, hence seven line
  numbers but six statements). These are table truncation for >10/>12 rows,
  existing-tornado detection and optional tornado failure in
  `create_export_packet`, not generic Excel validation errors. The new test
  exercises `_packet_render_figures` and stubs `create_export_packet` for its
  async check; the parallel logic in the real packet creator remains uncovered.
  These cases can be covered by future focused tests. The branch diagnostic
  leaves nine exits, including unexercised predicates whose lines are covered.
- **PDF:** Gate and focused diagnostic leave no statements or instrumented
  branch exits missing. Optional-import `pragma: no cover` is unchanged;
  100% instrumented coverage does not prove every optional dependency works.
- **Sweep Excel:** no missing Gate statements; the focused branch diagnostic
  still leaves three exits (`89→84`, `149→147`, `155→159`): worksheet column
  guard/iteration and async non-DataFrame/empty-summary paths. Line coverage
  alone cannot establish those branches are tested.

This repair records the accepted partial result. It does not add tests to fill
those gaps or redo/revert #2420.

## Ranking evidence and selection deviation

The retained [issue #2418 dispatch body](https://github.com/stranske/Portable-Alpha-Extension-Model/issues/2418)
is the frozen ordering source: **escaped-defect priority, then churn, then
uncovered mass**, excluding reserved/in-flight modules. It orders:

1. Wizard: 105 units, one escaped-defect signal, churn 1, “fix commit touched
   this file”.
2. PDF report: 32 units, uncovered mass.
3. Sweep Excel: 31 units, uncovered mass.
4. Export packet: 30 units, uncovered mass.

Local dispatch log `/Users/teacher/.codex/orchestrator/coverage_autopilot.log`,
line 1768, corroborates the round and total:

```text
2026-10-20T11:39:22-0500 [Portable-Alpha-Extension-Model#C] opened issue #2418 (lane C, 4 target(s), 198 units): https://github.com/stranske/Portable-Alpha-Extension-Model/issues/2418
```

The repo-specific rank cache
`/Users/teacher/.codex/orchestrator/lanes/Portable-Alpha-Extension-Model.targets.json`
was also inspected. SHA-256:
`e19c6de951f5cfa502775ca9c5ac23ba035f2be792c1a92dc415b13db63335d3`.
Its metadata is `gate-coverage-3.12-2`, branch `codex/issue-2421`, SHA `63a1c574`,
created `2026-10-20T19:15:46Z`, so it is a **later snapshot**, not the original
pre-dispatch ranker input. Its selected rows corroborate the tiers:

| Cached raw rank | Path | Escaped-defect signals | Churn |
| ---: | --- | ---: | ---: |
| 1 | `dashboard/pages/3_Scenario_Wizard.py` | 1 | 1 |
| 38 | `pa_core/viz/pdf_report.py` | 0 | 0 |
| 39 | `pa_core/reporting/sweep_excel.py` | 0 | 0 |
| 40 | `pa_core/reporting/export_packet.py` | 0 | 0 |

Original pre-dispatch raw ranking JSON and the underlying escaped-defect
incident were not recovered. The frozen issue and historical Gate tables
establish the selected targets/counts; the later cache is corroboration only.
Older repository `docs/evidence/issue-2336-*/ranking.json` artifacts belong to
other rounds and cannot establish this dispatch.

**Selection deviation:** #2420 touched all four named targets and no reserved
production modules, but delivered two PDF tests and substantial reporting
coverage while adding only a helper-normalization test for the highest-priority
Wizard. It did not exhaust Wizard first and did not reach ~180. The original
PR body acknowledged a UI-fixture shortfall; the recovered measurements now
quantify it. No evidence supports the prior repair document's “no deviation”
or its claim that the files were chosen solely by largest uncovered mass.

## Prior local repair branch disposition

Before editing, `git branch --list 'closer*/issue-2418'` found
`closer/issue-2418` with unique commit `aade09ad`. It was merged into the
provisioned `closer-r2/issue-2418` (merge `58429899`). Its only file was this
acceptance document. Kept: the evidence-only scope, useful mutation candidates,
and PDF/sweep/packet target descriptions that could be checked. Re-executed
all five candidates instead of accepting prose claims. Replaced/discarded:
unsupported local-as-CI assertions, invented pytest command, unlinked coverage
percentages, incomplete before/after counts, and the incorrect ranking/no-gap
claims. No valid tests or production changes were discarded.

## Fresh deliberate-break and revert transcripts

Each experiment below was executed on the current acceptance code in the
provisioned worktree, individually. The production file was restored in a
`finally` block and byte-compared with its original contents before the green
rerun. Tests were unchanged. All red results are named assertion failures,
not dependency/collection errors. This proves each nominated behavior is
asserted, not that every branch of each test has an independently proven oracle.

### 1. `test_pdf_report_export_and_fallback_branches`

Production target: `pa_core/viz/pdf_report.py`. Temporary replacement:

```python
# original
fh.write(payload.encode())
# deliberate defect
fh.write((payload + "_BROKEN").encode())
```

**RED**, exit **1**; command `pytest -q tests/test_coverage_round_2418.py::test_pdf_report_export_and_fallback_branches`:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
_________________ test_pdf_report_export_and_fallback_branches _________________

tmp_path = PosixPath('/Users/teacher/.codex/orchestrator/agent-runtime/codex/tmp/pytest-of-teacher/pytest-417/test_pdf_report_export_and_fal0')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x302c94380>

    def test_pdf_report_export_and_fallback_branches(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pa_core.viz import pdf_report

        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        fig = go.Figure()
        output = tmp_path / "report.pdf"

        pdf_report.save([], output)
>       assert output.read_text() == "[]"
E       AssertionError: assert '[]_BROKEN' == '[]'
E
E         - []
E         + []_BROKEN

tests/test_coverage_round_2418.py:27: AssertionError
=============================== warnings summary ===============================
../../../../../../opt/anaconda3/lib/python3.12/site-packages/pandas/core/computation/expressions.py:23
  /opt/anaconda3/lib/python3.12/site-packages/pandas/core/computation/expressions.py:23: UserWarning: Pandas requires version '2.10.2' or newer of 'numexpr' (version '2.8.7' currently installed).
    from pandas.core.computation.check import NUMEXPR_INSTALLED

../../../../../../opt/anaconda3/lib/python3.12/site-packages/pandas/core/arrays/masked.py:56
  /opt/anaconda3/lib/python3.12/site-packages/pandas/core/arrays/masked.py:56: UserWarning: Pandas requires version '1.4.2' or newer of 'bottleneck' (version '1.3.7' currently installed).
    from pandas.core import (

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2418.py::test_pdf_report_export_and_fallback_branches
1 failed, 2 warnings in 4.86s
```

**GREEN**, exit **0**; command `pytest -q tests/test_coverage_round_2418.py::test_pdf_report_export_and_fallback_branches`:

```text
.                                                                        [100%]
=============================== warnings summary ===============================
../../../../../../opt/anaconda3/lib/python3.12/site-packages/pandas/core/computation/expressions.py:23
  /opt/anaconda3/lib/python3.12/site-packages/pandas/core/computation/expressions.py:23: UserWarning: Pandas requires version '2.10.2' or newer of 'numexpr' (version '2.8.7' currently installed).
    from pandas.core.computation.check import NUMEXPR_INSTALLED

../../../../../../opt/anaconda3/lib/python3.12/site-packages/pandas/core/arrays/masked.py:56
  /opt/anaconda3/lib/python3.12/site-packages/pandas/core/arrays/masked.py:56: UserWarning: Pandas requires version '1.4.2' or newer of 'bottleneck' (version '1.3.7' currently installed).
    from pandas.core import (

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 2 warnings in 2.26s
```

### 2. `test_pdf_report_merges_and_async_uses_cache`

Production target: `pa_core/viz/pdf_report.py`. Temporary replacement:

```python
# original
        writer.write(fh)
# deliberate defect
        writer.write(fh)
        fh.write(b"BROKEN")
```

**RED**, exit **1**; command `pytest -q tests/test_coverage_round_2418.py::test_pdf_report_merges_and_async_uses_cache`:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
_________________ test_pdf_report_merges_and_async_uses_cache __________________

tmp_path = PosixPath('/Users/teacher/.codex/orchestrator/agent-runtime/codex/tmp/pytest-of-teacher/pytest-419/test_pdf_report_merges_and_asy0')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x306d1a420>

    def test_pdf_report_merges_and_async_uses_cache(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pa_core.viz import pdf_report

        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        output = tmp_path / "merged.pdf"
        appended: list[bytes] = []

        class Writer:
            def append(self, buffer: io.BytesIO) -> None:
                appended.append(buffer.read())

            def write(self, handle: Any) -> None:
                handle.write(b"merged")

        monkeypatch.setattr(pdf_report, "PdfWriterType", Writer)
        monkeypatch.setattr(pdf_report, "figure_to_pdf_bytes", lambda figure: b"pdf")
        pdf_report.save([go.Figure(), go.Figure()], output)
        assert appended == [b"pdf", b"pdf"]
>       assert output.read_bytes() == b"merged"
E       AssertionError: assert b'mergedBROKEN' == b'merged'
E
E         Use -v to get more diff

tests/test_coverage_round_2418.py:72: AssertionError
=============================== warnings summary ===============================
../../../../../../opt/anaconda3/lib/python3.12/site-packages/pandas/core/computation/expressions.py:23
  /opt/anaconda3/lib/python3.12/site-packages/pandas/core/computation/expressions.py:23: UserWarning: Pandas requires version '2.10.2' or newer of 'numexpr' (version '2.8.7' currently installed).
    from pandas.core.computation.check import NUMEXPR_INSTALLED

../../../../../../opt/anaconda3/lib/python3.12/site-packages/pandas/core/arrays/masked.py:56
  /opt/anaconda3/lib/python3.12/site-packages/pandas/core/arrays/masked.py:56: UserWarning: Pandas requires version '1.4.2' or newer of 'bottleneck' (version '1.3.7' currently installed).
    from pandas.core import (

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2418.py::test_pdf_report_merges_and_async_uses_cache
1 failed, 2 warnings in 2.31s
```

**GREEN**, exit **0**; command `pytest -q tests/test_coverage_round_2418.py::test_pdf_report_merges_and_async_uses_cache`:

```text
.                                                                        [100%]
=============================== warnings summary ===============================
../../../../../../opt/anaconda3/lib/python3.12/site-packages/pandas/core/computation/expressions.py:23
  /opt/anaconda3/lib/python3.12/site-packages/pandas/core/computation/expressions.py:23: UserWarning: Pandas requires version '2.10.2' or newer of 'numexpr' (version '2.8.7' currently installed).
    from pandas.core.computation.check import NUMEXPR_INSTALLED

../../../../../../opt/anaconda3/lib/python3.12/site-packages/pandas/core/arrays/masked.py:56
  /opt/anaconda3/lib/python3.12/site-packages/pandas/core/arrays/masked.py:56: UserWarning: Pandas requires version '1.4.2' or newer of 'bottleneck' (version '1.3.7' currently installed).
    from pandas.core import (

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 2 warnings in 1.96s
```

### 3. `test_sweep_excel_metadata_image_and_async_paths`

Production target: `pa_core/reporting/sweep_excel.py`. Temporary replacement:

```python
# original
return json.dumps(value, sort_keys=True)
# deliberate defect
return json.dumps(value, sort_keys=True) + "_BROKEN"
```

**RED**, exit **1**; command `pytest -q tests/test_coverage_round_2418.py::test_sweep_excel_metadata_image_and_async_paths`:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
_______________ test_sweep_excel_metadata_image_and_async_paths ________________

tmp_path = PosixPath('/Users/teacher/.codex/orchestrator/agent-runtime/codex/tmp/pytest-of-teacher/pytest-421/test_sweep_excel_metadata_imag0')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x175cbc290>

    def test_sweep_excel_metadata_image_and_async_paths(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pa_core.reporting import sweep_excel

        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.delenv("CI", raising=False)
        output = tmp_path / "sweep.xlsx"
        summary = pd.DataFrame({"Agent": ["Base"], "terminal_AnnReturn": [0.05]})
        results = [{"combination_id": 4, "summary": summary}]
        monkeypatch.setattr(sweep_excel.risk_return, "make", lambda frame: go.Figure())
        png = io.BytesIO()
        Image.new("RGB", (1, 1), color="white").save(png, format="PNG")
        monkeypatch.setattr(sweep_excel, "figure_to_png_bytes", lambda figure: png.getvalue())

        sweep_excel.export_sweep_results(results, str(output), metadata={"labels": ["a", "b"]})
        workbook = sweep_excel.openpyxl.load_workbook(output)
        assert {"Metadata", "Run4", "Summary", "SummaryStats"} <= set(workbook.sheetnames)
>       assert workbook["Metadata"]["B2"].value == '["a", "b"]'
E       assert '["a", "b"]_BROKEN' == '["a", "b"]'
E
E         - ["a", "b"]
E         + ["a", "b"]_BROKEN

tests/test_coverage_round_2418.py:125: AssertionError
=============================== warnings summary ===============================
../../../../../../opt/anaconda3/lib/python3.12/site-packages/pandas/core/computation/expressions.py:23
  /opt/anaconda3/lib/python3.12/site-packages/pandas/core/computation/expressions.py:23: UserWarning: Pandas requires version '2.10.2' or newer of 'numexpr' (version '2.8.7' currently installed).
    from pandas.core.computation.check import NUMEXPR_INSTALLED

../../../../../../opt/anaconda3/lib/python3.12/site-packages/pandas/core/arrays/masked.py:56
  /opt/anaconda3/lib/python3.12/site-packages/pandas/core/arrays/masked.py:56: UserWarning: Pandas requires version '1.4.2' or newer of 'bottleneck' (version '1.3.7' currently installed).
    from pandas.core import (

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2418.py::test_sweep_excel_metadata_image_and_async_paths
1 failed, 2 warnings in 2.76s
```

**GREEN**, exit **0**; command `pytest -q tests/test_coverage_round_2418.py::test_sweep_excel_metadata_image_and_async_paths`:

```text
.                                                                        [100%]
=============================== warnings summary ===============================
../../../../../../opt/anaconda3/lib/python3.12/site-packages/pandas/core/computation/expressions.py:23
  /opt/anaconda3/lib/python3.12/site-packages/pandas/core/computation/expressions.py:23: UserWarning: Pandas requires version '2.10.2' or newer of 'numexpr' (version '2.8.7' currently installed).
    from pandas.core.computation.check import NUMEXPR_INSTALLED

../../../../../../opt/anaconda3/lib/python3.12/site-packages/pandas/core/arrays/masked.py:56
  /opt/anaconda3/lib/python3.12/site-packages/pandas/core/arrays/masked.py:56: UserWarning: Pandas requires version '1.4.2' or newer of 'bottleneck' (version '1.3.7' currently installed).
    from pandas.core import (

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 2 warnings in 2.42s
```

### 4. `test_export_packet_render_figure_branches_and_async`

Production target: `pa_core/reporting/export_packet.py`. Temporary replacement:

```python
# original
    return render_figs
# deliberate defect
    return []  # deliberate defect: discard prepared figures
```

**RED**, exit **1**; command `pytest -q tests/test_coverage_round_2418.py::test_export_packet_render_figure_branches_and_async`:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
_____________ test_export_packet_render_figure_branches_and_async ______________

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x11fa07a70>

    def test_export_packet_render_figure_branches_and_async(monkeypatch: pytest.MonkeyPatch) -> None:
        from pa_core.reporting import export_packet

        sensitivity = pd.DataFrame(
            {
                "Parameter": ["mu_H"],
                "Base": [0.05],
                "Minus": [0.04],
                "Plus": [0.06],
                "Low": [-0.01],
                "High": [0.01],
                "DeltaAbs": [0.01],
            }
        )
        sensitivity.attrs["metric"] = "terminal_AnnReturn"
        sensitivity.attrs["units"] = "%"
        sensitivity.attrs["tickformat"] = ".2%"
        rendered = export_packet._packet_render_figures(
            figs=[], summary_df=pd.DataFrame(), inputs_dict={"_sensitivity_df": sensitivity}
        )
>       assert len(rendered) >= 1
E       assert 0 >= 1
E        +  where 0 = len([])

tests/test_coverage_round_2418.py:227: AssertionError
=============================== warnings summary ===============================
../../../../../../opt/anaconda3/lib/python3.12/site-packages/pandas/core/computation/expressions.py:23
  /opt/anaconda3/lib/python3.12/site-packages/pandas/core/computation/expressions.py:23: UserWarning: Pandas requires version '2.10.2' or newer of 'numexpr' (version '2.8.7' currently installed).
    from pandas.core.computation.check import NUMEXPR_INSTALLED

../../../../../../opt/anaconda3/lib/python3.12/site-packages/pandas/core/arrays/masked.py:56
  /opt/anaconda3/lib/python3.12/site-packages/pandas/core/arrays/masked.py:56: UserWarning: Pandas requires version '1.4.2' or newer of 'bottleneck' (version '1.3.7' currently installed).
    from pandas.core import (

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2418.py::test_export_packet_render_figure_branches_and_async
1 failed, 2 warnings in 2.21s
```

**GREEN**, exit **0**; command `pytest -q tests/test_coverage_round_2418.py::test_export_packet_render_figure_branches_and_async`:

```text
.                                                                        [100%]
=============================== warnings summary ===============================
../../../../../../opt/anaconda3/lib/python3.12/site-packages/pandas/core/computation/expressions.py:23
  /opt/anaconda3/lib/python3.12/site-packages/pandas/core/computation/expressions.py:23: UserWarning: Pandas requires version '2.10.2' or newer of 'numexpr' (version '2.8.7' currently installed).
    from pandas.core.computation.check import NUMEXPR_INSTALLED

../../../../../../opt/anaconda3/lib/python3.12/site-packages/pandas/core/arrays/masked.py:56
  /opt/anaconda3/lib/python3.12/site-packages/pandas/core/arrays/masked.py:56: UserWarning: Pandas requires version '1.4.2' or newer of 'bottleneck' (version '1.3.7' currently installed).
    from pandas.core import (

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 2 warnings in 2.04s
```

### 5. `test_wizard_value_normalizers_cover_mapping_tuple_and_unique_values`

Production target: `dashboard/pages/3_Scenario_Wizard.py`. Temporary replacement:

```python
# original
    if isinstance(value, tuple):
        return [_normalize_config_chat_value(child) for child in value]
# deliberate defect
    if isinstance(value, tuple):
        return value  # deliberate defect: omit tuple normalization
```

**RED**, exit **1**; command `pytest -q tests/test_coverage_round_2418.py::test_wizard_value_normalizers_cover_mapping_tuple_and_unique_values`:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
_____ test_wizard_value_normalizers_cover_mapping_tuple_and_unique_values ______

    def test_wizard_value_normalizers_cover_mapping_tuple_and_unique_values() -> None:
        from pa_core.wizard_schema import AnalysisMode, RiskMetric

        page = Path(__file__).resolve().parents[1] / "dashboard" / "pages" / "3_Scenario_Wizard.py"
        wizard = runpy.run_path(str(page))
        normalize = wizard["_normalize_config_chat_value"]
        unique = wizard["_stable_unique_text_items"]

>       assert normalize({"items": ("one", "two")}) == {"items": ["one", "two"]}
E       AssertionError: assert {'items': ('one', 'two')} == {'items': ['one', 'two']}
E
E         Differing items:
E         {'items': ('one', 'two')} != {'items': ['one', 'two']}
E         Use -v to get more diff

tests/test_coverage_round_2418.py:282: AssertionError
----------------------------- Captured stderr call -----------------------------
2026-11-03 09:44:16.940 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager
------------------------------ Captured log call -------------------------------
WARNING  streamlit.runtime.caching.cache_data_api:cache_data_api.py:372 No runtime found, using MemoryCacheStorageManager
=============================== warnings summary ===============================
../../../../../../opt/anaconda3/lib/python3.12/site-packages/pandas/core/computation/expressions.py:23
  /opt/anaconda3/lib/python3.12/site-packages/pandas/core/computation/expressions.py:23: UserWarning: Pandas requires version '2.10.2' or newer of 'numexpr' (version '2.8.7' currently installed).
    from pandas.core.computation.check import NUMEXPR_INSTALLED

../../../../../../opt/anaconda3/lib/python3.12/site-packages/pandas/core/arrays/masked.py:56
  /opt/anaconda3/lib/python3.12/site-packages/pandas/core/arrays/masked.py:56: UserWarning: Pandas requires version '1.4.2' or newer of 'bottleneck' (version '1.3.7' currently installed).
    from pandas.core import (

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2418.py::test_wizard_value_normalizers_cover_mapping_tuple_and_unique_values
1 failed, 2 warnings in 2.28s
```

**GREEN**, exit **0**; command `pytest -q tests/test_coverage_round_2418.py::test_wizard_value_normalizers_cover_mapping_tuple_and_unique_values`:

```text
.                                                                        [100%]
=============================== warnings summary ===============================
../../../../../../opt/anaconda3/lib/python3.12/site-packages/pandas/core/computation/expressions.py:23
  /opt/anaconda3/lib/python3.12/site-packages/pandas/core/computation/expressions.py:23: UserWarning: Pandas requires version '2.10.2' or newer of 'numexpr' (version '2.8.7' currently installed).
    from pandas.core.computation.check import NUMEXPR_INSTALLED

../../../../../../opt/anaconda3/lib/python3.12/site-packages/pandas/core/arrays/masked.py:56
  /opt/anaconda3/lib/python3.12/site-packages/pandas/core/arrays/masked.py:56: UserWarning: Pandas requires version '1.4.2' or newer of 'bottleneck' (version '1.3.7' currently installed).
    from pandas.core import (

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 2 warnings in 2.13s
```

## Final unchanged-module rerun

```text
$ /opt/anaconda3/bin/python3 -m pytest -q tests/test_coverage_round_2418.py
.....                                                                    [100%]
=============================== warnings summary ===============================
../../../../../../opt/anaconda3/lib/python3.12/site-packages/pandas/core/computation/expressions.py:23
  /opt/anaconda3/lib/python3.12/site-packages/pandas/core/computation/expressions.py:23: UserWarning: Pandas requires version '2.10.2' or newer of 'numexpr' (version '2.8.7' currently installed).
    from pandas.core.computation.check import NUMEXPR_INSTALLED

../../../../../../opt/anaconda3/lib/python3.12/site-packages/pandas/core/arrays/masked.py:56
  /opt/anaconda3/lib/python3.12/site-packages/pandas/core/arrays/masked.py:56: UserWarning: Pandas requires version '1.4.2' or newer of 'bottleneck' (version '1.3.7' currently installed).
    from pandas.core import (

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
5 passed, 2 warnings in 3.04s
exit=0
```

`git diff origin/main -- tests/test_coverage_round_2418.py pa_core dashboard
pyproject.toml` is empty after restoration. Only this evidence document is
changed in the repair PR. Configured CI/Gate are executed remotely, with the
run links and results recorded in the PR body; a local focused run is never
called a passing CI run.
