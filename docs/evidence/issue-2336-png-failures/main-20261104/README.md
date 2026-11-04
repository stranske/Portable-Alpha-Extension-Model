# Current-main evidence for #2477 / #2336

Production revision: `7613ffd97a450b15e550f499b7386ec4e67f2c51` (GitHub main,
confirmed through the GitHub API on 2026-11-04). The work started at
`4344002a5a0d2a012d33f86ac3ca3a18ea445b5a`; its only difference from main was
`agents/codex-2477.md`. Production directories and coverage configuration remain unchanged.

## Completed measurement and limits

**13,493 / 14,626 statements covered: 92.25352112676056%.**
`coverage.json` is the raw completed-run report; `summary.json` preserves its exact
totals, every module's coverage and missing line numbers, the 15 largest gaps,
and all failed nodes. `summarize.py` checks file totals against aggregate totals
and JUnit counts against testcase outcomes; interrupted captures are rejected.

The completed run has **2,001 passed, 2 failed, 3 skipped, 4 deselected**, exits 1,
and finishes in 219.08 seconds. Its console and named-node JUnit are
`completed-suite.log` and `completed-suite.xml`. `commands.json` records argv,
timeouts, Git identities, environment versions and input hashes; `manifest.json`
binds retained artifact bytes.

```sh
timeout --signal=INT --kill-after=30s 600s env COVERAGE_FILE=/tmp/issue-2336-completed.coverage python -m pytest --cov=pa_core --cov=dashboard --cov-report=json:docs/evidence/issue-2336-png-failures/main-20261104/coverage.json --junitxml=docs/evidence/issue-2336-png-failures/main-20261104/completed-suite.xml -m 'not slow and not live_llm' -o faulthandler_timeout=60 tests/
```

This measures current-main **production** with a documented test-harness repair:
`tests/test_cli.py::test_main_with_png` invokes the same native CLI in a child
process with a 30-second timeout, preserving the workbook and successful-exit
assertions. The timeout remains a failure, not a skip. Thirteen new summary
validator tests are also included. No product code or coverage policy changed.

The earlier `full-suite.log` is an incomplete run of the original tests, with a
600-second outer timeout and a 30-second kill grace. Its fault dump identifies
`tests/test_cli.py::test_main_with_png` waiting in Kaleido's synchronous server.
It supplies no completed-suite coverage claim. The keepalive requirement to use
`-m 'not slow'` excludes the wheel-install test; the existing live-provider
exclusions also remain in effect. A capture including those four tests is still
outstanding. This evidence is not an all-tests PASS or a native-export witness.

## Failure dispositions

| Test | Observed result and disposition |
| --- | --- |
| `tests/test_cli.py::test_main_with_png` | Original run hangs in Kaleido; repaired harness records `subprocess.TimeoutExpired` after 30 seconds. Environmental renderer startup failure: the separate `browser-probe.log` shows Chromium aborting with `setsockopt: Operation not permitted` even using a writable temporary profile. The unbounded test invocation was a real test-harness defect, fixed here. Rendering still requires verification in an environment where Chromium can start. |
| `tests/test_entrypoints.py::test_console_scripts_work_in_clean_venv` | Current run cannot resolve `pypi.org` while bootstrapping setuptools. Environmental network restriction; no installed-entrypoint behavior is reached. PR #2475 instead reached package installation under Python 3.12.2 and was rejected by the declared `>=3.13` requirement. |
| `tests/test_dashboard_sample_data.py::test_wheel_install_exposes_bundled_dashboard_samples` | One of PR #2475's four failures: its Python 3.12 wheel-build subprocess exited 1. That interpreter is unsupported; captured wheel stderr was not retained, so the exact wheel rejection is not independently proven. This slow test is deselected here by the required keepalive marker. Supported-Python wheel validation remains outstanding; no successful wheel result is claimed. |
| `tests/test_schema_export.py::test_parameter_dictionary_matches_schema` | PR #2475 generated `Optional[Dict[...]]` where committed text uses `Dict[...] \| None`. Environmental Python-version representation difference in the unsupported Python 3.12 run. The unchanged dictionary regenerates byte-identically and this test passes on the supported Python 3.14 interpreter. |
| `tests/test_sleeve_suggestor.py::test_suggest_sleeve_sizes_optimize_prefers_best` | PR #2475 imports top-level SciPy but returns a grid fallback (`optimizer_success=False`). Its environment includes the old SciPy/NumPy warning and main already catches an incompatible `scipy.optimize` import. Dependency incompatibility is the likely cause, but that capture does not retain `optimizer_status`, so an optimizer failure cannot be ruled out. Here SciPy is absent and the existing `importorskip` skips the test. Existing incompatible-import and missing-SciPy fallback regressions both pass; a fresh run with compatible SciPy/NumPy is still required before treating the historical assertion as conclusively environmental. |

The other two current skips are the Anthropic constructor option matrix tests
(`test_audit_provider_options_reach_real_offline_constructor[anthropic]` in
`tests/test_dashboard_llm_settings.py` and
`test_create_llm_provider_options_keep_supported_fields_out_of_anthropic_payload`
in `tests/test_llm_provider_missing_keys.py`): `langchain_anthropic` is absent.

`pr2475-full-suite.log` is copied byte-for-byte from PR head
`d112c2238441e3f9035f42360f5cf1e51839f5ea`'s archive.
`pr2475-reconciliation.json` retains its archive/report hashes, four named failure
messages and actual totals: **13,537 / 14,670 (92.27675528289025%)**. The published
13,441 numerator is wrong by 96; the JSON percentage is consistent with 13,537.
`coverage-tool-comparison.json` records every module delta between that report
and this one. Both covered and total statements fall by 44 exclusively in
`pa_core/wizard_schema.py`; exclusions gain nine lines. The toolchains differ
(Python 3.12 / coverage 7.16.2 versus Python 3.14.9 / coverage 7.17.0), with the
same product source and file set. These totals are reported separately.

## PNG proof and validation

The focused command uses unchanged production code:

```sh
python -m pytest tests/test_cli_png_failures.py tests/test_cli_pdf_failures.py tests/test_cli_sweep_png_export.py -q -m 'not slow and not live_llm' --junitxml=docs/evidence/issue-2336-png-failures/main-20261104/focused.xml
python docs/evidence/issue-2336-png-failures/replay.py --output <fresh-nonexistent-directory>
```

The first command reports **23 passed**. `png-proof.tar.gz` preserves eight exact
PNG assertion RED/restored GREEN pairs: `renderer`, `module`, `io`, `permission`,
`value`, `type`, `runtime-renderer`, `runtime-other`. Each RED exits 1 with exactly
one named-node `AssertionError`; each GREEN exits 0 with exactly one PASS and no
errors/skips. All restorations are byte-identical to main's CLI SHA256
`d979ab73274b096be0f5f0f90edf7644550aba09ac8b4d3b50b058780dba567e`.
The archive includes mutant sources, console logs, JUnit, argv, private working
directories and input/mutant/restored hashes. The active checkout is unchanged.
Historical PDF, next-boundaries and PNG archives retain their original bindings.

The summary, strict JUnit validator, schema and optimizer fallback regression
command reports **34 passed, 1 skipped**. Targeted summary-helper coverage is
reported separately in `summary-targeted-coverage.log`; tooling coverage is not
part of the product percentage. Black's required repository-wide check passes
for 534 Python files. Sandbox multiprocessing and threadpool behavior required
priming a temporary Black cache using individual-file checks, then running the
unchanged repository-wide check with one worker and that cache.

## Verified progress and remaining acceptance

- [x] Capture completed routine-suite coverage JSON, exact command, Git identity,
  counts, every module's coverage and largest gaps, with the harness delta explicit.
- [x] Reconcile the historical 13,441 versus 13,537 mismatch from archived JSON.
- [x] Re-run the 23 focused cases and eight PNG mutation RED/GREEN pairs on current
  main production; preserve logs, JUnit, argv and hashes.
- [x] Exceed 90% product coverage in the completed measured run.
- [ ] Commit the changes. `git add` cannot create `.git/index.lock` because the
  workspace mounts `.git` read-only; approval escalation is unavailable in this run.
  The GitHub attempts to add `needs-human` and a blocker comment were also rejected:
  those actions require approval and the approval policy is `never`. PR #2478 was
  verified open with `isDraft=false` after those attempts.
- [ ] Complete the slow wheel test and a native renderer run in a suitable environment.
- [ ] Re-run the historical optimizer failure with a compatible SciPy/NumPy pair;
  fix a product bug if that run demonstrates one.
- [ ] Publish durable evidence and close #2336 only after those dispositions are
  conclusive. The issue is not closed by this change.

Coverage exceeds 90%, so the conditional below-target module addition is not
needed. If further product coverage work is scheduled, the largest remaining gap
is `pa_core/cli.py` (165 missing statements), followed by
`pa_core/sleeve_suggestor.py` (115); existing tests for the latter are
`tests/test_sleeve_suggestor.py`.
