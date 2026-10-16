# Single-run PowerPoint export failure evidence

This test-only chunk of #2336 exercises the real CLI, simulation, persisted Excel workbook and final manifest. Only the visualization constructors and optional PowerPoint save boundary are replaced. Eight cases cover missing renderer, missing generic module, missing python-pptx, filesystem and permission errors, invalid value/type, and renderer runtime failure. They check the export arguments, exact operator guidance, one ERROR diagnostic, final manifest warning, and unchanged saved workbook bytes.

Production `pa_core/cli.py` is unchanged from base `0e98dcaf6b43a6320ce5e3c74d764d182701d04a`, SHA256 `95f0e38b1e2491ff23fe4f278c2d5c5d29a6293bfbf5ef14b64795a4c954906e`. Existing tests, historical proof, coverage scope and exclusions are unchanged.

## Results

| Measurement | Base | Candidate |
| --- | ---: | ---: |
| Full-suite passed | 1685 | 1693 |
| Full-suite failed | 1 | 1 |
| Skipped / default-deselected | 1 / 2 | 1 / 2 |
| Covered statements / same total | 12241 / 14554 | 12274 / 14554 |
| Statement coverage | 84.10746% | 84.33420% |

All prior JUnit node identities and outcomes match, with exactly eight new passing nodes. Across the same 186 source files and 181 excluded lines, all 33 newly covered statements are in `pa_core/cli.py`; none lose coverage. Both full commands exit 1 for the same pre-existing `tests/test_sleeve_suggestor.py::test_suggest_sleeve_sizes_optimize_prefers_best` local SciPy/NumPy import failure. Neither full run is reported as passing. The related CLI export suite passes all 31 cases. Black checks 495 Python files; focused Ruff and diff checks pass.

The current last-500-source-commit ranking selects CLI first: 27 repair-subject proxy matches, 133 touches and 244 baseline uncovered statements. Repair subjects are proxies, not verified escaped incidents. Full ranking and raw history are in the archive.

## Reproduce the causal proof

Use Python 3.12 or newer with the repository's test dependencies installed. The confirmed target has a non-PATH, application-bundled Python interpreter (see `docs/TARGET_WORK_ENVIRONMENT.md`). Obtain its absolute executable path from that application's runtime configuration or dependency information, then set `PA_PYTHON` below; do not assume a bare `python3` command resolves to it. Verify the selected executable's version and location before running the tests.

In this Codex app task, the verified executable was the owner-default `/opt/anaconda3/bin/python3` in a linked worktree with no selected virtual environment. The canonical checkout's old virtual environment lacks `coverage`; it was not altered. Use that verified path for this host, or replace it with the confirmed bundled interpreter on another host.

```sh
PA_PYTHON="/absolute/path/from-your-application-runtime-configuration"
"$PA_PYTHON" -c 'import sys; print(sys.executable); print(sys.version)'
"$PA_PYTHON" -m pytest tests/test_cli_pptx_failures.py tests/test_cli_png_failures.py tests/test_cli_pdf_failures.py tests/test_cli_sweep_png_export.py -q
"$PA_PYTHON" docs/evidence/issue-2336-pptx-failures/replay.py --output /tmp/pa-pptx-proof-new
```

Choose a fresh output path. Replay creates a private snapshot of the pinned base, overlays the current test, and strips Git repository-location environment variables before archiving. Each exact parametrized node fails by assertion (exit 1, no errors/skips) after its production ERROR log is changed to DEBUG, then passes (exit 0) after byte-identical restoration. Eight pairs cover six distinct logging anchors. Original, mutant, test and driver bytes, exact argv/cwd, raw console, named-module JUnit and phase/source digests are retained. The caller stays unchanged. Reusing an output path fails before mutation and preserves all 44 proof files. Five malformed JUnit controls are rejected.

## Raw evidence

`raw-evidence.tar.gz` SHA256: `8a859ae73a6283b920e99ce37de7899e6c04f69978325e3ff598e2209bf6f57d`.

Its `manifest.json` binds 66 members by path, byte length and SHA256. Every member and all three current source/test/driver bindings were checked. It contains the full baseline/candidate coverage JSON and JUnit, exact suite invocation records, focused/related runs, ranking, all 16 mutation/restoration phases, and evidence-safety controls. `summary.json` lists the exact added nodes and gained lines. Historical executions remain historical and were not rebound to this chunk.

The keepalive follow-up adds 20 CI regression cases in `tests/test_pptx_replay_junit_validation.py` for the proof validator and output-reuse refusal. They accept both assertion-report formats and a restored pass, reject wrong identities, missing/extra nodes, errors, skips, non-assertion failures, duplicate failures, inconsistent exits and truncated XML, and preserve existing output before Git archiving. All 63 focused PPTX/PNG proof and related CLI export tests pass locally with `-m "not slow"`. The archived full-suite counts and coverage above describe the original eight-test capture, before these 20 additional proof checks; they are not a new full-suite result. The original archive and its source/test/driver bindings are unchanged.

The optional PowerPoint export is fault-injected. No real browser renderer or Office execution, hosted CI success, provider PASS, branch coverage measurement, or 90% source completion is claimed. #2336 remains open for its broader goal. Matching keepalive owns hosted checks and review findings; closer owns exact-head delivery and actual comparison.
