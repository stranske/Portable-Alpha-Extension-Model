# Issue 2336: `dashboard/browser_exports` deliberate-break evidence

Target: offline stlite/browser PNG and async export helpers (`dashboard/browser_exports.py`).

## Measured coverage before this round

Full suite (`pytest tests/ -m "not slow" --cov=pa_core --cov=dashboard`): **83.07%** line coverage
(measured 2026-10-06 in experiment worktree). Below the 90% autopilot close threshold, so work
proceeded.

Gap ranking (escaped-defect priority, then churn, then uncovered mass among non-wizard modules):

| Module | Cover | Missing stmts | Recent churn |
|--------|-------|---------------|--------------|
| `dashboard/browser_exports.py` | 15% | 50 | low |
| `pa_core/viz/export_backend.py` | 75% | 28 | low |
| `pa_core/llm/config_patch_chain.py` | 73% | 55 | low |

`browser_exports` was chosen: it wires the browser export path used by Results/Scenario Grid pages,
had almost no tests despite depending on `export_backend`, and is small enough for a test-only PR.

## Deliberate break (FAIL)

Mutation: force the server-runtime branch to pass `b"broken"` into `st.download_button` instead of
`figure_to_png_bytes(fig, **opts)`. This validates that the test suite correctly detects code changes.

```bash
pytest tests/test_browser_exports.py::test_render_png_download_server_uses_kaleido_path -q
```

Observed:

```
FAILED tests/test_browser_exports.py::test_render_png_download_server_uses_kaleido_path
AssertionError: ... 'data': b'broken' ... != ... 'data': b'png-bytes' ...
```

✅ **Break validation confirmed**: Test suite correctly fails when code is deliberately broken.

## Revert (PASS)

Restored `dashboard/browser_exports.py` and re-ran:

```bash
pytest tests/test_browser_exports.py -q
```

✅ **Revert validation confirmed**: All 11 tests pass, achieving 100% coverage on target module.

### Coverage impact validation

Command:
```bash
pytest tests/test_browser_exports.py --cov=dashboard.browser_exports --cov-report=term-missing -q
```

Observed: `dashboard/browser_exports.py` **100%** (59 statements).

Validation results:
- ✅ Target module: 15% → 100% coverage achieved
- ✅ All 59 statements in `browser_exports.py` now covered by 11 comprehensive tests
- ✅ Both server and browser runtime branches exercised
- ✅ Async task lifecycle fully tested (pending → done → success/failure paths)
- ✅ Error handling and user feedback paths validated
- ✅ Test-only PR: zero production code changes, minimal blast radius
