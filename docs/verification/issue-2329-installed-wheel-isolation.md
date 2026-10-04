# Installed-wheel acceptance follow-through for #2329

Source main: `f40f635546674e479cb43c50bb6399f59aa5dbe5`. Merged repair #2331 has provider CONCERNS in report5962524752. Direct current-code inspection shows the wheel install and probe already use `cwd=tmp_path`; the original working-directory allegation is obsolete. Inherited `PYTHONPATH` could still import source modules, so this follow-up runs the real installed-wheel probe with isolated Python (`-I`), deliberately poisons PYTHONPATH with a dashboard package that raises, and asserts dashboard/pa_core modules resolve under the installed interpreter prefix. It retains all index, asset, portfolio loading and console-script assertions.

Validation used a private project environment, because shared Anaconda has an incompatible xarray/NumPy combination. No shared environment was changed.

- Named actual wheel install/sample test:1passed in71.38s; durable fixture repetition1passed in52.29s.
- Negative control reuses that actual installed wheel, exact probe body, outside-source cwd and poisoned PYTHONPATH, but removes only `-I`: exit1 with `RuntimeError: installed wheel probe imported PYTHONPATH decoy`.
- Restore `-I` using that same installed wheel/probe: exit0, `ok`. These are subprocess controls against a real non-editable install; the full named test is independently green.
- `python -m pytest -q --no-cov tests/test_entrypoints.py -n 2`:7passed in26.25s; repeat7passed in11.31s. The first actual wheel build and clean-venv entrypoint build ran concurrently in separate staged source trees; no shared build-artifact failure reproduced. The synthetic copy-control test is retained but is not the sole build proof.
- Ruff/Black and `git diff --check` pass.

Commands were invoked with closer `work/20261004T1444Z/pae-test-env/bin/python`. Full logs and installed-interpreter/control identity are retained in that round's `pae-wheel-*.log`, `pae-parallel-entrypoints-*.log`, and `pae-wheel-control.json`.

Acceptance remains open until this follow-up's exact-head CI/reviews, squash merge, verify:compare report and original2331 check-floor/review disposition are complete. No generated sync PR is edited; maintenance rechecks remain Maint71-owned. No hosted production acceptance or provider PASS is invented.


## Isolated pip installation recovery

Exact CI diagnostic run37213701485 at1b2222a reported `ModuleNotFoundError: No module named dashboard` after1569 other tests passed. A fresh venv pip invocation can discover matching distribution metadata through inherited PYTHONPATH, declare the wheel already installed and skip installation. The isolated import probe then exposes the missing package. Run wheel installation with `python -I -m pip` too, and reproduce the inherited metadata in the committed test using the real wheel METADATA. All installed imports/resources/assertions remain.

Same production test and wheel-build path, strip only `-I` from wheel installation at runtime (no source edit):

```text
FAILED tests/test_dashboard_sample_data.py::test_wheel_install_exposes_bundled_dashboard_samples
AssertionError: installed-wheel probe failed (1)
ModuleNotFoundError: No module named 'dashboard'
1 failed in 31.03s
```

Isolated installation restored, metadata decoy retained:

```text
.
1 passed in 52.13s

```

The earlier diagnostic assertion still prints real stdout/stderr and fails on a nonzero probe. No Gate failure or original provider CONCERNS is relabeled PASS; fresh-head CI and verifier disposition remain required.


## Current exact-head evidence recovery (2026-10-04T16:53Z)

Reproduced from merged repair head `9817a37e343b30eb8cf4ad5f4d8c2bc6efb9f871`, with the same committed production test and staged-wheel build. A runtime subprocess wrapper removed only `-I` from the wheel-install pip command; no production source or assertion changed. The inherited distribution-metadata decoy caused pip to skip the actual install, then the unchanged isolated probe failed with `ModuleNotFoundError: No module named dashboard` (1 failed in 31.03s). Restoring the exact subprocess command, retaining the decoy and all assertions, passed (1 passed in 46.74s). Full raw logs and runtime wrappers are retained in closer `work/20261004T1641Z/pip-{red,green}.log` and `pip-{red,green}.py`.

Production test SHA256: `ccf62e54c2075f04f7e4dae182512a9a762d0c18c4685b2fa627afdd3e98871b`.

Merge CI run37216777796 failed Black solely on an extra blank line in `tests/conftest.py` (1 file would be reformatted; 471 unchanged). This follow-up removes that line and fills the empty negative-control transcript. It does not claim provider PASS, full CI success, or source closure before exact-head checks and verifier disposition.
