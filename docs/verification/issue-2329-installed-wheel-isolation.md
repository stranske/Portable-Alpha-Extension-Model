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
AssertionError: installed-wheel probe failed (1)
stdout:

stderr:
Traceback (most recent call last):
  File "<string>", line 4, in <module>
ModuleNotFoundError: No module named 'dashboard'

FAILED tests/test_dashboard_sample_data.py::test_wheel_install_exposes_bundled_dashboard_samples
1 failed in 32.36s
```

Isolated installation restored, metadata decoy retained:

```text
.
1 passed in 52.13s

```

The earlier diagnostic assertion still prints real stdout/stderr and fails on a nonzero probe. No Gate failure or original provider CONCERNS is relabeled PASS; fresh-head CI and verifier disposition remain required.


## Post-merge verifier follow-through (2026-10-04)

The original negative-control log was empty; it was not recoverable evidence.
A new control ran the actual production test and its real wheel build at merged
main `7e7fbba8485ef655895029cf4786f1144bcc05d9`, stripping only the wheel
installation invocation's `-I` at runtime. No production source was mutated.
The complete nonzero control above was captured from this new run, not reconstructed.
The restored unmodified production command produced:

```text
$ python -m pytest tests/test_dashboard_sample_data.py::test_wheel_install_exposes_bundled_dashboard_samples -q -o addopts=
.                                                                        [100%]
1 passed in 44.50s
```

The private environment was the same for both runs. Each run staged/builds its
own real wheel; the inherited matching wheel metadata decoy causes the unisolated
pip invocation to skip installation, and the isolated probe then cannot import
`dashboard`. Isolated installation restores the actual installed package.

The source merge's Python CI run37216777796 passed Python3.12, Ruff and Mypy,
but lint-format failed on `tests/conftest.py`. The exact full-repository command
`black --check --line-length 100 --exclude '(\.venv|\.workflows-lib|node_modules)' .`
reproduced the same single-file failure locally. Formatting that file is
AST-equivalent; the full check now passes472 files. The prior focused Black
claim did not establish the full-repository gate. The historical failed run and
provider CONCERNS remain unchanged. Fresh follow-up CI, review and compare must
finish before source2329 closes; Maint71 retains generated-delivery acceptance.
