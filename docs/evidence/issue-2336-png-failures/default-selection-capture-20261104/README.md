# Default-selection bounded main capture — 2026-11-04

**13,550 / 14,670 covered statements (92.3653715064758%).** These values equal
[coverage.json](coverage.json) exactly. The completed run reports **2,048 passed,
4 failed, 0 skipped, 3 deselected**, exit 1, in 192.86 pytest seconds (195.64
seconds including startup and reporting). Every one of the **2,052 selected nodes
finishes**; [completion.json](completion.json) retains their identities.

[summary.json](summary.json) records every module's coverage, missing line numbers,
and the largest source gaps. The raw console and named-node JUnit are
[full-suite.txt](full-suite.txt) and [full-suite.xml](full-suite.xml).

## Main revision, inputs and command

GitHub main is `06881dbdf0e915c305669e7fe3f8baa7d8a21ec4`. An isolated local clone
checks out that commit. Product files and test/coverage configuration match main.
The only test overlay is the existing 30-second native PNG subprocess bound,
preserved in [tests.patch](tests.patch).

The command, with absolute interpreter and artifact paths in
[command.json](command.json), is:

```sh
timeout --signal=INT --kill-after=10s 600s python -m pytest -p png_capture_completion --capture-completion-json=completion.json --cov=pa_core --cov=dashboard --cov-report=json:coverage.json --junitxml=full-suite.xml tests/
```

Repository default selection applies (`not live_llm` from `pyproject.toml`), including
the slow installed-wheel test. Three live-provider tests are deselected. The
interpreter is Python 3.12.2 (`/opt/anaconda3/bin/python3`); pytest 9.1.1,
pytest-cov 7.1.0, and coverage 7.16.2 match the recorded versions in
`command.json`. GNU `timeout` 9.12 wraps the suite on the capture host.

Production CLI SHA256 remains
`d979ab73274b096be0f5f0f90edf7644550aba09ac8b4d3b50b058780dba567e`. The eight
retained PNG assertion pairs at that hash were not repeated; see
[main-06881dbd-20261104/png-proof](../main-06881dbd-20261104/png-proof/results.json).

## Failure dispositions (this capture only)

| Test | Evidence-supported disposition |
| --- | --- |
| `tests/test_dashboard_sample_data.py::test_wheel_install_exposes_bundled_dashboard_samples` | Pip wheel build stderr reports `requires a different Python: 3.12.2 not in '>=3.13'` in the isolated build environment. Observed interpreter incompatibility in this capture; does not establish the historical online wheel failure cause. |
| `tests/test_dashboard_sample_data.py::test_offline_wheel_install_exposes_bundled_dashboard_samples` | Same pip stderr and Python floor mismatch as the online wheel test. Observed incompatibility in this capture; does not reconstruct discarded historical stderr. |
| `tests/test_entrypoints.py::test_console_scripts_work_in_clean_venv` | Wheel build for the clean-venv probe fails with the same `3.12.2 not in '>=3.13'` stderr before entrypoint execution is reached. Environmental for this interpreter choice; historical setuptools/network causes remain unresolved. |
| `tests/test_schema_export.py::test_parameter_dictionary_matches_schema` | Assertion reports parameter-dictionary drift against the committed golden file when generated under this interpreter. Environmental doc rendering mismatch for this run; does not establish historical failure causes. |

`tests/test_cli.py::test_main_with_png` **passed** within the retained 30-second
subprocess bound. That pass does not establish native renderer success in
restricted sandboxes documented elsewhere.

`tests/test_sleeve_suggestor.py::test_suggest_sleeve_sizes_optimize_prefers_best`
**passed** with real SciPy on this stack. That pass does not establish causes for
earlier optimizer failures whose diagnostics are unavailable.

Historical wheel-build and optimizer causes remain unresolved. GitHub reports #2336
already closed; this capture does not change issue state.
