# Sweep packet export failure boundaries

Related to #2336. Production code is unchanged at `18ac626ed7af301083edca2074b009a5ac3544cd`. Six new named CLI cases exercise runtime, ImportError, ModuleNotFoundError, ValueError, TypeError and KeyError export failures. They assert one export attempt, correct diagnostic/remediation, no success announcement, and a preserved sweep workbook and manifest. Export/rendering itself is stubbed; no real PowerPoint, deployment or physical export claim.

Last500 source-touching commits rank CLI first:21 repair-subject proxy matches,133 touches,297 uncovered statements. These are subject proxies, not verified escaped incidents. Complete ranking/raw history is in the archive.

## Matched full-suite evidence

The same isolated Python3.12.2 interpreter/environment ran `python -m pytest --cov=pa_core --cov=dashboard --cov-report=json:<capture> tests/ --junitxml=<capture>`. Baseline1643PASS/1SKIP/2default-deselected; candidate1649PASS with the same skip/deselections. Exactly six new nodes and zero changed old outcomes. Both commands exit0. Coverage12149→12159/14554 statements (83.47533324172049→83.54404287481105%), identical186source files/181excluded lines, no covered-line regression. CLI810→820/1107; new lines[1487, 1488, 1489, 1490, 1491, 1492, 1493, 1494, 1495, 1496]. No floor, test selection, source scope or exclusion changed. The NumPy/pandas/xarray overlay and complete raw warnings are retained; this is not hosted-environment parity.

## Reproduce

With Python3.12+ and repository test dependencies:

```sh
python3 -m pytest tests/test_cli_sweep_export_failures.py -q -o addopts=
python3 docs/evidence/issue-2336-sweep-export-failures/replay_sweep_export_failures.py --output /tmp/new-sweep-export-proof
python3 -m pytest --cov=pa_core --cov=dashboard --cov-report=json:coverage.json tests/
```

Replay extracts an exact committed Git snapshot, overlays the named test into a private temporary tree, narrows one of three sweep exception handlers to FloatingPointError, and executes all six named cases separately. Every real mutation execution returnsRED1; byte-identical restored source returnsGREEN0 with exact one-node JUnit validation. Three distinct mutant shapes, six named RED/GREEN executions. Caller source/test hashes are unchanged and private trees are cleaned. Reusing an output directory fails before mutation and preserves existing evidence hashes.

`validation.tar.gz` retains complete full/focused/mutation logs, JUnit, coverage, argv/cwd/process exits, source/test/replay snapshots, ranking, and reuse proof. `manifest.json` binds every archive member and current input. Full Black482files, focused Ruff and diff checks pass. Broader source2336 remainsOPEN below90%; no providerPASS or merge authorization. Matchingkeeper owns hosted checks/reviews; closer requires full exact-head topology, zero active threads, seven-minute review floor, guarded merge and actualcompare/bounded disposition.
