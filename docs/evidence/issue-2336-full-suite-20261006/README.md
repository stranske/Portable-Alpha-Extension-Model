# Full-suite acceptance recovery for #2336 / merged #2338

Both revisions completed the literal `python -m pytest --cov=pa_core --cov=dashboard --cov-report=json:<artifact> tests/` command. JUnit output and a 120-second diagnostic fault-handler timeout were added; repository default marker exclusions were retained. No application, test, coverage configuration, floor or exclusion was changed for these runs.

| Revision | Result | Covered / statements | Total coverage |
| --- | --- | --- | --- |
| Original parent `7e7fbba8485ef655895029cf4786f1144bcc05d9` | 1571 passed, 1 skipped, 2 deselected; 180.24s | 12127 / 14551 | 83.34135110988936% |
| Merged current main `b25c491b1b2e94d1ad2480949e6bbdedcc25a0a7` | 1594 passed, 1 skipped, 2 deselected; 172.38s | 12133 / 14553 | 83.37112622826909% |

Both reports contain the same 186 source files and 181 excluded lines. `pa_core/config.py` rises from 590/635 to 596/637. The 23 additional passing tests are the bounded metadata-shape and mapping-compatibility cases already covered by the retained actual mutation receipts. Full logs, unabridged coverage JSON and process exit receipts are adjacent, bound by SHA-256 in `manifest.json`.

The initial shared-environment run failed during collection because optional xarray 2023.6 referenced removed NumPy `unicode_`. A round-local virtual environment inherited the same installed packages and overlaid xarray 2026.9; both revisions used that exact environment. Plotly was 6.8, and the optional xarray package was the incompatibility. No global package was changed. The historical PNG-export stall is not reproduced in these completed runs.

This recovers the outstanding full-scope reproduction claim. It does not establish 90% initiative coverage: #2336 remains open, and the reviewed opener owns the next ranked bounded coverage chunk. Original provider CONCERNS and historical receipts remain intact. This evidence-only follow-up does not assert a deployed change.
