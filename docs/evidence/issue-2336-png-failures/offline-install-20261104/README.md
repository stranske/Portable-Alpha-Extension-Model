# Offline installed-wheel verification for #2477

The new `test_offline_wheel_install_exposes_bundled_dashboard_samples` passes on
Python 3.14.9 with setuptools 84.2.1. Unlike the previous build-only probe, it
builds and installs a wheel and loads the installed index CSV, asset CSV and
portfolio YAML. Both product packages and all three resource paths must reside
inside the new venv. The shared probe still poisons inherited `PYTHONPATH` and
runs the installed interpreter with `-I`.

The offline check uses existing third-party dependencies through
`system_site_packages=True`; wheel building and installation use `--no-index`
and do not resolve dependencies. It complements the existing slow test, which
retains its fresh-dependency installation. This is evidence of installed product
behavior on a supported interpreter, not a clean-dependency integration run or a
recovery of PR #2475's missing wheel-build stderr. Production code and packaging
configuration remain byte-identical to main
`7613ffd97a450b15e550f499b7386ec4e67f2c51`.

[regressions.txt](regressions.txt) and [named-node JUnit](regressions.xml) retain
**57 passed, 1 skipped, 1 deselected**. The new installed-wheel node passes; the
SciPy optimizer success node skips because SciPy is absent, and the slow online
wheel test is deselected by the required `not slow` selection. Exact commands,
exits and timing are in [commands.json](commands.json). [verification.json](verification.json)
records the interpreter, tool versions, source hashes, exact build/install argv
and the built wheel's hash. The wheel itself is a temporary test artifact.

The shared probe now checks resource origins as well as product module origins,
and parses the installed index returns in addition to the asset and portfolio
samples. No existing test is removed or disabled. The required whole-tree Black
check and Ruff pass. Black's initial worker stall and the subsequent cache-primed
check are recorded separately; cache priming checks each file individually.

## Reconciliation and remaining blockers

The retained recheck manifest was independently verified before edits. Its raw
JSON and named outcomes still establish **13,493 / 14,626 covered statements
(92.25352112676056%)**, with **2,008 passed, 2 failed, 3 skipped, 4 deselected**.
These are the earlier completed coverage capture's counts, not this round's
focused counts. No new full-suite coverage measurement or PNG mutation execution
is claimed. The retained PNG archive remains pinned to main's CLI SHA256
`d979ab73274b096be0f5f0f90edf7644550aba09ac8b4d3b50b058780dba567e`.
PDF and next-boundaries evidence are untouched.

The installed-wheel check narrows the historical packaging investigation: the
current wheel's samples install and load successfully on supported Python.
PR #2475 used unsupported Python 3.12, but its exact wheel-build subprocess error
was lost. This passing check cannot recover that error. Its optimizer assertion
also lacks the diagnostic status needed to distinguish the optional SciPy/NumPy
import problem from an optimization failure. SciPy is absent here, and the prior
installation attempt could not obtain it. Neither historical cause is promoted
from plausible to established. A compatible SciPy integration run and historical
wheel-build reproduction remain outstanding; #2336 stays open.

Verified statuses for both repeated copies of the original PR checklist:

- [x] Bounded coverage capture with counts, per-module totals and largest gaps.
- [ ] Conclusive dispositions of all four historical failures and the PNG timeout.
- [x] Focused PNG/PDF/sweep run and eight PNG mutation pairs pinned to main's CLI.
- [ ] Publish evidence and close #2336 after completing historical dispositions.
- [x] Completed coverage JSON committed with percent, command and source revision.
- [x] README covered/total statements match coverage JSON exactly.
- [x] Eight assertion RED/restored GREEN pairs with the current main CLI hash.
- [x] Every failure in the completed full-suite capture is named with its cause.
- [x] Measured product coverage exceeds 90%.

Updating PR #2478's checkboxes was attempted after reviewing the three previous
implementation commits and validating the retained capture. Automatic approval
review rejected the update: `MCP tool call requires approval, but approval policy
is never`. Remote checkboxes remain unchanged. No issue-closure request was made.
The checkout's `.git` is read-only; the delivery commit uses temporary Git metadata
under `/tmp` and does not advance or publish this checkout's branch.
