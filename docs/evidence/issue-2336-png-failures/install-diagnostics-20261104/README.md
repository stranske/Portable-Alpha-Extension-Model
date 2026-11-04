# Installed-wheel failure diagnostics for #2477

The first unfinished task is dispositioning the historical optimizer and wheel
failures. The installed-wheel probe still discarded pip's captured stdout/stderr
when installation failed. It now uses the same reporting helper as wheel builds,
preserving command, exit status and both streams in pytest reports and JUnit.
The installation environment, isolated interpreter and pip arguments are preserved.
Production and packaging configuration remain unchanged.

Two parameterized regressions exercise online and offline installation failures.
Both fail before the repair with an unreported `CalledProcessError` in
[red.txt](red.txt); both pass afterward in [regressions.txt](regressions.txt).
The combined packaging, optimizer, schema, summary and export suite reports
**85 passed, 1 skipped, 1 deselected**; [regressions.xml](regressions.xml) retains
the named outcomes. The real supported-Python offline wheel installation passes.
The existing slow online wheel test remains deselected by the required marker.

## Previously delivered evidence and reconciled tasks

The five prior implementation commits were reviewed before editing. The
[verified capture](../verified-20261104/README.md) and all its manifest entries
were independently checked: **13,493 / 14,626 statements covered
(92.25352112676056%)**, with **2,018 passed, 2 failed, 3 skipped, 4 deselected**.
Its README totals equal its raw coverage JSON exactly. These are retained capture
figures, not a new full-suite execution for this diagnostic-only change.

All 41 members of the current-main PNG archive match its manifest. Each of eight
named nodes has assertion RED and restored GREEN JUnit outcomes, and input and
restored hashes match current source. The CLI SHA256 remains
`d979ab73274b096be0f5f0f90edf7644550aba09ac8b4d3b50b058780dba567e`,
identical to PR #2478's main base
`7613ffd97a450b15e550f499b7386ec4e67f2c51`. PDF and next-boundaries evidence
is unchanged. This is archive validation, not a fresh mutation execution.

Each status below applies to both copies of the PR checklist (14/18 verified):

- [x] Completed bounded coverage capture, counts, module totals and largest gaps.
- [ ] Conclusive dispositions of all historical failures and the PNG timeout.
- [x] Focused export run and eight PNG mutation pairs bound to current main.
- [ ] Publish final evidence and close #2336 after historical dispositions.
- [x] Committed completed-run coverage JSON with percent, command and Git SHA.
- [x] README covered/total statements equal coverage JSON exactly.
- [x] Eight assertion RED/restored GREEN pairs and unchanged current CLI hash.
- [x] Every current full-run failure is named with its cause in the retained capture.
- [x] Total product coverage exceeds 90%.

## Remaining limits and delivery

The optimizer test still skips because SciPy is absent. An attempted binary SciPy
installation found no available distribution; no compatible optimizer integration
result is claimed. PR #2475 did not retain its wheel-build stderr or optimizer
status. The supported-Python installed-wheel pass does not recover those historical
causes. Both historical dispositions and #2336 closure remain open.

Automatic approval review rejected the PR-body update, `needs-human` label and
blocker comment: `MCP tool call requires approval, but approval policy is never`.
The reconciled PR body is retained at `/tmp/issue2336-install-diagnostics-pr-body.md`.
The checkout's `.git` is read-only, so its branch cannot be advanced here.
`delivery.json` records the temporary delivery commit and patch outside that checkout.
The evidence manifest records artifact sizes and hashes; `checks.json` records
validation commands and the initial Black timeout separately from the final check.
