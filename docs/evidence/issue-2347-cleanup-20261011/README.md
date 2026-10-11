# PR2347 cleanup recovery

Exact baseline747ae47e86a51289ab319762a39dcf778b03fe63 retained the two originating review findings4240287476/4240287477 after branch recovery. Both were outdated in GitHub but still valid in the current code. Classic branch protection requires conversation resolution; no guard bypass was used.

Six new cases interrupt the actual snapshot replay at its first RED, then inject concurrent cache removal, cache PermissionError, or restored-source read failure. Original production:6failed/6deselected(exit1). Repaired driver:22combined replay/missing-workbook isolation testsPASS(exit0). These are simulated phase interruptions testing the actual driver cleanup, not new CLI mutation or full-suite acceptance. Raw console/JUnit and the baseline driver are retained as gzip members with stored/decoded hashes.

Cleanup restores source before attempting cache cleanup, uses missing_ok, records cleanup errors and an unavailable restored hash asnull, and always attempts controls.json. An original TimeoutExpired/KeyboardInterrupt survives cleanup and receipt errors. With no original phase error, cleanup errors still raise. README now describes three interruption points/0,1,2completed phases and spaces its historical20test result correctly.

Historical manifests below issue-2336-cli-snapshots remain unchanged and revision-bound to their evaluated_base. Their current_inputs describe those historical captures; they are not current repair-input hashes. This manifest binds the current repair driver/tests separately. Original CLI and observability test bytes are unchanged. Source2336 broad coverage remainsOPEN; fresh exacthead CI/review/topology/seven-minute floor and actual postmerge verification remain required.
