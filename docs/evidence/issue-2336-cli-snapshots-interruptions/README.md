# Snapshot replay interruption acceptance

PR2342 / source2336 / review thread PRRT_kwDOO15QxM6p5gQy.
Two new focused cases load the actual CLI snapshots driver, force TimeoutExpired
and KeyboardInterrupt at three points: the first RED, its restored GREEN, and
the next RED. They assert original source/test bytes plus partial controls with
original/restored hashes and respectively 0, 1, or 2 completed phases. The fixture copies the actual tracked CLI and test source.

Disabling only the driver's finally source restoration yields two named assertion
failures; exact source bytes are restored in finally by the outer proof driver.
The historical restored focused isolation/observability run passed 20 tests. Raw RED/GREEN
and driver SHA are retained alongside this file. No production CLI/driver bytes,
coverage scopes/floors or original proof artifacts are changed.

Initial execution hit the known system xarray2023/NumPy2 import incompatibility;
those errors are not mutation evidence. Corrected runs reuse the existing private
xarray2026.9 overlay, with global packages unchanged. These logs are corrected
assertion failures and restored passes. Prior source2336/PR2342 mutation and
coverage receipts remain bound to their original revisions. New hosted CI,
exact-head review disposition/topology and seven-minute floor are still required.
