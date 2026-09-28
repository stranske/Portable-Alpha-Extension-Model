# PR #2317 sweep PNG deliberate-break evidence

This is a fresh reproduction on current `main` for follow-up issue #2322. It
does not claim to reconstruct console output from the historical PR #2317 run.

- Original issue: #2307
- Merged PR: #2317 (`7383f84355a1`)
- Reproduction base: `1bde952b11decc61a48aa36dc5cd0c28dcb5e607`
- Captured: `2026-09-28T20:08:23Z`

## Baseline

```text
uv run pytest tests/test_cli_sweep_png_export.py::test_sweep_mode_honors_png_flag -q -o addopts=
.                                                                        [100%]
1 passed, 2 warnings in 21.68s
```

## Deliberate break

Only the sweep-branch PNG guard in `pa_core/cli.py` was changed temporarily:

```diff
-            if flags.png:
+            if False and flags.png:  # deliberate-break: prove sweep PNG coverage
```

The named gate then failed because the requested sweep PNG was absent:

```text
uv run pytest tests/test_cli_sweep_png_export.py::test_sweep_mode_honors_png_flag -q -o addopts=
F                                                                        [100%]
_______________________ test_sweep_mode_honors_png_flag ________________________
>       assert out_file.with_suffix(".png").read_bytes() == b"png"
E       FileNotFoundError: [Errno 2] No such file or directory: '.../sweep.png'
FAILED tests/test_cli_sweep_png_export.py::test_sweep_mode_honors_png_flag
1 failed, 2 warnings in 1.90s
```

## Restoration

The exact production line was restored to `if flags.png:` and the same named
gate passed:

```text
uv run pytest tests/test_cli_sweep_png_export.py::test_sweep_mode_honors_png_flag -q -o addopts=
.                                                                        [100%]
1 passed, 2 warnings in 1.44s
```

Cleanup proof:

```text
git diff --exit-code HEAD -- pa_core/cli.py tests/test_cli_sweep_png_export.py
```

The cleanup command exited `0` with no output, proving the temporary production
change and any test-file changes were fully removed before this evidence file
was added.
