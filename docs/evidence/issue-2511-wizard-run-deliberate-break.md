# Issue #2511 wizard Run persistence deliberate-break evidence

Reproduces the acceptance gate on branch `cursor/issue-2511` at
`64647c794af008b9841a7c7bbc55b0136e664059` (persistence fix) and later commits.

- Issue: #2511
- Named gate: `tests/test_wizard_run_journey_apptest.py::test_index_uploader_survives_rerun_after_run_click`
- Captured: `2026-11-16T03:39:00Z`

## Baseline (restored persistence)

```text
python3 -m pytest tests/test_wizard_run_journey_apptest.py::test_index_uploader_survives_rerun_after_run_click -q
.                                                                        [100%]
1 passed, 18 warnings in 2.11s

python3 -m pytest tests/test_wizard_run_journey_apptest.py -q
..                                                                       [100%]
2 passed, 21 warnings in 2.77s
```

## Deliberate break

Removed session-state persistence: the step-5 index gate was gated only on the
same-run `run_simulation` click (no `wizard_run_requested` flag), matching pre-fix
behavior.

```diff
         if run_simulation:
-            st.session_state[_WIZARD_RUN_REQUESTED_KEY] = True
             st.session_state.last_wizard_config = config
 
-        # Handle simulation execution (persist run request across reruns)
-        if st.session_state.get(_WIZARD_RUN_REQUESTED_KEY):
+        # Handle simulation execution
+        if run_simulation:
```

The named gate failed on the post-click rerun because `Index Data Required` was
no longer rendered:

```text
python3 -m pytest tests/test_wizard_run_journey_apptest.py::test_index_uploader_survives_rerun_after_run_click -q
F                                                                        [100%]
________________ test_index_uploader_survives_rerun_after_run_click ________________
>       assert any("Index Data Required" in text for text in _subheader_text(app))
E       assert False
FAILED tests/test_wizard_run_journey_apptest.py::test_index_uploader_survives_rerun_after_run_click
1 failed, 16 warnings in 2.02s
```

## Restoration

Persistence logic was restored (`wizard_run_requested` set on Run, uploader
gated on the flag, cleared after run or when leaving step 5). The named gate and
full file passed again (see Baseline).

Cleanup proof:

```text
git diff --exit-code HEAD -- dashboard/pages/3_Scenario_Wizard.py
```

The cleanup command exited `0` with no output, proving the temporary production
change was fully removed before this evidence file was added.