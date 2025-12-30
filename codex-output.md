Moved backend-dependent imports in `pa_core/__main__.py` to after backend selection so the chosen backend is respected, and sped up the backend CLI tests by stubbing the sweep pipeline.  
- `pa_core/__main__.py` now imports simulation modules after `resolve_and_set_backend` to avoid binding NumPy before a CuPy selection.  
- `tests/test_backend_selection.py` and `tests/test_backend_cli_integration.py` clamp sweep ranges and add an autouse fixture that stubs `run_parameter_sweep` and `export_sweep_results` to keep backend flag tests fast.  
- `codex-prompt.md` checkboxes and progress are updated to reflect completed tasks and acceptance criteria.

**Tests**
- `python -m pytest tests/test_backend_selection_refactor.py tests/test_backend_selection.py tests/test_backend_cli_integration.py`

If you want to broaden coverage, a natural next step is:  
1) `python -m pytest`