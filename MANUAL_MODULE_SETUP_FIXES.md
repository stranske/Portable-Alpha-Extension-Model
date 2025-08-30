# Manual Module Setup Pattern Fix Documentation

## Issue Description

Multiple test files in the repository use a manual module setup pattern that bypasses the proper development environment setup described in the coding guidelines. These patterns use:

```python
import sys
import types
from pathlib import Path

# Manual setup pattern
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
PKG = types.ModuleType("pa_core")
PKG.__path__ = [str(root / "pa_core")]
sys.modules.setdefault("pa_core", PKG)
```

## Recommended Approach

Instead, use the recommended development environment setup:
- Run tests with `./dev.sh test`  
- Or ensure `PYTHONPATH=$PWD` is set when running pytest directly

## Files Fixed

### ✅ tests/test_create_launchers.py
**Before:**
```python
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from scripts.create_launchers import (
    make_mac_launcher,
    make_windows_launcher,
)  # noqa: E402
```

**After:**
```python
from scripts.create_launchers import (
    make_mac_launcher,
    make_windows_launcher,
)
```

**Verification:**
```bash
PYTHONPATH=$PWD python -m pytest tests/test_create_launchers.py -v
# PASSED - tests/test_create_launchers.py::test_make_launchers
```

## Files Pending Fix

The following files have the manual module setup pattern and need similar fixes:

### Files requiring pa_core imports (blocked by validators.py syntax error):
- tests/test_dashboard_asset_library.py
- tests/test_agents.py  
- tests/test_schema.py
- tests/test_data_calibration.py
- tests/test_pa_cli_validate.py
- tests/test_portfolio_aggregator.py
- tests/test_risk_metrics_agent.py
- tests/test_covariance_psd.py
- tests/test_validate_cli.py
- tests/test_nearest_psd.py
- tests/test_orchestrator.py
- tests/test_metrics.py
- tests/test_cli.py
- tests/golden/test_scenario_smoke.py

### Files with sys.path.insert patterns:
- tests/test_dashboard_pages.py (imports dashboard pages which import pa_core)

## Fix Template

For files that import from pa_core or scripts:

**Before:**
```python
import sys
import types
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
PKG = types.ModuleType("pa_core")
PKG.__path__ = [str(root / "pa_core")]
sys.modules.setdefault("pa_core", PKG)

from pa_core.some_module import some_function
```

**After:**
```python
from pa_core.some_module import some_function
```

## Testing Approach

1. Ensure virtual environment is activated: `source .venv/bin/activate`
2. Run with PYTHONPATH: `PYTHONPATH=$PWD python -m pytest tests/test_file.py -v`
3. Or use the dev script: `./dev.sh test`

## Verification

Each fixed file should be tested to ensure:
1. The test passes with `PYTHONPATH=$PWD python -m pytest`
2. The test passes with `./dev.sh test` (when syntax errors are resolved)
3. No functionality is lost

## Blockers

Currently blocked by syntax error in `pa_core/validators.py` line 148:
```
IndentationError: unexpected indent
```

This prevents testing most pa_core imports until resolved.