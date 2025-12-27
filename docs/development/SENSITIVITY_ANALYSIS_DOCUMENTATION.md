# Sensitivity Analysis Feature

## Overview

The sensitivity analysis feature addresses issue #482 by replacing broad exception handling that silently skipped parameters with comprehensive error logging and reporting. This feature tests how small parameter changes affect portfolio performance metrics.

## Usage

### Command Line Interface

Add the `--sensitivity` flag to any simulation:

```bash
# Basic sensitivity analysis
python -m pa_core.cli --config my_scenario.yml --index returns.csv --output results.xlsx --sensitivity

# With parameter sweep (shows sweep-level sensitivity)
python -m pa_core.cli --config my_scenario.yml --index returns.csv --mode returns --sensitivity
```

## How It Works

### Parameter Perturbation Testing

The sensitivity analysis tests the following parameters with ±5% changes:

- `mu_H`: In-house alpha mean return
- `sigma_H`: In-house alpha volatility  
- `mu_E`: Extension alpha mean return
- `sigma_E`: Extension alpha volatility
- `mu_M`: Market alpha mean return
- `sigma_M`: Market alpha volatility

### Error Handling Enhancement

**Before (Issue #482):**
- Broad exception handling silently skipped failing parameters
- No indication to user which parameters caused issues
- No actionable feedback for troubleshooting

**After (Fixed):**
- Individual parameter failures are logged with specific error messages
- All failed parameters are collected and reported to the user
- Actionable guidance provided for resolving issues
- Successful evaluations continue despite failures

## Output Examples

### Successful Analysis
```
🔍 Running sensitivity analysis...

📊 Sensitivity Analysis Results:
==================================================
📈 mu_H_+5%             | Delta:  +0.2500%
📉 sigma_H_+5%          | Delta:  -0.1200%
📈 mu_E_+5%             | Delta:  +0.1800%
📉 mu_E_-5%             | Delta:  -0.1800%

✅ Sensitivity analysis completed. Evaluated 12 scenarios.
```

### Analysis with Failures  
```
🔍 Running sensitivity analysis...
⚠️  Parameter evaluation failed for mu_H_-5%: Invalid correlation matrix detected
⚠️  Parameter evaluation failed for sigma_E_-5%: Negative volatility not allowed

📊 Sensitivity Analysis Results:
==================================================
📈 mu_H_+5%             | Delta:  +0.2500%
📉 sigma_H_+5%          | Delta:  -0.1200%

⚠️  Warning: 2 parameter evaluations failed and were skipped:
   • mu_H_-5%
   • sigma_E_-5%

💡 Consider reviewing parameter ranges or model constraints.

✅ Sensitivity analysis completed. Evaluated 10 scenarios.
```

## Implementation Details

### Key Components

1. **`_eval()` function**: Evaluates AnnReturn for Base agent with parameter overrides
2. **Error collection**: Tracks failed parameters and specific error messages
3. **Results reporting**: Shows successful deltas and summarizes failures
4. **User guidance**: Provides actionable next steps

### Error Types Handled

- **Validation errors**: Invalid parameter values or ranges
- **Simulation failures**: Runtime errors during Monte Carlo simulation
- **Correlation matrix issues**: Invalid covariance structures
- **Constraint violations**: Parameters outside acceptable bounds

### Integration with Parameter Sweeps

When `--sensitivity` is used with parameter sweep modes (`--mode returns`, etc.):
- Shows parameter sweep results first
- Provides sensitivity analysis of sweep outcomes
- Reports best/worst combination performance ranges

## Testing

### Test Coverage

- **Unit tests**: Core sensitivity function tests (`test_sensitivity.py`)
- **Integration tests**: CLI functionality tests (`test_cli_sensitivity.py`)  
- **Error simulation**: Mock scenarios to test failure logging
- **Argument parsing**: Verification that `--sensitivity` flag is recognized

### Running Tests

```bash
# Run sensitivity-specific tests
python -m pytest tests/test_sensitivity.py -v
python -m pytest tests/test_cli_sensitivity.py -v

# Run full test suite
./dev.sh ci
```

## Migration from Issue #482

This implementation directly addresses the original issue:

> "The broad exception handling silently skips parameters that cause evaluator failures without providing any indication to the user. Consider logging which parameters were skipped or collecting failed parameter names to report back to the caller."

**Solution provided:**
- ✅ Parameter failures are now logged individually with specific error details
- ✅ Failed parameter names are collected and reported in summary
- ✅ Users receive clear indication of what failed and why
- ✅ Actionable guidance helps users resolve parameter issues
- ✅ No more silent skipping - all actions are transparent

## Future Enhancements

Potential improvements for future releases:

1. **Customizable perturbation sizes**: Allow users to specify ±X% changes
2. **Additional parameters**: Extend beyond the core 6 parameters
3. **Sensitivity thresholds**: Highlight parameters with high sensitivity
4. **Export capabilities**: Save sensitivity results to dedicated sheets
5. **Visualization**: Charts showing parameter impact rankings