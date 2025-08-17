# Golden Tests

This directory contains golden tests that validate the core functionality of the Portable Alpha Extension Model by running complete tutorial workflows and validating results.

## Test Structure

### `test_scenario_smoke.py`
- Basic smoke test with fixed simulation parameters
- Validates core orchestrator functionality with deterministic results
- Uses synthetic index data for consistent testing

### `test_tutorial_golden.py`
- Comprehensive tutorial workflow validation
- Tests all major CLI functionality as demonstrated in tutorials
- Validates file generation, metrics calculation, and parameter sweeps

## Test Categories

### Tutorial 1: Parameter Sweeps
- `test_basic_scenario_single_run`: Basic single scenario execution
- `test_returns_mode_parameter_sweep`: Returns sensitivity analysis 
- `test_capital_mode_parameter_sweep`: Capital allocation optimization
- `test_alpha_shares_mode_parameter_sweep`: Alpha/beta split analysis
- `test_vol_mult_mode_parameter_sweep`: Volatility stress testing

### Tutorial 2: Threshold Analysis
- `test_threshold_scenario_analysis`: Risk threshold validation

### Export Functionality
- `test_png_export`: PNG chart generation (Chrome-dependent)
- `test_pptx_export`: PowerPoint export (Chrome-dependent)

### Deterministic Results
- `test_deterministic_run`: Validates identical results with same seed

## Key Features

- **Deterministic**: All tests use fixed seeds (seed=42) for reproducible results
- **Comprehensive**: Validates file existence, size, content structure, and metrics
- **Robust**: Handles optional dependencies gracefully (Chrome for exports)
- **Fast**: Typically completes in under 60 seconds for full suite

## Running Tests

```bash
# Run all golden tests
python -m pytest tests/golden/ -v

# Run specific test category
python -m pytest tests/golden/test_tutorial_golden.py::TestTutorial1ParameterSweeps -v

# Run with environment setup
PYTHONPATH=$PWD python -m pytest tests/golden/ -v
```

## CI Integration

These tests are integrated into the CI pipeline as the `golden-tutorial-tests` job, which:
1. Runs all golden tests to validate functionality
2. Generates tutorial artifacts for reviewer inspection
3. Uploads artifacts (XLSX, PPTX) for 7-day retention
4. Validates the complete tutorial workflow end-to-end

## Expected Outputs

Golden tests validate that the CLI produces:
- Excel files with Summary sheets containing expected agents and metrics
- Multiple scenario results for parameter sweeps
- Deterministic results across repeated runs
- File sizes appropriate for the complexity of analysis

## Maintenance

When updating core functionality:
1. Run golden tests to ensure backward compatibility
2. Update expected values if legitimate changes occur
3. Add new tests for new tutorial scenarios
4. Ensure deterministic behavior is maintained