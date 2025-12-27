# Portable Alpha Extension Model - GitHub Copilot Instructions

Always follow these instructions first and fallback to additional search and context gathering only if the information here is incomplete or found to be in error.

## Working Effectively

### Bootstrap and Setup (REQUIRED FIRST)
Set up the development environment:
```bash
# Initial setup using dev script (RECOMMENDED)
./dev.sh setup

# Alternative: Manual setup
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

**TIMING**: Setup takes 3-5 minutes initially. NEVER CANCEL during dependency installation.
**CRITICAL**: If pip install fails due to network timeouts, work with existing `.venv` using `PYTHONPATH=/path/to/repo` prefix for commands.

### Build and Test Commands
Run all development checks:
```bash
# Full CI pipeline - takes 15 seconds. NEVER CANCEL.
./dev.sh ci              # OR: source .venv/bin/activate && PYTHONPATH=$PWD ./dev.sh ci

# Individual checks
./dev.sh lint            # Ruff linting - takes <1 second
./dev.sh typecheck       # Pyright type checking - takes 10 seconds  
./dev.sh test            # Pytest test suite - takes 2-3 seconds
```

**TIMING EXPECTATIONS**:
- **Linting**: <1 second - NEVER CANCEL
- **Type checking**: 10 seconds - NEVER CANCEL  
- **Test suite**: 2-3 seconds - NEVER CANCEL
- **Full CI**: 15 seconds total - NEVER CANCEL

**WORKING ENVIRONMENT FIX**: If imports fail, prefix commands with:
```bash
source .venv/bin/activate && PYTHONPATH=/path/to/repository [command]
```

### Run the Application
**CLI Usage** - Core functionality:
```bash
# Basic simulation (takes 16 seconds - NEVER CANCEL)
python -m pa_core.cli --config examples/scenarios/my_first_scenario.yml --index data/sp500tr_fred_divyield.csv --output Results.xlsx

# Parameter sweep modes (takes 4-16 seconds - NEVER CANCEL)
python -m pa_core.cli --config examples/scenarios/my_first_scenario.yml --index data/sp500tr_fred_divyield.csv --mode returns --output ReturnsSweep.xlsx
python -m pa_core.cli --config config/capital_mode_template.csv --index data/sp500tr_fred_divyield.csv --output CapitalSweep.xlsx

# With exports (requires Chrome/Chromium - takes 4-16 seconds - NEVER CANCEL)
python -m pa_core.cli --config examples/scenarios/my_first_scenario.yml --index data/sp500tr_fred_divyield.csv --png --pdf --alt-text "Risk analysis"

# Parameter conversion (takes 1-2 seconds)
python -m pa_core.data.convert config/parameters_template.csv converted_params.yml
```

**Dashboard** - Interactive interface:
```bash
# Launch Streamlit dashboard (starts in 10-15 seconds)
./dev.sh dashboard
# OR manually:
python -m streamlit run dashboard/app.py --server.headless=true --server.port=8501
```

**CRITICAL TIMING NOTES**:
- **Single simulation**: 16 seconds - NEVER CANCEL, set timeout to 30+ seconds  
- **Parameter sweeps**: 4-16 seconds - NEVER CANCEL, set timeout to 30+ seconds
- **Dashboard startup**: 10-15 seconds - NEVER CANCEL, set timeout to 30+ seconds

## Validation

### Mandatory Pre-Commit Checks
Always run before committing:
```bash
./dev.sh ci                    # Full quality checks - takes 15 seconds
python -m pa_core.validate my_config.yml  # Config validation - takes 1 second
```

### Manual Testing Scenarios  
**ALWAYS** test these complete workflows after making changes:

1. **Basic Simulation Workflow**:
   ```bash
   # Test single scenario (16 seconds - NEVER CANCEL)
   python -m pa_core.cli --config examples/scenarios/my_first_scenario.yml --index data/sp500tr_fred_divyield.csv --output Test1.xlsx
   # Verify: Test1.xlsx created with ~160KB size
   ```

2. **Parameter Sweep Workflow**:
   ```bash  
   # Test returns sweep (4 seconds - NEVER CANCEL)
   python -m pa_core.cli --config examples/scenarios/my_first_scenario.yml --index data/sp500tr_fred_divyield.csv --mode returns --output Test2.xlsx
   # Verify: Test2.xlsx created successfully
   ```

3. **Dashboard Integration**:
   ```bash
   # Start dashboard (10-15 seconds - NEVER CANCEL)
   python -m streamlit run dashboard/app.py --server.headless=true --server.port=8501
   # Verify: "You can now view your Streamlit app" message appears
   # Verify: Local URL shows http://localhost:8501
   ```

4. **Configuration Validation**:
   ```bash
   # Test validation (1 second)  
   python -m pa_core.validate templates/scenario_example.yaml
   # Expected: Validation errors for missing fields (this is correct behavior)
   ```

### Export Testing
Test static export functionality:
```bash
# PNG/PDF exports (4 seconds - NEVER CANCEL, requires Chrome/Chromium)
python -m pa_core.cli --config examples/scenarios/my_first_scenario.yml --index data/sp500tr_fred_divyield.csv --png --pdf --output ExportTest.xlsx
# Verify: ExportTest.xlsx and associated PNG/PDF files created
```

**DEPENDENCY CHECK**: Verify Chrome/Chromium available: `which chromium-browser` or `sudo apt-get install chromium-browser`

## Common Tasks

### File Locations and Structure
```
├── pa_core/                 # Main Python package
│   ├── cli.py              # Command-line interface
│   ├── config.py           # Configuration handling  
│   ├── agents/             # Simulation agents
│   ├── data/               # Data loading/conversion
│   ├── reporting/          # Excel/export functionality
│   └── viz/                # Visualization components
├── dashboard/              # Streamlit dashboard
├── tests/                  # Test suite (pytest)
├── config/                 # Configuration templates
├── templates/              # YAML scenario templates  
├── my_first_scenario.yml   # Demo configuration
└── sp500tr_fred_divyield.csv # Demo index data
```

### Configuration Files
- **YAML configs** (single scenarios): `my_first_scenario.yml`, `templates/scenario_example.yaml`
- **CSV templates** (parameter sweeps): `config/capital_mode_template.csv`, `config/returns_mode_template.csv`  
- **Demo files**: `my_first_scenario.csv`, `sp500tr_fred_divyield.csv`, `parameters.csv`

### Development Workflow Commands
```bash
# Quick development cycle
./dev.sh lint                 # <1 second
./dev.sh test                 # 2-3 seconds  
python -m pa_core.cli --config examples/scenarios/my_first_scenario.yml --index data/sp500tr_fred_divyield.csv --output QuickTest.xlsx  # 16 seconds

# Full validation cycle  
./dev.sh ci                   # 15 seconds total
python -m pa_core.cli --config examples/scenarios/my_first_scenario.yml --index data/sp500tr_fred_divyield.csv --mode returns --output ValidationTest.xlsx  # 4 seconds
```

## Troubleshooting

### Common Issues and Solutions

**Import Errors**:
```bash
# If "cannot import name 'RunFlags'" or similar:
source .venv/bin/activate
export PYTHONPATH=/path/to/Portable-Alpha-Extension-Model  
# Then run your command
```

**Network Timeouts During Setup**:
```bash
# Work with existing environment:
source .venv/bin/activate
PYTHONPATH=$PWD python -m [command]
# Dependencies are likely already available
```

**Configuration Errors**:
```bash
# Validate config files first:
python -m pa_core.validate your_config.yml
# Use working templates:
cp my_first_scenario.yml your_config.yml  # For YAML configs
cp config/parameters_template.csv your_params.csv  # For CSV templates
```

**Dashboard Not Starting**:
```bash
# Check Streamlit installation and run manually:
pip list | grep streamlit
python -m streamlit run dashboard/app.py --server.headless=true --server.port=8501
# Allow 10-15 seconds for startup - NEVER CANCEL
```

### Expected Command Results
- **Successful simulation**: Creates Excel file (150-200KB) with multiple sheets
- **Successful dashboard**: Shows "Local URL: http://localhost:8501" message
- **Successful validation**: Shows Pydantic validation errors for incomplete configs (expected)
- **Successful tests**: "X passed in Y.YYs" message, typically 2-3 seconds total

## Key Parameters and Modes

### CLI Analysis Modes
- `--mode capital`: Capital allocation parameter sweep
- `--mode returns`: Return assumption parameter sweep  
- `--mode alpha_shares`: Alpha/beta split parameter sweep
- `--mode vol_mult`: Volatility stress testing

### Required Configuration Elements
- `N_SIMULATIONS`: Monte Carlo trial count (default: 1000)
- `N_MONTHS`: Simulation horizon (default: 12)
- `risk_metrics`: Must include `['Return', 'Risk', 'ShortfallProb']`
- `analysis_mode`: One of `capital|returns|alpha_shares|vol_mult`

### File Format Requirements
- **Index data**: CSV with Date,Return columns (`sp500tr_fred_divyield.csv`)
- **YAML configs**: Complete scenario definitions (`my_first_scenario.yml`)  
- **CSV templates**: Parameter sweep definitions (`config/*_mode_template.csv`)

**NEVER CANCEL REMINDER**: All build, test, and simulation commands complete in under 30 seconds. Set appropriate timeouts and wait for completion.