# Portable Zip Creation

The portable zip tool creates a clean distribution of the Portable Alpha Extension Model by filtering out development artifacts. On Windows, it can optionally bundle the embeddable CPython runtime and generate launcher scripts so users don't need to install Python.

## Usage

```bash
# Create a basic portable archive
python scripts/make_portable_zip.py

# Specify custom output name
python scripts/make_portable_zip.py --output my_distribution.zip

# Show detailed output (what gets excluded)
python scripts/make_portable_zip.py --verbose

# Add custom exclusion patterns
python scripts/make_portable_zip.py --exclude-pattern "*.log" --exclude-pattern "temp_*"

# Windows only: include embeddable Python and create .bat launchers
pa-make-zip --with-python --python-version 3.12.11 --output portable_windows.zip
```

## What Gets Included

The portable archive includes only essential runtime files:

- **Core Python package** (`pa_core/` directory)
- **Dashboard and web interface** (`dashboard/` directory)  
- **Configuration templates** (`config/`, `templates/` directories)
- **Setup and requirements** (`requirements.txt`, `pyproject.toml`)
- **Sample configurations** (`my_first_scenario.yml`, `sp500tr_fred_divyield.csv`)
- **Documentation** (`README.md`, `docs/`, `tutorials/`)
- **Launch scripts** (`dev.sh`, `scripts/launch_dashboard.*`)

When using `--with-python` on Windows, the archive also includes:

- `python/` (CPython embeddable runtime)
- Launcher batch files in the root:
	- `pa.bat` (CLI)
	- `pa-dashboard.bat` (Streamlit dashboard)
	- `pa-validate.bat`, `pa-convert-params.bat`

## What Gets Excluded

The script automatically excludes common development artifacts:

- **Version control** (`.git/`, `.gitignore`, `.github/`)
- **Python caches** (`__pycache__/`, `*.pyc`, `.pytest_cache/`)
- **Virtual environments** (`.venv/`, `venv/`)
- **Development tools** (`.vscode/`, `.idea/`, linter configs)
- **Build artifacts** (`*.egg-info/`, `htmlcov/`, `docs/_build/`)
- **Development documentation** (testing results, codex files, debug logs)
- **Output files** (`*.xlsx`, `plots/`, temporary files)
- **OS files** (`.DS_Store`, `Thumbs.db`)

## Benefits

Compared to the original implementation that archived the entire project directory:

- **Smaller size**: ~0.25 MB vs potentially 100+ MB (due to excluding `.git`, `.venv`, etc.)
- **Faster creation**: Only processes essential files
- **More secure**: Excludes sensitive development files and git history
- **Cleaner distribution**: Users get only what they need to run the application

## Customization

You can add custom exclusion patterns using `--exclude-pattern`:

```bash
python scripts/make_portable_zip.py --exclude-pattern "*.log" --exclude-pattern "debug_*"
```

This is useful for excluding project-specific temporary files or additional development artifacts.

## Usage (Windows portable)

1. Unzip anywhere (e.g., `C:\PortableAlpha`)
2. Double-click `pa-dashboard.bat` to open the dashboard, or run `pa.bat --help`
3. Example:
	`pa.bat --config my_first_scenario.yml --index sp500tr_fred_divyield.csv --output Outputs.xlsx`

Notes:
- First run may bootstrap pip to install dependencies into the embedded Python.
- For offline environments, pre-bundle wheel files or build the archive in a connected environment and distribute the completed zip.
