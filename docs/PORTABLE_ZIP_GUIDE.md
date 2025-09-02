# Portable Zip Creation

The `make_portable_zip.py` script creates a clean, runtime-only distribution of the Portable Alpha Extension Model project by filtering out development artifacts.

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