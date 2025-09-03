# Portable Windows Zip

This project can generate a self-contained Windows zip that runs without a prior Python install.

## Options

- Source-only archive (any OS):
  - pa-make-zip --output portable_windows.zip
- Windows portable with embeddable Python:
  - pa-make-zip --with-python --python-version 3.12.11 --output portable_windows.zip

## Contents (Windows portable)
- python/ (CPython embeddable runtime)
- Project sources (pa_core/, dashboard/, etc.)
- Launchers:
  - pa.bat (CLI)
  - pa-dashboard.bat (Streamlit)
  - pa-validate.bat
  - pa-convert-params.bat

## Usage
1. Unzip anywhere (e.g., C:\PortableAlpha)
2. Double-click pa-dashboard.bat to open the dashboard
3. Or run pa.bat --config my_first_scenario.yml --index sp500tr_fred_divyield.csv --output Results.xlsx

## Notes
- First run may initialize pip; ensure internet connectivity.
- For offline usage, pre-bundle dependencies in the zip.
- Source-only zips require Python 3.10+ installed and available in PATH.
