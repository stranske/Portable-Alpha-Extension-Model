# Templates

This folder contains starter templates for the Portable Alpha Extension Model.

## Scenario Templates

- `scenario_example.yaml` - Minimal scenario example for quick testing
- `scenario_template.yaml` - Full template with index, assets, correlations, and sleeve configuration

## Config Templates

- `params_template.yaml` - YAML parameter template generated from the schema
- `parameters_template.csv` - CSV parameter template generated from the schema (legacy format)

## Data Import Templates

- `asset_timeseries_template.xlsx` - Excel template for importing asset time series
- `asset_timeseries_wide_returns.csv` - Wide-format CSV template (columns per asset)
- `asset_timeseries_long_returns.csv` - Long-format CSV template (Date, Id, Return columns)
- Index return CSVs should include a `Date` column and a return column named `Monthly_TR` (preferred) or `Return`.

## Usage

1. Copy the appropriate template to your working directory
2. Modify the values to match your scenario
3. Run validation: `python -m pa_core.validate your_scenario.yaml`
4. Run simulation: `python -m pa_core.cli --config your_scenario.yaml --index data/sp500tr_fred_divyield.csv`

See `docs/UserGuide.md` for detailed configuration options.
