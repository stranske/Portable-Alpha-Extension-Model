# DATA_IMPORT_SPEC.md

## Accepted file types
- `.csv` (UTF-8), `.xlsx` (Excel)

## Accepted shapes
1. **Wide**: columns = Date, <Instrument1>, <Instrument2>, ...
2. **Long**: columns = Date, Id, Return

## Index return CSVs (load_index_returns)
- Required: `Date` column plus a monthly total return column.
- Required return column name: `Monthly_TR`.
- If the name is missing, the loader raises a ValueError listing expected and available columns.
- Dates are parsed from `Date` (case-sensitive). Provide an explicit date format when needed.

## Required mappings (captured via UI)
- frequency: daily or monthly
- value_type: price or return
- If CSV wide: select the date column and the instrument columns
- If CSV long: select date column, id column, and value column

## Transformations
- price → return: simple return (P_t / P_{t-1} - 1)
- daily → monthly: compound within calendar month
- Annualization in calibration: mean×12, stdev×√12

## Validation
- ≥ 36 monthly observations per instrument by default
- No duplicate dates per instrument
- Missing values handled by forward-fill for prices; rows with NA returns dropped

## Persisted data quality
- `load_index_returns` attaches `series.attrs["data_quality"]` with:
  - `rows_in`: source return rows before numeric coercion and drop filtering.
  - `coerced_non_numeric`: non-empty source values that became missing during numeric coercion.
  - `rows_dropped`: rows excluded from the returned series after value/date drop filtering.
  - `detected_frequency`: the inferred or fallback frequency recorded on the series.
- Simulation manifests persist the index `data_quality` block so downstream readers can distinguish clean inputs from runs with dropped/coerced rows.
- `DataImportAgent` records per-series calibration quality under `data_quality.series.<id>` with `n_obs` and `below_min_obs`.
- The default importer still raises on series below `min_obs`. The calibrate `--input` command enables the additive record-and-filter path so below-threshold series are omitted from estimates while their `n_obs` and `below_min_obs: true` state remains in the asset-library YAML.
