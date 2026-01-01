# DATA_IMPORT_SPEC.md

## Accepted file types
- `.csv` (UTF-8), `.xlsx` (Excel)

## Accepted shapes
1. **Wide**: columns = Date, <Instrument1>, <Instrument2>, ...
2. **Long**: columns = Date, Id, Return

## Index return CSVs (load_index_returns)
- Required: `Date` column plus a monthly total return column.
- Preferred column names: `Monthly_TR`, then `Return`.
- If neither name is present, the loader falls back to the second column and emits a warning.
- To avoid warnings, label the return column as `Monthly_TR` in new files.

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
