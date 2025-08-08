# MIGRATION_NOTES.md
**Subject:** Deprecating Excel/CSV as parameter inputs; keeping Excel/CSV for data uploads only  
Date: 2025-08-08

## Policy
- Parameters: YAML only (Scenario v1 schema).
- Data uploads (index/funds): CSV/XLSX allowed via UI, then persisted to YAML Asset Library.

## Timeline
- Release N: `pa convert --from-xlsx/--from-csv` available; Excel/CSV parameter load paths print deprecation warnings.
- Release N+1: Excel/CSV parameter inputs removed.

## Action items
- Replace any docs/screenshots that show spreadsheet parameters.
- Provide import templates in `/templates`.
