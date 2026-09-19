# Data Cleaning Log

Generated automatically by `src/cleaning.py`. Each row records one
transformation applied to the raw data and how many rows or cells it touched.

| # | Step | Detail | Rows/cells affected |
|---|------|--------|--------------------|
| 1 | Standardise text | `region.Region` — trimmed whitespace and unified casing | 1 |
| 2 | Standardise text | `city.CityName` — trimmed whitespace and unified casing | 19 |
| 3 | Standardise text | `attraction_type.AttractionType` — trimmed whitespace and unified casing | 1 |
| 4 | Standardise text | `transaction.VisitMode` — trimmed whitespace and unified casing (e.g. `' FAMILY '` → `'Family'`) | 1,210 |
| 5 | Reconcile categories | `transaction.VisitMode` — folded variant spellings onto the Mode vocabulary | 21 |
| 6 | Deduplicate | `transaction` — dropped fully identical rows | 192 |
| 7 | Deduplicate | `transaction` — dropped repeated `TransactionId` | 8 |
| 8 | Repair outliers | `transaction.Rating` — values outside [1, 5] set to null | 159 |
| 9 | Repair outliers | `transaction.VisitMonth` — values outside 1–12 set to null | 120 |
| 10 | Referential integrity | `transaction` — removed rows whose `UserId`/`AttractionId` has no matching record in the User/Item tables | 80 |
| 11 | Handle missing | `transaction.Rating` — dropped rows with no rating (target variable; imputing it would leak the imputation rule into the model) | 755 |
| 12 | Handle missing | `transaction.VisitMode` — dropped rows with no visit mode (target variable) | 311 |
| 13 | Handle missing | `transaction.VisitMonth` — imputed with modal value `9` | 117 |
| 14 | Summary | `transaction` — 40,200 raw rows → 38,854 clean rows (1,346 removed, 3.35%) | 38,854 |
| 15 | Integrate | Joined transaction → user → city/country/region/continent and → item → attraction type/city (all LEFT joins, fact grain preserved) | 38,854 |
| 16 | Handle missing | `CityName` — unmatched dimension lookups labelled `'Unknown'` | 442 |
