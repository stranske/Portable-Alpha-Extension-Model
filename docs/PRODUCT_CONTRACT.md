# Product contract — stranske/Portable-Alpha-Extension-Model
_First draft generated 2026-09-20 from the audit scorecard; the repo owns this file from now on. A PR that adds a user-facing route, command or page adds a line here. The audit's Phase 1.5 scores every line below and prints any surface not listed as UNSCORED._

## Purpose
Allocate capital across portable-alpha sleeves, simulate returns and financing, and report risk/return results, sweeps and exports.

## Primary journey
Provide config and index → run simulation → inspect sleeve metrics → sweep parameters → export Excel, PNG or board pack.

## Core functions
| id | a user can … and sees … | entry point | probe (how to exercise it; vary these determinants) | status 2026-09-20 |
|---|---|---|---|---|
| F1 | an analyst can run the model and sees per-sleeve return and risk metrics plus run artifacts | `pa run --config ... --index ...`; `run_single(config, index_series)` | vary `theta_extpa` 0.2/0.8; compare metrics and CLI artifacts | PARTIAL |
| F2 | an analyst can run a sweep and sees grid-point summaries and Excel charts | CLI or sweep library/orchestrator | run `alpha_shares` sweep; compare 187 points and ExternalPA returns; test CLI export | PARTIAL |
| F3 | an analyst can export results and sees Excel, PNG or board-pack files | `pa run --png --packet` | exercise Excel, PNG and packet flags under Plotly 7; inspect produced artifacts and logged image-export failure | PARTIAL |
| F4 | an analyst can use the Scenario Wizard and sees configured results, grid and stress views | Streamlit Scenario Wizard → Results | vary theta and alpha share; compare displayed metrics | NOT-EXERCISED |

## Known gaps at draft time
- F1: API metrics respond to theta; Plotly 7 image export is caught and CLI continues without requested image (issue #2287).
- F2: library sweep varies for `alpha_shares`; image export is caught and omitted. `monthly_returns` scales financing by `beta_share` and alpha by `alpha_share`; InternalPA has `beta_share` zero (#2280 fixed).
- F3: no public summary-CSV API exists; CLI image-export failure is caught and omits PNG/PDF (issue #2287).
