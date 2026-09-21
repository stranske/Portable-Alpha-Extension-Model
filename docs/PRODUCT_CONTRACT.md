# Product contract — stranske/Portable-Alpha-Extension-Model
_First draft generated 2026-09-20 from the audit scorecard; the repo owns this file from now on. A PR that adds a user-facing route, command or page adds a line here. The audit's Phase 1.5 scores every line below and prints any surface not listed as UNSCORED._

## Purpose
Allocate capital across portable-alpha sleeves, simulate returns and financing, and report risk/return results, sweeps and exports.

## Primary journey
Provide config and index → run simulation → inspect sleeve metrics → sweep parameters → export Excel, PNG or board pack.

## Core functions
| id | a user can … and sees … | entry point | probe (how to exercise it; vary these determinants) | status 2026-09-20 |
|---|---|---|---|---|
| F1 | an analyst can run the model and sees per-sleeve return and risk metrics plus run artifacts | `run_single(config, index_series)` (library); CLI `pa run --config <cfg> --index <idx>` reaches `run_single` only when the config's `analysis_mode` is not a sweep mode or `--sensitivity` is passed, otherwise it dispatches to the sweep | vary `theta_extpa` 0.2/0.8; compare metrics and returned artifacts | PARTIAL |
| F2 | an analyst can run a sweep and sees grid-point summaries and Excel charts | CLI or sweep library/orchestrator | run `alpha_shares` sweep; compare 187 points and ExternalPA returns; test CLI export | PARTIAL |
| F3 | an analyst can export results and sees Excel, PNG or board-pack files | `pa run --config <cfg> --index <idx> --png` and, separately, `pa run --config <cfg> --index <idx> --packet` (`--config` is required; `--index` is required unless `--validate-only`) | run each export command on its own under Plotly 7; inspect the produced artifacts and the image-export failure for each | PARTIAL |
| F4 | an analyst can use the Scenario Wizard and sees the configured run on the Results page | Streamlit `pages/3_Scenario_Wizard.py` → `pages/4_Results.py` | vary theta and alpha share in the wizard; compare the Results metrics | NOT-EXERCISED |
| F5 | an analyst can run the Scenario Grid and sees in-page grid and frontier results | Streamlit `pages/5_Scenario_Grid.py` (independent of the wizard) | vary the grid ranges; compare the in-page results | NOT-EXERCISED |
| F6 | an analyst can apply a Stress Lab preset and sees in-page stressed results | Streamlit `pages/6_Stress_Lab.py` (independent of the wizard) | vary the preset; compare the in-page results | NOT-EXERCISED |

## Known gaps at draft time
- F1: API metrics respond to theta; Plotly 7 image export is caught and CLI continues without requested image (issue #2287).
- F2: library sweep varies for `alpha_shares`; the CLI sweep aborts in `export_sweep_results` on the Plotly 7 `engine` TypeError before writing its manifest (#2287), so the sweep export path fails rather than completing without an image. `monthly_returns` scales financing by `beta_share` and alpha by `alpha_share`; InternalPA has `beta_share` zero (#2280 fixed).
- F3: no public summary-CSV API exists; CLI image-export failure is caught and omits PNG/PDF (issue #2287).
