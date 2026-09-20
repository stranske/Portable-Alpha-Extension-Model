# Product contract — stranske/Portable-Alpha-Extension-Model
_First draft generated 2026-09-20 from the audit scorecard; the repo owns this file from now on. A PR that adds a user-facing route, command or page adds a line here. The audit's Phase 1.5 scores every line below and prints any surface not listed as UNSCORED._

## Purpose
Allocate capital across portable-alpha sleeves, simulate returns and financing, and report risk/return results, sweeps and exports.

## Primary journey
Provide config and index → run simulation → inspect sleeve metrics → sweep parameters → export Excel, PNG or board pack.

## Core functions
| id | a <user> can … and sees … | entry point | probe (how to exercise it; vary these determinants) | status 2026-09-20 |
|---|---|---|---|---|
| F1 | an analyst can run the model and sees per-sleeve return and risk metrics plus run artifacts | `pa run --config ... --index ...`; `SimulatorOrchestrator(...).run()` | reload YAML with `theta_extpa` 0.2 vs 0.8; compare API metrics and CLI result | PARTIAL |
| F2 | an analyst can run a sweep and sees grid-point summaries and Excel charts | CLI or sweep library/orchestrator | run `alpha_shares` sweep; compare 187 points and ExternalPA returns; test CLI export | PARTIAL |
| F3 | an analyst can export results and sees Excel, PNG, board pack or CSV files | `pa run`, `--png`, `--pptx`, summary CSV API | write summary CSV then exercise CLI export under Plotly 7; inspect artifacts/error | PARTIAL |

## Known gaps at draft time
- F1: API metrics respond to theta, but fresh-environment CLI crashes before metrics because Plotly 7 rejects `engine="kaleido"` (issue #2287).
- F2: library sweep varies for `alpha_shares`, but CLI export crashes on the same Plotly error; InternalPA financing also does not scale with capital share (issue #2280).
- F3: CSV API export works, but CLI Excel/PNG export crashes in image generation with Plotly 7 (issue #2287).
