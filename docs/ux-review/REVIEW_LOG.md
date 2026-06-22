# UX Review Log — Portable-Alpha-Extension-Model

Diff-anchored record of UX Review (`/ux-review`) passes. Each entry's commit SHA anchors the next
review's git-diff focus. Detailed artifacts live in `Orchestrator/ux_reviews/`.

## 2026-06-22 — Dashboard (`dashboard/app.py`), FULL coverage — commit `98a2250` — overall 2.0/10 (gate FAIL)

- **Coverage:** Home ✓; Asset Library ✓ (upload-only); Portfolio Builder ✓ (upload-only); Scenario Wizard ✓ (5 steps; Run with defaults → infeasible); Results ✓ (bare error); Scenario Grid ✓ (sample → error); Stress Lab ✓ (sample → error); Run Logs ✓ (empty). **NOT driven:** a *successful* run with valid uploaded data — every built-in run path dead-ended.
- **Scores:** wired 2.0 / usability 2.0 / help_clarity 4.0 / workflow 2.0 (4 sev-4 blockers + 1 sev-3, all 4/4).
- **Headline:** all three run entry points dead-end for a first-timer, and the home page recommends two of the broken ones.
- **Findings → filed:**
  - Stress Lab + Scenario Grid bundled-sample paths build `ModelConfig` without required `financing_mode` → **#2018**.
  - Scenario Wizard default capital allocation infeasible (Internal PA 99.8% + margin > total) → **#2020**.
  - Raw developer errors/empty states (Results "Outputs.xlsx not found" `4_Results.py:200`; Run Logs "--log-json" `7_Run_Logs.py:21`; raw pydantic dumps) → **#2021**.
  - Asset Library / Portfolio Builder are upload-only with no bundled-sample path (onboarding gap) — noted; revisit after the run paths work.
- **Next focus:** after #2018/#2020/#2021, re-review driving a *successful* run (Results with real output, charts, export) + confirm the gate.
