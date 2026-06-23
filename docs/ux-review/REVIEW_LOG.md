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

## 2026-06-23 — Re-test after #2018/#2020/#2021/#2026 — commit `741da9a` — overall 8.0/10 (gate PASS ✅)

- **Diff since `98a2250`:** all dashboard pages + `utils.py` reworked (+358/-64): bundled-sample paths, feasible defaults, friendly empty-states.
- **Coverage (the prior next-focus achieved — a successful run with real output, charts, export):** Home ✓ (onboarding panel + friendly empty-state); Asset Library ✓ (bundled-sample checkbox + template); Scenario Wizard Steps 1–2 ✓ (default allocation now **feasible**); **Stress Lab ✓ full successful run** (Base vs Stressed + deltas + ~30 charts + 6 tables + Excel export); **Scenario Grid ✓ full sweep** (frontier/heatmap + PNG); Results ✓ (friendly empty-state). **Not driven:** Wizard Steps 3–5 + final Run, Portfolio Builder, Run Logs (→ next focus).
- **Scores:** wired 8.5 / usability 7.5 / help_clarity 7.5 / workflow 7.5; **no blockers**; adversarial critic refuted nothing. Panel: claude 7/6/7/7 · codex 8/8/8/8 · cursor 9/7/8/7 · vibe 10/8/7/8.
- **Headline:** dramatic, legitimate recovery from 2.0 → 8.0. All three run entry points that dead-ended last time now succeed end-to-end; raw errors replaced with friendly empty-states; bundled-sample onboarding added.
- **Findings → disposition:** #2018 (financing_mode dead-ends) **FIXED** (Stress Lab + Scenario Grid sample runs succeed); #2020 (infeasible default allocation) **FIXED** (Wizard Step 2 "✅ Capital allocation balanced" + positive margin buffer); #2021 (raw errors) **FIXED** (Results/Home/Stress Lab friendly empty-states); #2026 (upload-only onboarding) **FIXED** (bundled-sample paths). New polish (corroborated, non-blocker) → **filed #2041**: home run-path discoverability ("start here" CTA + disclose where each run's output appears).
- **Next focus:** drive Wizard Steps 3–5 + final Run → Results (confirm the full wizard→Results flow); Portfolio Builder bundled sample (#2027) + Run Logs (#2032); adopt the synced design-system kit (start with #2041).
