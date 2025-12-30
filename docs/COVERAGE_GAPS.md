# Coverage Gaps Report

> **Auto-generated reference for prioritizing test coverage improvements.**
> Last updated: 2025-01-27 | Current coverage: 66% | Target: 85%

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Coverage | 66% |
| Tests | 346 passing, 5 skipped |
| Test Runtime | ~68 seconds (with coverage) |
| Modules at 0% | 4 |
| Modules at 100% | 61 |

## Priority Modules (Under 50% Coverage)

These modules should be targeted first for maximum impact:

| Coverage | Module | Missing Lines | Notes |
|----------|--------|---------------|-------|
| 0% | `dashboard/cli.py` | 3-12 | Dashboard CLI entry point |
| 0% | `dashboard/pages/7_Run_Logs.py` | 1-67 | Run logs page |
| 0% | `dashboard/validation_ui.py` | 3-297 | Validation UI (~300 lines) |
| 0% | `pa_core/__main__.py` | 1-127 | Package entry point |
| 6% | `dashboard/pages/3_Scenario_Wizard.py` | 936-1194 | Scenario wizard page |
| 8% | `dashboard/pages/5_Scenario_Grid.py` | 46-368 | Scenario grid page |
| 11% | `dashboard/pages/4_Results.py` | 201 | Results page |
| 14% | `dashboard/pages/2_Portfolio_Builder.py` | 16-113 | Portfolio builder page |
| 27% | `dashboard/pages/6_Stress_Lab.py` | 53-184 | Stress lab page |
| 42% | `pa_core/viz/live.py` | 29-40 | Live visualization |
| 43% | `pa_core/reporting/export_packet.py` | 185-253 | Export packet generation |
| 47% | `pa_core/cli.py` | 1132-1390 | Main CLI module |

## Medium Priority (50-80% Coverage)

| Coverage | Module | Missing Lines |
|----------|--------|---------------|
| 54% | `dashboard/app.py` | 145-164 |
| 56% | `pa_core/sim/covariance.py` | 124-172 |
| 59% | `pa_core/reporting/excel.py` | 233-239 |
| 59% | `pa_core/viz/pdf_report.py` | 31-44 |
| 60% | `pa_core/data/loaders.py` | 83 |
| 60% | `pa_core/viz/metric_selector.py` | 13-16 |
| 62% | `pa_core/viz/grid_heatmap.py` | 36-48 |
| 67% | `dashboard/pages/1_Asset_Library.py` | 231 |
| 67% | `pa_core/viz/utils.py` | 42-50 |
| 68% | `pa_core/data/convert.py` | 25-32 |
| 71% | `pa_core/viz/pptx_export.py` | 36-42 |
| 80% | `pa_core/sim/paths.py` | 227-229 |

## Low Priority (80-99% Coverage)

| Coverage | Module | Missing Lines |
|----------|--------|---------------|
| 84% | `pa_core/sleeve_suggestor.py` | 120 |
| 85% | `pa_core/sweep.py` | 250 |
| 88% | `pa_core/viz/theme.py` | 85-86 |
| 89% | `pa_core/pa.py` | 98-108 |
| 89% | `pa_core/viz/tornado.py` | 30 |
| 90% | `pa_core/viz/fan.py` | 40-41 |
| 91% | `pa_core/sim/metrics.py` | 144 |
| 92% | `pa_core/agents/registry.py` | 46 |
| 92% | `pa_core/presets.py` | 130-133 |
| 92% | `pa_core/viz/capital_treemap.py` | 19 |
| 92% | `pa_core/viz/data_table.py` | 31 |
| 92% | `pa_core/viz/scenario_slider.py` | 13 |
| 93% | `pa_core/data/importer.py` | 239 |
| 93% | `pa_core/logging_utils.py` | 103 |
| 93% | `pa_core/reporting/run_diff.py` | 54-55 |
| 93% | `pa_core/viz/violin.py` | 34 |
| 93% | `pa_core/viz/waterfall.py` | 14 |
| 94% | `pa_core/random.py` | 21 |
| 94% | `pa_core/reporting/sweep_excel.py` | 67-68 |
| 94% | `pa_core/validators.py` | 506 |
| 94% | `pa_core/viz/boxplot.py` | 23 |
| 94% | `pa_core/viz/crossfilter.py` | 20 |
| 94% | `pa_core/viz/factor_bar.py` | 14 |
| 94% | `pa_core/viz/overlay.py` | 24, 31 |
| 95% | `pa_core/backend.py` | 31 |
| 95% | `pa_core/config.py` | 141-142 |
| 95% | `pa_core/manifest.py` | 58-59 |
| 95% | `pa_core/reporting/console.py` | 29 |
| 96% | `pa_core/agents/types.py` | 47 |
| 96% | `pa_core/viz/breach_calendar.py` | 18 |
| 96% | `pa_core/viz/rolling_corr_heatmap.py` | 34 |
| 96% | `pa_core/viz/sunburst.py` | 24 |
| 97% | `pa_core/portfolio/aggregator.py` | 42 |
| 97% | `pa_core/sim/sensitivity.py` | 37 |
| 97% | `pa_core/simulations.py` | 49 |
| 97% | `pa_core/wizard_schema.py` | 281, 283 |
| 98% | `pa_core/data/calibration.py` | 36 |
| 99% | `pa_core/reporting/attribution.py` | 226 |
| 99% | `pa_core/schema.py` | 38 |

## Modules at 100% Coverage (61 modules)

<details>
<summary>Click to expand full list</summary>

- `dashboard/__init__.py`
- `dashboard/glossary.py`
- `dashboard/pages/__init__.py`
- `pa_core/__init__.py`
- `pa_core/agents/__init__.py`
- `pa_core/agents/active_ext.py`
- `pa_core/agents/base.py`
- `pa_core/agents/external_pa.py`
- `pa_core/agents/internal_beta.py`
- `pa_core/agents/internal_pa.py`
- `pa_core/agents/risk_metrics.py`
- `pa_core/data/__init__.py`
- `pa_core/orchestrator.py`
- `pa_core/portfolio/__init__.py`
- `pa_core/reporting/__init__.py`
- `pa_core/run_flags.py`
- `pa_core/sensitivity.py`
- `pa_core/sim/__init__.py`
- `pa_core/stress.py`
- `pa_core/validate.py`
- `pa_core/viz/__init__.py`
- `pa_core/viz/animation.py`
- `pa_core/viz/beta_heatmap.py`
- `pa_core/viz/beta_scatter.py`
- `pa_core/viz/beta_te_scatter.py`
- `pa_core/viz/bookmark.py`
- `pa_core/viz/category_pie.py`
- `pa_core/viz/corr_heatmap.py`
- `pa_core/viz/corr_network.py`
- `pa_core/viz/dashboard_templates.py`
- `pa_core/viz/data_quality.py`
- `pa_core/viz/delta_heatmap.py`
- `pa_core/viz/export_bundle.py`
- `pa_core/viz/exposure_timeline.py`
- `pa_core/viz/factor_matrix.py`
- `pa_core/viz/factor_timeline.py`
- `pa_core/viz/funnel.py`
- `pa_core/viz/gauge.py`
- `pa_core/viz/geo_exposure.py`
- `pa_core/viz/grid_panel.py`
- `pa_core/viz/horizon_slicer.py`
- `pa_core/viz/hover_sync.py`
- `pa_core/viz/html_export.py`
- `pa_core/viz/inset.py`
- `pa_core/viz/milestone_timeline.py`
- `pa_core/viz/moments_panel.py`
- `pa_core/viz/mosaic.py`
- `pa_core/viz/multi_fan.py`
- `pa_core/viz/overlay_weighted.py`
- `pa_core/viz/panel.py`
- `pa_core/viz/parallel_coords.py`
- `pa_core/viz/path_dist.py`
- `pa_core/viz/pdf_export.py`
- `pa_core/viz/quantile_band.py`
- `pa_core/viz/quantile_fan.py`
- `pa_core/viz/radar.py`
- `pa_core/viz/rank_table.py`
- `pa_core/viz/risk_return.py`
- `pa_core/viz/risk_return_bubble.py`
- `pa_core/viz/rolling_panel.py`
- `pa_core/viz/rolling_var.py`
- `pa_core/viz/scatter_matrix.py`
- `pa_core/viz/scenario_play.py`
- `pa_core/viz/scenario_viewer.py`
- `pa_core/viz/seasonality_heatmap.py`
- `pa_core/viz/sharpe_ladder.py`
- `pa_core/viz/spark_matrix.py`
- `pa_core/viz/surface.py`
- `pa_core/viz/surface_animation.py`
- `pa_core/viz/surface_slice.py`
- `pa_core/viz/te_cvar_scatter.py`
- `pa_core/viz/triple_scatter.py`
- `pa_core/viz/weighted_stack.py`
- `pa_core/viz/widgets.py`

</details>

---

## Instructions for Codex Agent

When working on coverage improvements:

1. **Start with 0% modules** - These are untested entry points
2. **Dashboard pages need Streamlit mocking** - Use `pytest-mock` with `st.session_state`
3. **Test file naming**: `test_<module_name>.py` in `tests/` directory
4. **Focus on branches and error paths** - Missing lines often indicate uncovered branches
5. **Time constraint**: Full test suite runs in ~68 seconds locally

### Quick Coverage Check Command
```bash
pytest --cov=pa_core --cov=dashboard --cov-report=term-missing tests/
```

### To Update This File
```bash
pytest --cov=pa_core --cov=dashboard --cov-report=term-missing tests/ 2>&1 | \
  grep -E "^(pa_core|dashboard).*[0-9]+%" | sort -t'%' -k1 -n
```
