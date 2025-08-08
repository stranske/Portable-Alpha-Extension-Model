# ISSUES_BACKLOG.md

Open these as individual issues. Each should link back to the acceptance criteria in INSTRUCTIONS_FOR_CODEX.md.

1. Schema v1 for YAML parameters (pydantic models, CLI validate)
2. DataImportAgent (CSV/XLSX → monthly returns) + UI mapping
3. CalibrationAgent (μ/σ/ρ) + write Asset Library YAML
4. PortfolioAggregator (aggregate μ/σ and cross-ρ) + tests
5. CovarianceBuilder + PSDProjection (Higham) + tests
6. SimulatorOrchestrator refactor + SleeveAgents invariants
7. RiskMetricsAgent (CVaR, MaxDD, TUW, breaches)
8. Streamlit MVP pages (Home, Asset Library, Portfolio Builder, Wizard, Results)
9. Packaging: console scripts + Win/mac launchers; portable Windows zip
10. Remove Excel/CSV parameter inputs; keep converter for one release
11. CI: golden tutorial tests + artifact upload
12. Docs: three-chapter tutorials wired to app

Assignee: @tim (review), @codex (implementation)
Reviewer: 1 required (Tim)
