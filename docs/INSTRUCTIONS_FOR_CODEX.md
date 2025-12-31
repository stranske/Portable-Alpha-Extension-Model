# INSTRUCTIONS_FOR_CODEX.md
**How to contribute code safely and quickly**  
Version: draft-2025-08-08

## Ground rules
1. **Inputs**: Parameters are YAML only. Data uploads (index/fund) can be CSV/XLSX and are converted to YAML-backed assets via the UI.
2. **Small PRs**: One feature per PR. Include tests and a short GIF or screenshots.
3. **Determinism**: Fix seeds in tests; assert metric vectors within tolerances.
4. **No silent math**: If projecting to PSD or adjusting inputs, log it and surface it in results.

## Local dev
- Python ≥ 3.11
- `uv` or `pipx` preferred. Typical: `uv venv && source .venv/bin/activate && uv pip install -e .[dev]`
- Run all checks: `./dev.sh ci`

## Agent quickstart
See `CONTRIBUTING.md` for the canonical agent setup steps under "Agent quickstart".

## Branch & PR workflow
- Branch from `main`: `feature/<short-name>`
- Write acceptance criteria in the PR description
- Include: tests, docs snippet, screenshots/GIF, and the exact YAML used
- Labels: `ui`, `engine`, `data`, `docs`, `infra`

## Code style
- Ruff + Black, Mypy where annotated
- Pydantic for config schemas
- Prefer pure functions; keep state at edges (UI, I/O)

## ModelConfig vs Scenario
Use `pa_core/config.py` `ModelConfig` for run-level settings, parameter sweeps, and
capital allocation. Use `pa_core/schema.py` `Scenario` for market data, portfolio
structure, and correlation inputs. They are intentionally separate; pair one
`ModelConfig` with one `Scenario` when running a full simulation.

## Testing
- Unit tests for each agent
- Golden tests:
  - `tests/golden/test_scenario_smoke.py` runs a small scenario (N=500, T=120) with fixed seed and asserts top-line metrics within tolerance
  - Data import tests using the provided templates

## CI expectations
- Pre-commit passes
- Unit + golden tests pass
- Built artifacts (XLSX, PPTX) uploaded for reviewer

## Step-by-step tasks (open these issues and ship in order)

1. **Schema v1 (YAML only for parameters)**
   - Define pydantic models: `Index`, `Asset`, `Portfolio`, `Sleeve`, `Scenario`
   - CLI: `pa validate scenario.yaml`
   - Tests: load/save roundtrip; failure cases for bad sums and missing ρ

2. **DataImportAgent + CalibrationAgent**
   - Streamlit: upload CSV/XLSX, choose wide vs long, map columns, frequency, value_type
   - Transform prices→returns; daily→monthly compounding
   - Calibrate μ/σ and pairwise ρ against chosen index
   - Save to Asset Library YAML
   - Tests: use `/templates` CSV/XLSX here; assert μ/σ/ρ

3. **PortfolioAggregator**
   - Compute sleeve-level aggregates from weights
   - Cross-sleeve ρ and sleeve-vs-index ρ
   - Tests: single-asset equals original; two-asset hand check

4. **CovarianceBuilder + PSDProjection**
   - Assemble covariance for [Index, H, E, AE]; Higham projection if needed
   - Emit warnings and persist adjusted matrix
   - Tests: off-PSD example corrected; deltas bounded

5. **SimulatorOrchestrator + SleeveAgents refit**
   - Wire distribution sampler, sleeves, financing
   - Ensure invariants: financing on beta only
   - Tests: α=0 collapses to beta-only; θ=0 collapses; reproducibility

6. **RiskMetricsAgent**
   - Add CVaR, MaxDD, TimeUnderWater, breach counts
   - Tests: monotonicity and smoke

7. **Streamlit MVP**
   - Pages: Home, Asset Library, Portfolio Builder, Scenario Wizard, Results
   - Simple vs Advanced mode
   - Run history from `/runs` and `Outputs.parquet`
   - One-click exports (XLSX, PPTX, PNG/PDF)

8. **Packaging (desktop)**
   - Console scripts: `pa run`, `pa dashboard`
   - Windows `.bat` and macOS `.command` launchers
   - Portable Windows zip build in release pipeline

9. **Remove Excel/CSV parameter inputs**
   - Keep converter `pa convert ...` for one release; remove after deprecation
   - Update docs and CHANGELOG

## Acceptance criteria snippets

- **Data import**: Given `asset_timeseries_wide.csv`, after mapping and selecting SP500_TR as index, the calibration writes an Asset Library YAML with μ/σ within ±5 bps annualized and pairwise ρ within ±0.02 of golden values.

- **Portfolio aggregation**: For portfolio {A:0.6, B:0.4} with known Σ, aggregated σ matches sqrt(wᵀΣw) to ±1e-6.

- **PSD projection**: Given an intentionally off-PSD ρ, the projected matrix has min eigenvalue ≥ 0 and max|Δρ| < 0.03.

- **MVP UI**: A user can upload CSV, calibrate, select portfolios per sleeve, run N=2000, and export PPTX in < 20 seconds on a laptop.
