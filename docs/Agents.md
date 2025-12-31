# Agents.md
**Project:** Portable Alpha + Active Extension Model  
**Version:** draft-2025-08-08

This document defines the core modules (“agents”), their responsibilities, inputs/outputs, invariants, and test hooks.
It is written to enable small, focused pull requests and predictable reviews.

---

## 0) Architecture overview

**Goal:** simulate a portfolio that combines internal portable alpha, external portable alpha, and an active extension sleeve, 
with financing costs, then report risk/return metrics and board-ready artifacts.

**Design principles**
- Input parameters live in **YAML only**.
- Index/fund data for calibration may be uploaded as **CSV or Excel**, then converted into asset parameters (mean, vol, correlations) recorded back in YAML.
- Default to **aggregate-first** portfolio handling: compute portfolio-level μ/σ/ρ from asset-level inputs before simulation.
- Keep the state small: simulate only [Index, Internal α, External α, Active-Ext α] unless full-path mode is explicitly requested.
- Every run is **reproducible**: save YAML, seed, version, git SHA, and any PSD-adjusted correlation matrices.
- Canonical return path: `pa_core.sim.paths.draw_joint_returns` with `pa_core.sim.params.build_simulation_params` shared by CLI, sweep, and orchestrator.

```
Data (CSV/XLSX) ──► DataImportAgent ──► CalibrationAgent ──► Asset Library (YAML)
         YAML ──► ScenarioLoader ──► PortfolioAggregator ──► CovarianceBuilder + PSDProjection
                                   └─► FinancingAgent
         ↑                                              └─► DistributionSampler ─► SleeveAgents ─► RiskMetrics
         └────────────────────────────────────────────────────────────────────────────────────────┘
                                                         └─► ReportWriter + RunRegistry
```

---

## 1) DataImportAgent
**Purpose:** Accept index or fund time series in CSV/XLSX and hand structured frames to CalibrationAgent.

**Inputs**
- File: `.csv` or `.xlsx`.
- User-provided or UI-captured mapping:
  - `frequency`: daily or monthly
  - `value_type`: `price` or `return`
  - Column mapping:
    - Wide: `date_col` + one column per instrument
    - Long: `date_col`, `id_col`, `value_col`

**Outputs**
- DataFrame with columns: `date`, `id`, `return` (monthly), one row per id/date.
- Metadata captured in `DataImportAgent.metadata` including column mapping,
  source frequency, and any price→return or resampling steps.

**Rules & invariants**
- Dates must be strictly increasing within each id.
- If `price`, compute simple returns; if daily, compound to monthly.
- Handle missing by forward-fill prices; drop incomplete trailing months for returns.
- Validate at least 36 monthly observations by default unless overridden.

**Test hooks**
- Golden file tests for each template (wide CSV, long CSV, XLSX).

---

## 2) CalibrationAgent
**Purpose:** Convert return series to asset parameters.

**Inputs**
- DataFrame: `date`, `id`, `return` (monthly)
- Required anchors: market index id

**Outputs**
- Per-asset: annualized `mu`, `sigma`
- Pairwise correlations over the lookback window
- Optional Student-t `df` if enabled later

**Rules**
- Annualization: mean×12, stdev×√12
- Exclude partial months and assets with < 36 obs by default

**Options**
- `covariance_shrinkage`: `none` | `ledoit_wolf` (shrinks covariance toward the diagonal)
- `vol_regime`: `single` | `two_state` (selects low/high vol based on a recent window)
- `vol_regime_window`: length in months for the recent window

**Trade-offs**
- Shrinkage stabilizes short samples but can mute true cross-asset covariation.
- Two-state regimes track recent volatility but can be noisy with very short windows.

**Test hooks**
- Known synthetic input with deterministic μ/σ/ρ

---

## 3) AssetLibrary
**Purpose:** Persist calibrated assets and correlations to YAML.

**Schema (YAML)**
- `index`: id, label, mu, sigma
- `assets`: list of [id, label, mu, sigma]
- `correlations`: list of [id_a, id_b, rho]

**Rules**
- Symmetry and -0.999 ≤ ρ ≤ 0.999 enforced
- On write, sort ids and pairs for stable diffs

---

## 4) PortfolioAggregator
**Purpose:** Turn user-defined portfolios into aggregate μ/σ and cross-sleeve correlations.

**Inputs**
- `assets` with μ/σ
- `correlations`
- Portfolio weights per sleeve

**Outputs**
- Aggregated μ/σ for each sleeve’s α stream
- Cross-sleeve correlations ρ(α_H, α_E), etc.

**Formulas**
- μ_α = wᵀ μ
- σ_α = sqrt(wᵀ Σ w)
- ρ(α_i, α_j) = (w_iᵀ Σ w_j) / (σ_i σ_j)
- ρ(α_i, Index) = (w_iᵀ Σ w_index) / (σ_i σ_index)

**Tests**
- Single-asset portfolio equals original asset
- Two-asset hand-computable example

---

## 5) CovarianceBuilder + PSDProjection
**Purpose:** Build the covariance for [Index, H, E, AE] and ensure it is positive semidefinite.

**Rules**
- Validate pairwise ρ presence for all required pairs
- If not PSD, project to nearest PSD (Higham) and emit a warning with delta norms
- Persist the adjusted matrix alongside the run

**Test hooks**
- Property test: injected tiny off-PSD gets corrected

---

## 6) DistributionSampler
**Purpose:** Draw joint shocks for the factor set with configurable distribution.

**Modes**
- `normal` (default)
- future: `student_t` with df

**Controls**
- seed, draws N, horizon T

**Tests**
- Reproducibility under fixed seed
- Shape and mean/var sanity checks

---

## 7) FinancingAgent
**Purpose:** Generate financing costs applied to beta component of sleeves.

**v1 (static)**
- Base curve + fixed spread, clipped at ≥ 0

**v2+ (optional)**
- Spread stochasticity and correlation to equity drawdowns

**Tests**
- Zero-financing equals no-cost baseline

---

## 8) SleeveAgents
**Internal Portable Alpha (H)**
- Return: Index + θ_H * α_H on its sleeve capital share
- Financing: applied to beta only

**External Portable Alpha (E)**
- Same as above with θ_E

**Active Extension (AE)**
- Return: Index + active_share * α_AE
- Financing applied to beta leg only
- Borrow/short costs excluded in v1 unless modeled via α stream

**Invariants**
- Sleeve capital shares sum to 1.0
- Financing never applied to α component in v1

**Tests**
- α=0 collapses to pure beta sleeve
- θ=0 collapses to beta-only

---

## 9) SimulatorOrchestrator
**Purpose:** Wire sampler, sleeves, and financing; return panel of portfolio/base returns.
**Canonical return engine:** `pa_core.sim.paths.draw_joint_returns` (used by CLI, sweep, and orchestrator for aligned draws).
**Shared parameter builder:** `pa_core.sim.params.build_simulation_params` keeps return/financing inputs consistent.

**Outputs**
- Paths for Portfolio, Base, individual sleeves
- Metadata: seed, adjusted PSD flag

**Tests**
- TE=0 when all α=0
- Reproducibility with fixed seed

---

## 10) RiskMetricsAgent
**Default metrics**
- Ann. return, vol, Sharpe
- Tracking error vs Base
- Shortfall probability vs threshold
- VaR and CVaR (monthly and annualized)
- Max drawdown and Time Under Water
- Policy breach counts

**Tests**
- Closed-form checks on simple normals
- Monotonicity under scaling

---

## 11) ReportWriter
**Outputs**
- XLSX summary and detail
- PPTX one-pager
- PNG/PDF charts
- Alt-text embedded where feasible

**Rules**
- Include run metadata and config snapshot
- Include PSD-adjustment note if applicable

---

## 12) RunRegistry
**Purpose:** Persist run history for UI

**Artifacts per run**
- `/runs/<run_id>/scenario.yaml`
- `/runs/<run_id>/summary.json`
- exported files
- Indexed by `Outputs.parquet` for quick listing

---

## 13) Streamlit Pages (UI Agents)
- **HomePage**: scenario cards, run history
- **AssetLibraryPage**: upload CSV/XLSX, map columns, preview returns
- **PortfolioBuilderPage**: build portfolios, view aggregate μ/σ/ρ
- **ScenarioWizardPage**: choose portfolios per sleeve, financing, thresholds
- **ResultsPage**: Overview, Risk, Compare, Export

**UX modes**
- Simple (PM/CIO): pick predefined portfolios; minimal controls
- Advanced (Analyst): full schema and editors

---

## 14) Errors and messages
- Human-first phrasing with remediation steps
- PSD projection warning shows max|Δρ| and a link to download adjusted matrix

---

## 15) Testing strategy
- Unit tests per agent
- Scenario golden tests: fixed seed produces known summary vector
- Tutorial tests executed headless produce expected artifacts
- CI uploads XLSX/PPTX for review

---

## 16) Non-goals (v1)
- Full per-asset path simulation for large universes
- On-the-fly optimization beyond small parameter sweeps
