# PAEM app behavior baseline kit

Scenario-driven wiring/sensibility/regression tests built on the shared
**`baseline_kit`** package (the same core used by Trend_Model_Project and
trip-planner). It drives the real Monte Carlo simulation
(`pa_core.facade.run_single`) off `examples/scenarios/my_first_scenario.yml`.

## Requires

`baseline_kit` must be importable. It lives in `stranske/Workflows` under
`packages/app-baseline-kit`:

```bash
pip install "app-baseline-kit @ git+https://github.com/stranske/Workflows.git#subdirectory=packages/app-baseline-kit"
```

## Layout

```
adapter.py                # load config -> patch inputs -> run_single -> per-agent metrics  (app-specific glue)
catalog.yaml              # directional scenarios + priority inputs (edit this)
invariants.py             # economic/structural bounds -> baseline_kit.InvariantResult
test_golden.py            # golden master of the funded baseline summary (all agents)
test_directional.py       # directional sensibility (variant vs baseline)
test_invariants.py        # invariants across all agents
test_coverage_manifest.py # priority-input coverage -> docs/reports/baseline-coverage.md
```

## Running

```bash
PYTHONHASHSEED=0 pytest tests/baseline/ -n0
PYTHONHASHSEED=0 pytest tests/baseline/test_golden.py -n0 --force-regen   # re-bless
```

## What it checks

- **Directional (enforced)** — confirmed economics on the real simulation:
  doubling in-house alpha vol raises the in-house sleeve's annualized vol;
  doubling in-house mean raises its return; cranking all correlations to 0.9
  (killing diversification) raises Total portfolio vol.
- **Invariants** — across all six agents (Base/ExternalPA/ActiveExt/InternalPA/
  InternalBeta/Total): probabilities in [0,1], vol ≥ 0, annualized return > −1,
  tracking error ≥ 0, max drawdown in [−1, 0].
- **Golden master** — per-agent summary metrics, diffed with tolerance.

## Findings (report-only, to confirm)

Two documented inputs appear **inert** under the demo's `analysis_mode: returns`:

1. **`internal_financing_mean_month`** 0 → 0.004 leaves the in-house return
   unchanged (even with the sleeve funded).
2. **`active_share`** 0.3 → 0.8 changes **zero** of 72 numeric summary cells.

Likely mode-dependent (these knobs may only be live in `analysis_mode`
`alpha_shares`/`capital`) rather than wiring gaps. The directional scenarios for
them are `enforce: false` (skipped, reported) until confirmed. To exercise them,
add scenarios with a base that sets the mode where each knob is live.
```
