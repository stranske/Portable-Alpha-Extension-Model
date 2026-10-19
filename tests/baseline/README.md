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

## Runtime-read and behavioral evidence

The adapter wraps an already-validated config only for the real `run_single`
call. Field access enters a per-run set; bound `model_dump` calls and later
report traversal do not. The manifest combines observations from the funded
control and catalog variants. Catalog membership and config serialization are
not runtime-read evidence.

The financing scenario uses `internal_pa_financing_mean_month` for the funded
InternalPA sleeve and enforces a return decrease in the demo's returns mode.
`internal_financing_mean_month` belongs to margin financing and does not stand
in for the dedicated cost. The focused disconnection regression still sees
the dedicated input read but fails when its actual simulation contribution
is zeroed. No simulation source or golden output is modified by this change.

The generated report separates touched, observed-read and behavioral columns.
The six-correlation scenario establishes joint directional behavior, not
independent numerical influence of each varied field. `active_share` remains
report-only: its config-validation/derived-agent path is outside this recorder,
so an unobserved read is not a claim that the input is ignored globally. The
read gate covers enforced runtime scenarios; report-only unknowns stay visible.
Broad repository coverage remains owned by #2336.

```bash
python -m pytest tests/baseline/test_coverage_manifest.py -q
python -m pytest tests/baseline/test_directional.py -k internal_financing -q
```
