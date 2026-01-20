# Portfolio Constraints

This guide describes the portfolio-level constraints supported by the validation
helpers in `pa_core.portfolio.constraints`. These checks are intended to catch
infeasible weights early and provide actionable fixes before running a full
simulation.

## Supported constraints

### Weight bounds (per-asset min and max)

Definition: each asset weight must fall within a minimum and maximum bound.

Common presets are provided in `COMMON_WEIGHT_BOUNDS`:

- `long_only`: min 0.0, max 1.0 (no short positions)
- `long_short_130_30`: min -0.30, max 1.30 (up to 30 percent short exposure)

Violation type: `weight_bounds`

Expected detail fields:

- `asset`: asset identifier
- `weight`: current weight
- `min_weight`: minimum allowed weight
- `max_weight`: maximum allowed weight

### Leverage (gross exposure)

Definition: gross exposure is the sum of absolute weights. It must stay under
the configured leverage cap.

Violation type: `leverage`

Expected detail fields:

- `gross_exposure`: sum of `abs(weight)` across assets
- `max_leverage`: maximum allowed gross exposure

### Concentration (single-asset cap)

Definition: no single asset exceeds a concentration limit.

Violation type: `concentration`

Expected detail fields:

- `asset`: asset identifier
- `weight`: current weight (optional but recommended)
- `max_weight`: maximum allowed weight for the asset

## Usage guidelines

- Keep weights normalized so they sum to 1.0 before validation.
- Ensure `min_weight` is less than or equal to `max_weight`.
- Align the leverage cap with your strategy. For example, a long-only portfolio
  typically targets a gross exposure near 1.0, while a 130/30 structure targets
  gross exposure near 1.6.
- Use concentration limits to encourage diversification even when broader
  weight bounds are permissive.
- Run validation without simulation using `pa --validate-only` to surface
  constraint messages early.

## Example: generating constraint fix suggestions

See `examples/portfolio_constraints.py` for a minimal end-to-end example of
constructing a `ConstraintViolation` and generating suggestions with
`suggest_constraint_fixes`.
