# Portfolio Constraints

This document describes the common portfolio constraints that ship with the model,
how they are validated, and how to customize them for your own portfolios.

## Overview

Portfolio constraints are applied to the `Portfolio.weights` mapping and checked
before a simulation runs. The default constraint set is available as
`COMMON_CONSTRAINTS` in `pa_core.portfolio.constraints`.

## Common constraints

The defaults are designed to catch obviously invalid portfolios:

| Constraint | Default | Meaning |
| --- | --- | --- |
| Weight bounds | `min_weight=0.0`, `max_weight=1.0` | Each asset weight must stay within bounds. |
| Leverage | `max_gross_leverage=1.0` | Sum of absolute weights cannot exceed this limit. |
| Concentration | `max_single_weight=0.2`, `max_top_n_weight=0.6`, `top_n=5` | Cap single-name and top-N concentration. |

## Usage guidelines

- Start with the defaults and tighten only when you have a documented policy.
- Keep `max_single_weight <= max_top_n_weight` so top-N caps are consistent.
- If you allow shorting, raise leverage and ensure the weight bounds include negatives.
- Use validation early with `--validate-only` to catch errors without simulation.

## Customizing constraints

You can override defaults in code by constructing a `PortfolioConstraints` value:

```python
from pa_core.portfolio.constraints import (
    ConcentrationConstraint,
    LeverageConstraint,
    PortfolioConstraints,
    WeightBoundsConstraint,
)
from pa_core.validators import ConstraintValidator

constraints = PortfolioConstraints(
    weight_bounds=WeightBoundsConstraint(min_weight=0.0, max_weight=0.5),
    leverage=LeverageConstraint(max_gross_leverage=1.2),
    concentration=ConcentrationConstraint(
        max_single_weight=0.15,
        max_top_n_weight=0.5,
        top_n=4,
    ),
)

validator = ConstraintValidator(constraints)
results = validator.validate({"A": 0.4, "B": 0.3, "C": 0.3})
```

The validator returns detailed error messages with suggestions when a
constraint is violated.
